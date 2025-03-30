import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch3d.ops import knn_points
from ....misc.slice_iterator import SliceIterator

def multi_head_voxel_to_point_cross_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    weights: torch.Tensor, 
    confidences: torch.Tensor, 
    num_head=1):
    """
    ### Multi head voxel to knn point cross attention
    Note that every voxel has its `k` neighbour points.
    
    input:
        q: [B, V, C]
        k: [B, V, P, C]
        v: [B, V, P, C]
        weights: [B, V, P]
        confidences: [B, V, P]
        
    output: [B, V, C]
    """
    # TODO: check if it works.
    assert q.dim() == 3
    assert k.dim() == v.dim() == 4
    
    b, vox, p, c = v.size()
    
    q = q.view(b, -1, num_head, c // num_head).permute(0, 2, 1, 3)  # [B, N, V, C/N]
    k = k.view(b, -1, p, num_head, c // num_head).permute(0, 3, 1, 4, 2) # [B, N, V, C/N, P]
    v = v.view(b, -1, p, num_head, c // num_head).permute(0, 3, 1, 2, 4) # [B, N, V, P, C/N]
    
    scores = torch.matmul(q.unsqueeze(-2), k).squeeze(-2) * weights.unsqueeze(1) / ((c // num_head) ** 0.5)  # [B, N, V, P]
    attn = torch.softmax(scores, dim=-1) # [B, N, V, P]
    out = torch.matmul(attn.unsqueeze(-2), confidences.view(b, 1, vox, p, 1) * v).squeeze(-2)  # [B, N, V, C/N]

    return out.permute(0, 2, 1, 3).reshape(b, -1, c) # [B, V, C]



class VoxelToPointTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        no_ffn=False,
        ffn_dim_expansion=4
    ):
        super(VoxelToPointTransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target, 
        weights, 
        confidences, 
    ):

        # source: [B, V, C], target: [B, V, P, C]
        # weights: [B, V, P], confidences: [B, V, P]
        query, key, value = source, target, target

        query = self.q_proj(query)  # [B, V, C]
        key = self.k_proj(key)  # [B, V, P, C]
        value = self.v_proj(value)  # [B, V, P, C]

        message = multi_head_voxel_to_point_cross_attention(
            q=query, k=key, v=value, 
            weights=weights, 
            confidences=confidences, 
            num_head=self.nhead
        )

        message = self.merge(message)  # [B, V, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class VoxelToPointTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        no_ffn=False, 
        ffn_dim_expansion=4
    ):
        super(VoxelToPointTransformerBlock, self).__init__()

        self.cross_attn_ffn = VoxelToPointTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            no_ffn=no_ffn, 
            ffn_dim_expansion=ffn_dim_expansion
        )

    def forward(
        self,
        source,
        target,
        weights, 
        confidences
    ):
        # source, target: [B, V, C], [B, V, P, C]
        # weights: [B, V, P], confidences: [B, V, P]
        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            weights, 
            confidences
        )

        return source

def voxel_positional_encoding(ijk: torch.Tensor, d_model: int):
    """
    input:
        ijk: [B, C, L]
        
    output: [B, 6D, L]
    """
    b, c, l = ijk.shape
    i = torch.arange(d_model, device=ijk.device).view(1, 1, d_model, 1) # (1, 1, D, 1)
    ijk = ijk.unsqueeze(-2) # (B, C, 1, L)
    
    pe_sin = torch.sin(ijk / 10000 ** (i / d_model)).reshape(b, c * d_model, l) # (B, C*D, L)
    pe_cos = torch.cos(ijk / 10000 ** (i / d_model)).reshape(b, c * d_model, l) # (B, C*D, L)
    
    return torch.cat((pe_sin, pe_cos), dim=1) # (B, 2CD, L)


def nearest_patch(yx: torch.Tensor, hw: torch.Tensor, patch_size=4):
    """
    ### Compute the nearest patch for every yx inside the 2d space (h*w)
    
    input:
        yx: Tensor(B, N, 2(yx))
        hw: Tensor(B, 2(hw))
        
    output:
        Tensor(B, N, patch_size * patch_size, 3(byx))
    """
    b, n, _ = yx.shape
    y, x = yx[..., 0], yx[..., 1] # (B, N)
    h, w = hw[..., 0].float().unsqueeze(-1), hw[..., 1].float().unsqueeze(-1) # (B, 1)
    # clip to a valid space (y in (0, h-1), w in (0, w-1))
    h_, w_ = h[0, 0], w[0, 0]
    y.masked_fill_(y < 0, 0)
    y.masked_fill_(y > h_ - 1, h_ - 1)
    x.masked_fill_(x < 0, 0)
    x.masked_fill_(x > w_ - 1, w_ - 1)
    
    # create patch
    patch_arange = torch.arange(patch_size, device=yx.device)
    dy, dx = torch.meshgrid(patch_arange, patch_arange) # (ps, ps)
    offset = (patch_size - 1) / 2
    dy, dx = (dy - offset).view(1, 1, -1), (dx - offset).view(1, 1, -1) # (1, 1, ps*ps)
    
    patch_y = y.unsqueeze(-1).round() + dy
    patch_x = x.unsqueeze(-1).round() + dx # (B, N, ps*ps)
    
    # still clip to a valid space
    patch_y.masked_fill_(patch_y < 0, 0)
    patch_y.masked_fill_(patch_y > h_ - 1, h_ - 1)
    patch_x.masked_fill_(patch_x < 0, 0)
    patch_x.masked_fill_(patch_x > w_ - 1, w_ - 1)
    
    # stack byx
    byx = torch.stack((
        torch.meshgrid(
            torch.arange(0, b, device=yx.device), 
            torch.arange(0, n, device=yx.device), 
            torch.arange(0, patch_size * patch_size, device=yx.device)
        )[0], patch_y.int(), patch_x.int()), dim=-1) # (B, N, ps*ps, 3(byx))
    
    return byx



def compute_voxel_interpolate_and_knn_features(
    cnn_features: torch.Tensor, 
    depths: torch.Tensor,
    extrinsics: torch.Tensor, 
    intrinsics: torch.Tensor, 
    voxel_xyz: torch.Tensor, 
    k=16
):
    """
    input:
        cnn_features: [B, C, H, W]
        extrinsics: [B, 4, 4]
        intrinsics: [B, 3, 3]
        voxel_xyz: [B, N, 3]

    output: 
        interpolate_features: [B, C, N]
        knn_features: [B, C, N, K]
        knn_byx: [B, N, K, 3(bhw)]
    """
    b, c, h, w = cnn_features.shape
    n = voxel_xyz.shape[1]
    voxel_xyz = F.pad(voxel_xyz, pad=(0, 1), value=1) # (B, N, 4)
    voxel_xyz = voxel_xyz.permute(0, 2, 1) # (B, 4, N)
    voxel_centers_uvd = torch.matmul(intrinsics, torch.matmul(torch.linalg.inv(extrinsics), voxel_xyz)[:, :3]) # (B, 4, N) -> (B, 3, N)
    voxel_centers_uv = (voxel_centers_uvd[:, :2] / voxel_centers_uvd[:, 2:]).permute(0, 2, 1) # (B, 3, N) -> (B, N, 2)
    # knn features
    if False: # use knn method, slowly
        byx = torch.stack(
            torch.meshgrid(
                torch.arange(0, b, device=cnn_features.device), 
                torch.arange(0, h, device=cnn_features.device), 
                torch.arange(0, w, device=cnn_features.device)), dim=-1) # (B, H, W, 3(bhw))

        yx = byx[..., 1:].view(b, -1, 2) # (B, H*W, 2)
        dist, idx, _ = knn_points(
            p1=voxel_centers_uv.roll(shifts=1, dims=-1), # uv -> vu
            p2=yx.float(),
            K=k
        ) # (B, N, K) (B, N, K)
        
        bidx = torch.stack((
            torch.meshgrid(
                torch.arange(0, b, device=cnn_features.device), 
                torch.arange(0, n, device=cnn_features.device), 
                torch.arange(0, k, device=cnn_features.device)
            )[0], idx), dim=-1)
        # (B, N, K, 2(vidx))
        
        knn_byx = byx.view(b, -1, 3)[bidx[..., 0], bidx[..., 1]] # (B, N, K, 3(vhw))
    
    knn_byx = nearest_patch(
        yx=voxel_centers_uv.roll(shifts=1, dims=-1), 
        hw=torch.tensor([[h, w]], device=cnn_features.device).repeat(b, 1), 
        patch_size=int(math.sqrt(k))
    )
    knn_features = cnn_features[knn_byx[..., 0], :, knn_byx[..., 1], knn_byx[..., 2]].permute(0, 3, 1, 2) # (B, N, K, C) -> (B, C, N, K)
    
    # normalize
    voxel_centers_uv[..., 0] /= ((w - 1) / 2)
    voxel_centers_uv[..., 0] -= 1
    voxel_centers_uv[..., 1] /= ((h - 1) / 2)
    voxel_centers_uv[..., 1] -= 1
    
    interpolated_feat = F.grid_sample(cnn_features, voxel_centers_uv.unsqueeze(-2), padding_mode="border")
    interpolated_depth = F.grid_sample(depths.unsqueeze(1), voxel_centers_uv.unsqueeze(-2), padding_mode="border").view(b, n) # (B, C=1, N, 1) -> (B, N)
    interpolated_dist = torch.abs(interpolated_depth - voxel_centers_uvd[:, 2]) # (B, N)
    return interpolated_feat.view(b, c, -1), interpolated_dist, knn_features, knn_byx, voxel_centers_uvd[:, 2] # (B, C, N), (B, N), (B, C, N, K), (B, N, K, 3(bhw)), (B, N)



class VoxelToPointTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=192,
        nhead=1,
        no_ffn=False, 
        ffn_dim_expansion=4, 
        max_voxels_foreach_processing=1000000
    ):
        super(VoxelToPointTransformer, self).__init__()
        
        assert d_model % 6 == 0 # for positional encoding
        
        self.d_model = d_model
        self.d_model_pe = d_model // 6
        self.nhead = nhead
        self.max_voxels_foreach_processing = max_voxels_foreach_processing
        self.feat_enhancer = nn.Linear(d_model+3+3, d_model) # merge direction and rgb features
        
        self.scale_weights_predictor = nn.Sequential(
            nn.Linear(1, 4), 
            nn.Linear(4, 8), 
            nn.GELU(),
            nn.Linear(8, 16), 
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, d_model),
            nn.GELU()
        )

        self.layers = nn.ModuleList(
            [
                VoxelToPointTransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    no_ffn=no_ffn, 
                    ffn_dim_expansion=ffn_dim_expansion
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def enhance_features(
        self, 
        imgs: torch.Tensor, # (B, 3, H, W)
        features: torch.Tensor, # (B, C, H, W)
        point_xyz: torch.Tensor, # (B, 3, H, W)
        extrinsics: torch.Tensor # (B, 4, 4)
        ):
        
        b, c, h, w = features.shape
        cam_points = extrinsics[:, :3, 3].view(-1, 3, 1, 1) # (B, 3, 1, 1)
        point_to_cam = cam_points - point_xyz # (B, 3, H, W)
        point_to_cam = point_to_cam / torch.norm(point_to_cam, dim=1, keepdim=True) # (B, 3, H, W)
        
        enhanced_features = self.feat_enhancer(
            torch.cat((features, point_to_cam, imgs), dim=1).permute(0, 2, 3, 1).reshape(b*h*w, c+3+3)
        ).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        return enhanced_features

    def forward(
        self,
        imgs: torch.Tensor, 
        cnn_features: torch.Tensor,
        depths: torch.Tensor,
        extrinsics: torch.Tensor, 
        intrinsics: torch.Tensor, 
        point_xyz: torch.Tensor, 
        voxel_xyz: torch.Tensor, 
        confidences: torch.Tensor, 
        voxel_length: torch.Tensor, 
        k=16
    ):
        # Note that `B` is number of views.
        # imgs: [B, 3, H, W]
        # cnn_features: [B, C, H, W]
        # extrinsics: [B, 4, 4]
        # intrinsics: [B, 3, 3]
        # point_xyz: [B, 3, H, W]
        # voxel_xyz: [3, V]
        # point_ijk: [B, 3, H, W]
        # voxel_ijk: [3, V] 
        # voxel_length: Tensor(1)
        # confidences: [B, H, W]
        b, c, h, w = cnn_features.shape
        _, v = voxel_xyz.shape
        assert self.d_model == c
        assert k == 1 or k == 4 or k == 9 or k == 16 or k == 25 or k == 36 or k == 49 # 1^2 to 6^2
        
        if v == 0: return torch.zeros(c, v, device=cnn_features.device)
        
        # enhance features
        cnn_features = self.enhance_features(imgs, cnn_features, point_xyz, extrinsics)
        
        # process voxels for multi times if vixel is too much.
        # and merge feature from all views.
        
        merged_source_slice_list = []
        
        for si in SliceIterator(0, v, self.max_voxels_foreach_processing):
            vi = si.stop - si.start
            voxel_xyz_slice = voxel_xyz[:, si].unsqueeze(0).repeat(b, 1, 1)
        
            interpolated_features, interpolated_dist, knn_features, knn_byx, voxel_depths = compute_voxel_interpolate_and_knn_features(
                cnn_features=cnn_features, 
                depths=depths,
                extrinsics=extrinsics, 
                intrinsics=intrinsics, 
                voxel_xyz=voxel_xyz_slice.permute(0, 2, 1), 
                k=k
            ) # (B, C, V), (B, C, V, K), (B, V, K, 3(bhw))
            
            knn_weights = voxel_length / (torch.norm(point_xyz[knn_byx[..., 0], :, knn_byx[..., 1], knn_byx[..., 2]] \
                - voxel_xyz_slice.permute(0, 2, 1).unsqueeze(-2).repeat(1, 1, k, 1), dim=-1) + 1e-6) # (B, V, K, 3) -> (B, V, K)
            voxel_based_confidences = confidences[knn_byx[..., 0], knn_byx[..., 1], knn_byx[..., 2]] # (B, V, K)
            
            source = interpolated_features.permute(0, 2, 1) # (B, V, C)
            target = knn_features.permute(0, 2, 3, 1) # (B, V, K, C)
            
            # voxel size embedding
            voxel_scale = voxel_length / voxel_depths # (B, V)
            voxel_size_emb = self.scale_weights_predictor(voxel_scale.unsqueeze(-1)) # (B, V, C)
            source *= voxel_size_emb
            target *= voxel_size_emb.view(b, vi, 1, c)
            
        
            for i, layer in enumerate(self.layers):
                source = layer(
                    source,
                    target,
                    knn_weights, 
                    voxel_based_confidences, 
                )
            
            view_weights = torch.softmax(voxel_length / (interpolated_dist + 1e-6), dim=0).unsqueeze(-1) # (B, VI, 1)
            merged_source_slice_list.append(torch.sum(source * view_weights, dim=0)) # (VI, C)
            del source
        
        return torch.cat(merged_source_slice_list, dim=0).transpose_(0, 1) # (C, V)
