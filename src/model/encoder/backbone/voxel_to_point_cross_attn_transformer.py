import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch3d.ops import knn_points


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
        ijk: [B, 3, L]
        
    output: [B, 6D, L]
    """
    b, _, l = ijk.shape
    i = torch.arange(d_model, device=ijk.device).view(1, 1, -1, 1) # (1, 1, D, 1)
    ijk = ijk.unsqueeze(-2) # (B, 3, 1, L)
    
    pe_sin = torch.sin(ijk / 10000 ** (i / d_model)).reshape(b, -1, l) # (B, 3D, L)
    pe_cos = torch.cos(ijk / 10000 ** (i / d_model)).reshape(b, -1, l) # (B, 3D, L)
    
    return torch.cat((pe_sin, pe_cos), dim=1) # (B, 6D, L)


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
    y[y < 0] = 0
    y[y > h - 1] = h - 1
    x[x < 0] = 0
    x[x > w - 1] = w - 1
    
    # create patch
    patch_arange = torch.arange(patch_size, device=yx.device)
    dy, dx = torch.meshgrid(patch_arange, patch_arange) # (ps, ps)
    offset = (patch_size - 1) / 2
    dy, dx = (dy - offset).view(1, 1, -1), (dx - offset).view(1, 1, -1) # (1, 1, ps*ps)
    
    patch_y = y.unsqueeze(-1).round() + dy
    patch_x = x.unsqueeze(-1).round() + dx # (B, N, ps*ps)
    
    # still clip to a valid space
    patch_y[patch_y < 0] = 0
    patch_y[patch_y > h - 1] = h - 1
    patch_x[patch_x < 0] = 0
    patch_x[patch_x > w - 1] = w - 1
    
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
    voxel_centers_uvd = torch.matmul(intrinsics, torch.matmul(extrinsics, voxel_xyz)[:, :3]) # (B, 4, N) -> (B, 3, N)
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
    return interpolated_feat.view(b, c, -1), knn_features, knn_byx



class VoxelToPointTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=192,
        nhead=1,
        no_ffn=False, 
        ffn_dim_expansion=4,
    ):
        super(VoxelToPointTransformer, self).__init__()
        
        assert d_model % 6 == 0 # for positional encoding
        
        self.d_model = d_model
        self.d_model_pe = d_model // 6
        self.nhead = nhead

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

    def forward(
        self,
        cnn_features: torch.Tensor,
        extrinsics: torch.Tensor, 
        intrinsics: torch.Tensor, 
        point_xyz: torch.Tensor, 
        voxel_xyz: torch.Tensor, 
        point_ijk: torch.Tensor, 
        voxel_ijk: torch.Tensor, 
        confidences: torch.Tensor, 
        voxel_length: torch.Tensor, 
        k=16
    ):
        # cnn_features: [B, C, H, W]
        # extrinsics: [B, 4, 4]
        # intrinsics: [B, 3, 3]
        # point_xyz: [B, 3, H, W]
        # voxel_xyz: [B, 3, V]
        # point_ijk: [B, 3, H, W]
        # voxel_ijk: [B, 3, V] 
        # confidences: [B, H, W]
        b, c, h, w = cnn_features.shape
        _, _, v = voxel_xyz.shape
        assert self.d_model == c
        assert k == 1 or k == 4 or k == 9 or k == 16 or k == 25 or k == 36 # 1^2 to 6^2
        
        interpolated_features, knn_features, knn_byx = compute_voxel_interpolate_and_knn_features(
            cnn_features=cnn_features, 
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            voxel_xyz=voxel_xyz.permute(0, 2, 1), 
            k=k
        ) # (B, C, V), (B, C, V, K), (B, V, K, 3(bhw))
        
        knn_weights = 1.0 / torch.norm(point_xyz[knn_byx[..., 0], :, knn_byx[..., 1], knn_byx[..., 2]] \
            - voxel_xyz.permute(0, 2, 1).unsqueeze(-2).repeat(1, 1, k, 1), dim=-1) # (B, V, K, 3) -> (B, V, K)
        
        confidences = confidences[knn_byx[..., 0], knn_byx[..., 1], knn_byx[..., 2]] # (B, V, K)
        
        source = interpolated_features.permute(0, 2, 1) # (B, V, C)
        target = knn_features.permute(0, 2, 3, 1) # (B, V, K, C)
        
        # position encoding
        source = source + voxel_positional_encoding(voxel_ijk, self.d_model_pe).permute(0, 2, 1)
        target = target + voxel_positional_encoding(
            point_ijk[knn_byx[..., 0], :, knn_byx[..., 1], knn_byx[..., 2]].view(b, -1, 3).permute(0, 2, 1), 
            self.d_model_pe
        ).permute(0, 2, 1).reshape(b, v, k, -1)
        
        # Add voxel size encoding
        voxel_size_encoding = voxel_positional_encoding(
            ijk=voxel_length.view(1, 1, 1), 
            d_model=self.d_model // 2
        )
        source = source + voxel_size_encoding.permute(0, 2, 1).repeat(b, v, 1)
        target = target + voxel_size_encoding.permute(0, 2, 1).repeat(b, v*k, 1).view(b, v, k, -1)
        

        for i, layer in enumerate(self.layers):
            source = layer(
                source,
                target,
                knn_weights, 
                confidences, 
            )

        return source.permute(0, 2, 1) # (B, C, V=Voxels)
