import torch
from torch import nn
import torch.nn.functional as F
import math
from .voxel_to_point_cross_attn_transformer import VoxelToPointTransformer
from ..mvsnet.cas_mvsnet_module import CasMVSNetModuleResult, PointCloudResult
from ...types import EncoderOutput, empty_encoder_output
from ...types import IConfigureOptimizers


CHANNEL_RGB = 3
CHANNEL_DELTA_MEANS = 3
CHANNEL_QUATERNION = 4
CHANNEL_SCALE = 3
CHANNEL_OPACITY = 1

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5



class BoundingBox:
    origin: torch.Tensor # 3D Vector (B, 3)
    size: torch.Tensor # scalar (B)
    
    def __init__(
        self, 
        extrinsics: torch.Tensor, # (B, V, 4, 4)
        intrinsics: torch.Tensor, # (B, V, 3, 3)
        nears: torch.Tensor, # (B, V)
        fars: torch.Tensor, # (B, V)
        width: int, 
        height: int):
        b, v, _, _ = extrinsics.shape
        extrinsics = extrinsics.view(b * v, 4, 4)
        intrinsics = intrinsics.view(b * v, 3, 3)
        nears = nears.view(b * v, 1, 1)
        fars = fars.view(b * v, 1, 1)
        
        uv_border = torch.tensor([
            [0, 0, 1], 
            [0, height, 1], 
            [width, 0, 1], 
            [width, height, 1]
        ], device=extrinsics.device, dtype=torch.float32).view(1, 4, 3).permute(0, 2, 1) # (1, 3, 4)
        
        
        far_border = torch.cat(
            (
                torch.matmul(torch.linalg.inv(intrinsics), uv_border) * fars, 
                torch.ones(b * v, 1, 4, device=extrinsics.device)
                ), dim=1
            ) # (1, 4, 4)
        
        near_border = torch.cat(
            (
                torch.matmul(torch.linalg.inv(intrinsics), uv_border) * nears, 
                torch.ones(b * v, 1, 4, device=extrinsics.device)
                ), dim=1
            ) # (1, 4, 4)
        
        border_xyz = torch.matmul(extrinsics, torch.cat((far_border, near_border), dim=-1)) # (B*V, 4, 8)
        
        border_points = border_xyz.view(b, v, 4, -1).permute(0, 2, 1, 3).reshape(b, 4, -1) # (B, 4, V * 8)
        min_point, max_point = border_points.min(dim=-1).values, border_points.max(dim=-1).values
        
        self.origin = min_point[:, :3] # (B, 3)
        self.size = (max_point - min_point).max(dim=-1).values # (B)
        pass
    
    def transform_ndc(self, voxel_center: torch.Tensor, batch: int, xyz_shape: tuple):
        return (voxel_center - self.origin[batch].view(*xyz_shape)) / self.size[batch]
    
    def transform_from_ndc(self, ndc: torch.Tensor, batch: int, xyz_shape: tuple):
        return ndc * self.size[batch] + self.origin[batch].view(*xyz_shape)
    
    def compute_ndc(self, coordinates: torch.Tensor, voxel_size: int):
        """
        ### Mapping cooordinates to normalized voxel center with origin `0` and length `1`.
        mapping (0, 0, 0) -> o',
                (S-1, S-1, S-1) -> 1 - o',
                (S, S, S) -> 1 + o'
                
        note that o' is normalized o.
        
        input:
            `coordinates`: coordinates with shape [N, 3] int
            
        output:
            voxel center with shape [N, 3] float
        """
        return (coordinates / voxel_size) + (0.5 / voxel_size)
    
    def compute_voxel_indices(self, ndc: torch.Tensor, voxel_size: int):
        """
        ### The inverse function of `compute_ndc` with origin `0`
        mapping (0, 0, 0) <- o' (+-o'),
                (S-1, S-1, S-1) <- 1 - o' (+-o'),
                (S, S, S) <- 1 + o' (+-o')
        
        input:
            `ndc`: voxel center with shape [N, 3] float
            
        output:
            coordinates with shape [N, 3] int
        """
        return ((ndc - (0.5 / voxel_size)) * voxel_size).round().int()
    

    pass

class GaussianFeaturesPredictor(nn.Module, IConfigureOptimizers):
    def __init__(self, input_dim, sh_degree):
        super().__init__()
        assert sh_degree < 4
        self.sh_degree = sh_degree
        self.input_dim = input_dim
        
        self.gaussian_scale_min = 0.1
        self.gaussian_scale_max = 10.0
        self.delta_means_activation = lambda x, voxel_size: (torch.sigmoid(x) - 0.5) / voxel_size
        self.scaling_activation = lambda x, voxel_size: (self.gaussian_scale_min + (self.gaussian_scale_max - self.gaussian_scale_min) * torch.sigmoid(x - 5)) / voxel_size
        self.quaternion_activation = lambda x: x
        self.opacity_activation = lambda x: torch.sigmoid(x)
        self.activated_shs = [lambda x : x] + [
            lambda x: x / (5 ** (i + 1)) for i in range(1, sh_degree + 1)
        ]
        
        def make_predictor(out_dim):
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4 if input_dim // 4 > out_dim else out_dim),
                nn.ReLU(),
                nn.Linear(input_dim // 4 if input_dim // 4 > out_dim else out_dim, out_dim)
            )
        
        self.delta_means_predictor = make_predictor(CHANNEL_DELTA_MEANS)
        self.quaternion_predictor = make_predictor(CHANNEL_QUATERNION)
        self.scale_predictor = make_predictor(CHANNEL_SCALE)
        self.opacity_predictor = make_predictor(CHANNEL_OPACITY)
        self.shs_predictor = nn.ModuleList([
            make_predictor(CHANNEL_RGB * (2 * i + 1)) for i in range(sh_degree + 1)
        ])
        
        # init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        pass
    

        
    def forward(self, feature: torch.Tensor, voxel_center: torch.Tensor, voxel_size: int, bbox: BoundingBox, batch: int) -> EncoderOutput:
        # input / output feature (N, C)
        delta_means = self.delta_means_predictor(feature)
        quaternion = self.quaternion_predictor(feature)
        scales = self.scale_predictor(feature)
        opacity = self.opacity_predictor(feature)
        shs = [sh_predictor(feature) for sh_predictor in self.shs_predictor]
        activated_shs = [sh_activation(sh) for sh, sh_activation in zip(shs, self.activated_shs)]
        
        
        activated_delta_means = self.delta_means_activation(delta_means, voxel_size)
        activated_quaternion = self.quaternion_activation(quaternion)
        activated_scales = self.scaling_activation(scales, voxel_size)
        activated_opacities = self.opacity_activation(opacity)
        activated_shs = torch.cat(activated_shs, dim=-1)
         
        # convert means and scales from ndc space to real world.
        means: torch.Tensor = bbox.transform_from_ndc(activated_delta_means + voxel_center, batch, xyz_shape=(1, 3)) # (N, 3)
        scales: torch.Tensor = activated_scales * bbox.size[batch]
        rotations: torch.Tensor = activated_quaternion
        harmonics: torch.Tensor = activated_shs # (N, 3*d^2)
        opacities: torch.Tensor = activated_opacities # (N, 1)
        
        b, dim, d_sh = 1, 3, (self.sh_degree + 1) ** 2
        n, c = feature.shape
        gaussians = EncoderOutput(
            means=means.view(b, n, dim), 
            scales=scales.view(b, n, dim),  # (B, N, 3)
            rotations=rotations.view(b, n, 4), 
            harmonics=harmonics.view(b, n, d_sh, 3).transpose(2, 3), # note that (d_sh, 3) in features
            opacities=opacities.view(b, n)
        )
        
        return gaussians
        
    def configure_optimizers(self, cfg):
        return [
            {'params': self.delta_means_predictor.parameters(), 'lr': cfg.delta_means_lr}, 
            {'params': self.quaternion_predictor.parameters(), 'lr': cfg.quaternion_lr}, 
            {'params': self.scale_predictor.parameters(), 'lr': cfg.scale_lr}, 
            {'params': self.opacity_predictor.parameters(), 'lr': cfg.opacity_lr}
        ] + [  # add more shs
            {'params': shs.parameters(), 'lr': lr} for shs, lr in zip(self.shs_predictor, cfg.shs_lr)
        ]
    pass

    
def flat_3d_coordinates(coor: torch.Tensor, little_endian=False):
    """
    ### Flat 3d coordinates into 1d-tensor. 
    #### Note that max element < 10000.
    
    input:
        `coor`: Tensor(*B, N, 3)
    output:
        Tensor(*B, N, dtype=long)
    """
    coor_l = coor.long()
    if little_endian: return coor_l[..., 0] + coor_l[..., 1] * 10000 + coor_l[..., 2] * 100000000
    else: return coor_l[..., 2] + coor_l[..., 1] * 10000 + coor_l[..., 0] * 100000000
    
def isin_3d_coordinates(coor_1: torch.Tensor, coor_2: torch.Tensor, return_inverse=False):
    """
    ### Compute whether the element of `coor_1` is in `coor_2`. Ensure: 1. dtype=int; 2. max element < 10000;
    
    input:
        `coor_1`: [N1, 3] int
        `coor_2`: [N2, 3] int
    output:
        [N1] bool
        [N2] bool if `return_inverse`
    """
    coor_1_flat = flat_3d_coordinates(coor_1)
    coor_2_flat = flat_3d_coordinates(coor_2)
    
    if return_inverse:
        return torch.isin(coor_1_flat, coor_2_flat), torch.isin(coor_2_flat, coor_1_flat)
    
    return torch.isin(coor_1_flat, coor_2_flat)


def create_local_coordinates(voxel_size: int, last_voxel_size: int = 0, last_coordinates: torch.Tensor = None):
    """
    input:
        `last_coordinates`: None or [N, 3]
        
    output:
        [N', 3]
    """
    if last_coordinates == None:
        # create dense voxel coordinates
        index_candidates = torch.arange(voxel_size, dtype=torch.int, device="cuda")
        x, y, z = torch.meshgrid(index_candidates, index_candidates, index_candidates)
        return torch.stack((x, y, z), dim=-1).view(-1, 3)
    
    # create sparse voxel coordinates from last level
    assert voxel_size % last_voxel_size == 0
    seg_times = voxel_size // last_voxel_size
    seg_times_candidates = torch.arange(seg_times, dtype=torch.int, device="cuda")
    d_x, d_y, d_z = torch.meshgrid(seg_times_candidates, seg_times_candidates, seg_times_candidates)
    d_grid = torch.stack((d_x, d_y, d_z), dim=-1).view(-1, 3) # (seg_times^3, 3)
    
    result = (last_coordinates * seg_times).view(-1, 1, 3) + d_grid # (N, seg_times^3, 3)
    return result.view(-1, 3).unique(sorted=True, dim=0) # use unique to sort index



def combine_batch_gaussians(batch_gaussians: list[EncoderOutput]) -> EncoderOutput:
    b, dim = 1, 3
    gaussian_size = 0
    means_list, scales_list, rotations_list, harmonics_list, opacities_list = [], [], [], [], []
    append_size_list = []
    for gaussian in batch_gaussians:
        if gaussian.opacities.shape[1] > gaussian_size:
            gaussian_size = gaussian.opacities.shape[1]
            
    for gaussian in batch_gaussians:
        append_size = gaussian_size - gaussian.opacities.shape[1]
        if append_size > 0: 
            gaussian.means = torch.cat((gaussian.means, torch.zeros(b, append_size, dim, device=gaussian.means.device)), dim=1)
            gaussian.scales = torch.cat((gaussian.scales, torch.zeros(b, append_size, dim, device=gaussian.means.device)), dim=1)
            gaussian.rotations = torch.cat((gaussian.rotations, torch.zeros(b, append_size, 4, device=gaussian.means.device)), dim=1)
            gaussian.harmonics = torch.cat((gaussian.harmonics, torch.zeros(b, append_size, 3, gaussian.harmonics.shape[-1], device=gaussian.means.device)), dim=1)
            gaussian.opacities = torch.cat((gaussian.opacities, torch.zeros(b, append_size, device=gaussian.means.device)), dim=1)
        means_list.append(gaussian.means)
        scales_list.append(gaussian.scales)
        rotations_list.append(gaussian.rotations)
        harmonics_list.append(gaussian.harmonics)
        opacities_list.append(gaussian.opacities)
        append_size_list.append(append_size)
        
    combined_gaussian = EncoderOutput(
        means = torch.cat(means_list, dim=0), 
        scales = torch.cat(scales_list, dim=0), 
        rotations = torch.cat(rotations_list, dim=0), 
        harmonics = torch.cat(harmonics_list, dim=0), 
        opacities = torch.cat(opacities_list, dim=0)
    )
    
    combined_gaussian.others["append_size_list"] = append_size_list
    return combined_gaussian

def voxel_down_sample(pcd: torch.Tensor, voxel_indices: torch.Tensor, need_sort=True):
    """
    input:
        pcd: [N, C]
        voxel_indices: [N, 3(ijk)]
        
    output:
        downsampled_pcd: [N', C]
        unique_voxel_indices: [N', 3]
    """
    if need_sort:
        flat_voxel_indices = flat_3d_coordinates(voxel_indices)
        indices = flat_voxel_indices.sort().indices
        voxel_indices = voxel_indices[indices]
        pcd = pcd[indices]
    
    unique_voxel_indices, counts = voxel_indices.unique(dim=0, return_counts=True) # (N', 3), (N)
    
    cum_pcd, cum_counts = torch.cumsum(pcd, dim=0), torch.cumsum(counts, dim=0) # (N, C), (N)
    # Add zero to end for the index of first element
    cum_pcd, cum_counts = F.pad(cum_pcd, (0, 0, 0, 1)), F.pad(cum_counts, (0, 1)) # (N+1, C), (N+1)
    # compute the first and the last index
    last_idx = cum_counts - 1
    first_idx = last_idx.roll(shifts=1)
    
    downsampled_pcd = (cum_pcd[last_idx] - cum_pcd[first_idx])[:-1] / counts.unsqueeze(-1)
    
    return downsampled_pcd, unique_voxel_indices


@torch.no_grad()
def downsample_pcd(
    xyz_ndc: torch.Tensor, 
    rgb: torch.Tensor, 
    voxel_size_list: list[int], 
    bbox: BoundingBox, 
    batch_idx: int):
    """
    ### Downsample and classify point cloud to multi-scale voxels
    Note that only the finest scale downsampled points will be classified.
    
    output: 
        downsampled_pcd_list: [(n, iiixyzrgb) * 3] from max to min
        classified_pcd_list: [(n, iiixyzrgb) * 3]

    """
    xyzrgb = torch.cat((xyz_ndc, rgb), dim=1)

    downsampled_pcd_list = [] 
    
    for voxel_size in voxel_size_list:
        voxel_indices = bbox.compute_voxel_indices(xyz_ndc, voxel_size)
        downsampled_xyzrgb, downsampled_voxels = voxel_down_sample(xyzrgb, voxel_indices)
        downsampled_pcd_list.append(torch.cat((downsampled_voxels, downsampled_xyzrgb), dim=1))
    
    return downsampled_pcd_list # [(n, iiixyzrgb) * 3], [(n, iiixyzrgb) * 3] on GPU


def compute_max_scale_voxel_existence_coordinates_by_pcd(max_downsampled_pcd: torch.Tensor):
    delta = torch.arange(3, device="cuda")
    dxs, dys, dzs = torch.meshgrid(delta, delta, delta)
    dxs, dys, dzs = (dxs - 1).view(-1, 1, 1), (dys - 1).view(-1, 1, 1), (dzs - 1).view(-1, 1, 1)
    
    d_coordinates = torch.cat((dxs, dys, dzs), dim=-1) # (27, 1, 3)
    
    max_downsampled_pcd_coordinates = max_downsampled_pcd[:, :3].floor().int() # (N, 3)
    
    return (max_downsampled_pcd_coordinates + d_coordinates).reshape(-1, 3).unique(dim=0) # (27, N, 3) -> (N', 3)
    


def compute_struct_loss(
    downsampled_pcd: torch.Tensor, 
    scale_idx: int, 
    local_coordinates: torch.Tensor, 
    single_point_coordinates: torch.Tensor, 
    gaussians: EncoderOutput, 
    voxel_size_list: list[int], 
    bbox: BoundingBox, 
    batch: int):
    """
    ### Compute L_struct for a given scale.
    input:
        downsampled_pcd: [n1, 9(iiixyzrgb)]
        local_coordinates: [N, 3]
        single_point_coordinates: [N', 3]
        
    output:
        existence_loss, offset_loss, color_loss
    
    #### Note that assume coordinates contains all points.
    #### Note that the downsampled pcd must align the gaussians
    """
    # Note that Gaussians are no longer in ndc space! we should convert means into ndc space.
    get_opacity = lambda mask: gaussians.opacities[mask.unsqueeze(0)]
    get_means = lambda mask: bbox.transform_ndc(gaussians.means[mask.unsqueeze(0)], batch, xyz_shape=(1, 3)) # (N, 3)
    get_color = lambda mask: SH2RGB(gaussians.harmonics[mask.unsqueeze(0)].reshape(-1, 3, gaussians.harmonics.shape[-1])[..., 0])
    
    may_exist_coordinates = torch.cat((single_point_coordinates, downsampled_pcd[:, :3].int()), dim=0).unique(dim=0) # (N', 3)

    may_exist_voxels_mask = isin_3d_coordinates(local_coordinates, may_exist_coordinates)
    must_exist_voxels_mask, must_exist_points_mask = isin_3d_coordinates(local_coordinates, downsampled_pcd[:, :3].int(), return_inverse=True)
    must_accurate_voxels_mask = isin_3d_coordinates(single_point_coordinates, downsampled_pcd[:, :3].int())
    # compute existence loss, offset loss and color loss
    existence_loss, existence_n = 0., 0
    
    # for case 2
    # Apply a "soft regression loss," 
    # i.e., a Gaussian opacity of 1 for voxels where the point is present and 0 for voxels where the point is absent, 
    # and set the loss based on the distance weight.
    must_empty_voxels_mask = torch.logical_not(may_exist_voxels_mask)
    inaccurate_voxels_mask = torch.logical_not(must_accurate_voxels_mask)
    inaccurate_voxel_num = inaccurate_voxels_mask.sum()
    
    # Due to the characteristics of LoD, for non-minimum resolution voxels, 
    # the distance from the voxel to the nearest point can be estimated based on the voxel size. 
    # Because there must be a voxel next to the voxel, we estimate that the distance is voxel length.
    dist_weight = voxel_size_list[0] / voxel_size_list[scale_idx]
    
    existence_loss += (1. - get_opacity(must_exist_voxels_mask)).sum()
    existence_loss += (get_opacity(must_empty_voxels_mask) * dist_weight ** 2).sum()
    existence_n += (must_exist_points_mask.sum() + must_empty_voxels_mask.sum())
    
    predicted_means: torch.Tensor = get_means(must_exist_voxels_mask)
    predicted_color = get_color(must_exist_voxels_mask)
    exist_points = downsampled_pcd[must_exist_points_mask]
    
    offset_loss = (exist_points[:, 3:6] - predicted_means).norm(dim=1).sum() * voxel_size_list[scale_idx] / math.sqrt(3) # devide diagonal length to normalize
    offset_loss += inaccurate_voxel_num
    color_loss = (exist_points[:, 6:] - predicted_color).norm(p=1, dim=1).sum() / 3
    color_loss += inaccurate_voxel_num
    offset_n = color_n = must_exist_points_mask.sum() + inaccurate_voxel_num
    
    # loss normalization
    if existence_n != 0: existence_loss /= existence_n
    if offset_n != 0: offset_loss /= offset_n
    if color_n != 0: color_loss /= color_n
        
    return existence_loss, offset_loss, color_loss
    

def identify_is_current_scale(point_coordinates: torch.Tensor, local_coordinates: torch.Tensor):
    """
    Calculate the number of points in the voxel, 
    if greater than 1, split into octree(or more), reserved for next resolution, 
    otherwise add the voxel index to current resolution.
    
    
    Note that either `point_coordinates` and `local_coordinates` is great than 0 and at the same resolution.
    
    input:
        point_coordinates: Tensor(N1, 3(iii))
        local_coordinates: Tensor(N2, 3)
        
    output: 
        logical_not(multi_points_mask): Tensor(N2) with type `bool`, which is `True` if <= 1 points inside the voxel
        single_point_mask: Tensor(N2) with type `bool`, which is `True` if == 1 points inside the voxel
    """
    # use int64 to avoid data overflow
    point_coordinates_flat = flat_3d_coordinates(point_coordinates)
    voxel_coordinates_flat = flat_3d_coordinates(local_coordinates)
    
    has_point_coordinates_flat, counts = point_coordinates_flat.unique(return_counts=True) # (N3)
    
    multi_points_coordinates_flat = has_point_coordinates_flat[counts > 1]
    single_point_coordinates_flat = has_point_coordinates_flat[counts == 1]
    
    multi_points_mask = torch.isin(voxel_coordinates_flat, multi_points_coordinates_flat) # (N2) True if > 1 points inside
    single_point_mask = torch.isin(voxel_coordinates_flat, single_point_coordinates_flat) # (N2) True if == 1 points inside
    
    return torch.logical_not(multi_points_mask), single_point_mask

class VoxelizedGaussianAdapterModule(nn.Module, IConfigureOptimizers):

    def __init__(self, transformer: VoxelToPointTransformer, feature_channels=192, voxel_size_list=[32, 128, 512], patch_size_list=[3, 2, 1], sh_degree=3) -> None:
        super().__init__()
        self.transformer = transformer
        self.voxel_size_count = len(voxel_size_list)
        self.voxel_size_list = voxel_size_list
        self.patch_size_list = patch_size_list
        self.sh_degree = sh_degree
        assert len(patch_size_list) == len(voxel_size_list)

        self.gaussian_features_predictor = GaussianFeaturesPredictor(input_dim=feature_channels, sh_degree=sh_degree)
        
        pass
    
    def configure_optimizers(self, cfg):
        return self.gaussian_features_predictor.configure_optimizers(cfg)
        
    def forward(self, imgs: torch.Tensor, cnn_features: torch.Tensor, cas_module_result: CasMVSNetModuleResult, img_masks: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, nears: torch.Tensor, fars: torch.Tensor):
        b, v, c, h, w = cnn_features.shape
        far = fars[0, 0]
        is_trainning = cnn_features.grad_fn != None
        batch_gaussians = []
        batch_losses = [[], [], [], []] # total_existence_loss, total_current_loss, total_offset_loss, total_color_loss
        prob_pcd = cas_module_result.registed_prob_pcd
        
        bbox = BoundingBox(
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            nears=nears, 
            fars=fars, 
            width=w, height=h)

        depths = torch.stack([res.backbone["depth"] for res in cas_module_result.ref_view_result_list], dim=1) # (B, V, H, W)

        for batch in range(b):
            # for every batch the number of gaussian may be different (LoD)
            local_coordinates = None
            last_voxel_size = 0
            total_existence_loss, total_current_loss, total_offset_loss, total_color_loss = torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda")
            gaussians = empty_encoder_output(d_sh=(self.sh_degree + 1) ** 2)
            gaussians.others["scales"] = torch.zeros(b, 0, 3, device=cnn_features.device)
            extrinsic_ndc = extrinsics[batch].clone()
            extrinsic_ndc[:, :3, 3] = bbox.transform_ndc(extrinsic_ndc[:, :3, 3], batch, xyz_shape=(1, 3))
            depth_ndc = depths[batch] / bbox.size[batch]
                
            prob_pcd_xyz = prob_pcd.vertices[batch, :, :3] # (V, 3, H, W)
            prob_pcd_rgb = imgs[batch] # (V, 3, H, W)
            prob_pcd_xyz_ndc = bbox.transform_ndc(prob_pcd_xyz, batch, xyz_shape=(1, 3, 1, 1))
            prob_pcd_xyz_ndc_reshaped = prob_pcd_xyz_ndc.permute(0, 2, 3, 1)[img_masks[batch]] # (N, 3)
            prob_pcd_rgb_reshaped = prob_pcd_rgb.permute(0, 2, 3, 1)[img_masks[batch]] # (N, 3)
            max_resolution_voxel_size = self.voxel_size_list[-1]
            max_resolution_prob_pcd_xyz, max_resolution_prob_pcd_indices = voxel_down_sample(
                prob_pcd_xyz_ndc_reshaped, 
                bbox.compute_voxel_indices(
                    ndc=prob_pcd_xyz_ndc_reshaped[:, :3], 
                    voxel_size=max_resolution_voxel_size # max resolution 
                )
            ) # (N', 6), (N', 3)
            
            if is_trainning:
                with torch.no_grad():
                    prob_pcd_geo_mask_reshaped = prob_pcd.vertices_geometry_mask[batch][img_masks[batch]] # (N)
                    all_rectified_xyz_ndc = torch.cat((
                        bbox.transform_ndc(cas_module_result.registed_pcd.xyz_batches[batch][:, :3], batch, xyz_shape=(1, 3)), 
                        prob_pcd_xyz_ndc_reshaped[prob_pcd_geo_mask_reshaped]
                    ), dim=0) # (N'', 3)
                    all_rectified_rgb = torch.cat((
                        cas_module_result.registed_pcd.rgb_batches[batch], 
                        prob_pcd_rgb_reshaped[prob_pcd_geo_mask_reshaped]
                    ), dim=0) # (N'', 3)
                    downsampled_pcds = downsample_pcd(
                        xyz_ndc=all_rectified_xyz_ndc, 
                        rgb=all_rectified_rgb, 
                        voxel_size_list=self.voxel_size_list, 
                        bbox=bbox, 
                        batch_idx=batch
                    )
                
            for scale_idx in range(self.voxel_size_count):
                # TODO: Create multi-scale voxel according to points.
                voxel_size = self.voxel_size_list[scale_idx]
                local_coordinates = create_local_coordinates(
                    voxel_size=voxel_size, 
                    last_voxel_size=last_voxel_size, 
                    last_coordinates=local_coordinates
                )
                
                point_coordinates = max_resolution_prob_pcd_indices * voxel_size // max_resolution_voxel_size
                
                is_current_scale, is_single_point = identify_is_current_scale(
                    point_coordinates=point_coordinates, 
                    local_coordinates=local_coordinates
                )
                single_point_coordinates = local_coordinates[is_single_point]
                
                # update current local coordinates
                next_coordinates = local_coordinates[torch.logical_not(is_current_scale)]
                local_coordinates = local_coordinates[is_current_scale]
                # compute ndc
                centers_ndc = bbox.compute_ndc(local_coordinates, voxel_size)
                
                voxel_feature: torch.Tensor = self.transformer.forward(
                    cnn_features=cnn_features[batch], # (V, C, H, W)
                    depths=depth_ndc,
                    extrinsics=extrinsic_ndc, 
                    intrinsics=intrinsics[batch], 
                    point_xyz=prob_pcd_xyz_ndc, # (V, 3, H, W)
                    voxel_xyz=centers_ndc.transpose(0, 1), # (3, N)
                    confidences=prob_pcd.vertices_confidence[batch],  # (V, H, W)
                    voxel_length=torch.tensor(1 / voxel_size, device=cnn_features.device), 
                    k=self.patch_size_list[scale_idx]
                ) # (C, N)
                
                current_gaussians = self.gaussian_features_predictor.forward(
                    feature=(voxel_feature).transpose(0, 1), 
                    voxel_center=centers_ndc,
                    voxel_size=voxel_size, 
                    bbox=bbox, 
                    batch=batch) # (N, 15)
                
                if is_trainning:
                    # compute losses
                    existence_loss, offset_loss, color_loss = compute_struct_loss(
                        downsampled_pcd=downsampled_pcds[scale_idx], 
                        scale_idx=scale_idx, 
                        local_coordinates=local_coordinates, 
                        single_point_coordinates=single_point_coordinates, 
                        gaussians=current_gaussians, 
                        voxel_size_list=self.voxel_size_list, 
                        bbox=bbox, 
                        batch=batch
                    )
                    total_existence_loss += existence_loss
                    total_offset_loss += offset_loss
                    total_color_loss += color_loss
                
                # Append current gaussians
                gaussians.means = torch.cat((gaussians.means, current_gaussians.means), dim=1)
                gaussians.scales = torch.cat((gaussians.scales, current_gaussians.scales), dim=1)
                gaussians.rotations = torch.cat((gaussians.rotations, current_gaussians.rotations), dim=1)
                gaussians.harmonics = torch.cat((gaussians.harmonics, current_gaussians.harmonics), dim=1)
                gaussians.opacities = torch.cat((gaussians.opacities, current_gaussians.opacities), dim=1)
                
                # next level
                local_coordinates = next_coordinates
                last_voxel_size = voxel_size
            
            batch_losses[0].append(total_existence_loss / self.voxel_size_count)
            batch_losses[1].append(total_current_loss / self.voxel_size_count)
            batch_losses[2].append(total_offset_loss / self.voxel_size_count)
            batch_losses[3].append(total_color_loss / self.voxel_size_count)
            
            batch_gaussians.append(gaussians)
        
        
        combined_gaussians = combine_batch_gaussians(batch_gaussians)
        combined_gaussians.others["existence_loss"] = torch.stack(batch_losses[0])
        combined_gaussians.others["current_loss"] = torch.stack(batch_losses[1])
        combined_gaussians.others["offset_loss"] = torch.stack(batch_losses[2])
        combined_gaussians.others["color_loss"] = torch.stack(batch_losses[3])
        
        combined_gaussians.others["bbox"] = bbox
        
        return combined_gaussians