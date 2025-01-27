import torch
import gc
from torch import nn
import torch.nn.functional as F
import math
from .voxel_to_point_cross_attn_transformer import VoxelToPointTransformer
from ..mvsnet.cas_mvsnet_module import CasMVSNetModuleResult, PointCloudResult
from ...types import EncoderOutput, empty_encoder_output
from ....utils import build_covariance_from_scaling_rotation

SH_DEGREE = 4
GAUSSIAN_FEATURE_CHANNELS = 11 + 3 * SH_DEGREE ** 2

SLICE_DELTA_MEANS = slice(0, 3)
SLICE_QUATERNION = slice(3, 7)
SLICE_SCALE = slice(7, 10)
SLICE_OPACITY = slice(10, 11)
SLICE_SHS_D1 = slice(11, 14) # sh degree 1
SLICE_SHS_D2 = slice(14, 11 + 3 * 2 ** 2)
SLICE_SHS_D3 = slice(11 + 3 * 2 ** 2, 11 + 3 * 3 ** 2)
SLICE_SHS_D4 = slice(11 + 3 * 3 ** 2, 11 + 3 * 4 ** 2)
SLICE_SHS = slice(11, GAUSSIAN_FEATURE_CHANNELS)

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5


normalization = lambda x: (x - torch.mean(x)) / torch.std(x)

delta_means_activation = lambda x, far, size: normalization(x) * 2 * far / size / 6
scaling_activation = lambda x, far, size: torch.sigmoid(x) * 2 * far / size
quaternion_activation = lambda x: x
opacity_activation = lambda x: torch.sigmoid(x - 4) # sigmoid(-3 | -4) = 0.0474 | 0.018
color_activation = lambda x: RGB2SH(torch.sigmoid(x))
current_activation = lambda x: torch.sigmoid(x)


def activate_gaussians(gaussian_features: torch.Tensor, far: torch.Tensor, voxel_size: int):
    activated_gaussian_features = torch.zeros_like(gaussian_features, device="cuda")
    activated_gaussian_features[SLICE_DELTA_MEANS] = delta_means_activation(gaussian_features[SLICE_DELTA_MEANS], far, voxel_size)
    activated_gaussian_features[SLICE_QUATERNION] = quaternion_activation(gaussian_features[SLICE_QUATERNION])
    activated_gaussian_features[SLICE_SCALE] = scaling_activation(gaussian_features[SLICE_SCALE], far, voxel_size)
    activated_gaussian_features[SLICE_OPACITY] = opacity_activation(gaussian_features[SLICE_OPACITY])
    activated_gaussian_features[SLICE_SHS] = gaussian_features[SLICE_SHS]
    activated_gaussian_features[SLICE_SHS_D1] = color_activation(gaussian_features[SLICE_SHS_D1])
    activated_gaussian_features[SLICE_SHS_D2] = gaussian_features[SLICE_SHS_D2] / 20
    activated_gaussian_features[SLICE_SHS_D3] = gaussian_features[SLICE_SHS_D3] / 40
    activated_gaussian_features[SLICE_SHS_D4] = gaussian_features[SLICE_SHS_D4] / 80
    return activated_gaussian_features


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
        
        border_xyz = torch.matmul(torch.linalg.inv(extrinsics), torch.cat((far_border, near_border), dim=-1)) # (B*V, 4, 8)
        
        border_points = border_xyz.view(b, v, 4, -1).permute(0, 2, 1, 3).reshape(b, 4, -1) # (B, 4, V * 8)
        min_point, max_point = border_points.min(dim=-1).values, border_points.max(dim=-1).values
        
        self.origin = min_point[:, :3] # (B, 3)
        self.size = (max_point - min_point).max(dim=-1).values # (B)
        pass
    
    def compute_voxel_center(self, coordinates: torch.Tensor, voxel_size: int, batch: int):
        """
        ### Mapping cooordinates to voxel center with origin `0`.
        mapping (0, 0, 0) -> o
                (S-1, S-1, S-1) -> s - o
                (S, S, S) -> s + o
                
        
        input:
            `coordinates`: coordinates with shape [N, 3] int
            
        output:
            voxel center with shape [N, 3] float
        """
        return (coordinates / voxel_size * self.size[batch]) + (self.size[batch] / 2 / voxel_size)
    
    def compute_voxel_indices(self, voxel_center: torch.Tensor, voxel_size: int, batch: int):
        """
        ### The inverse function of `compute_voxel_center` with origin `0`
        mapping (0, 0, 0) <- o (+-o)
                (S-1, S-1, S-1) <- s - o (+-o)
                (S, S, S) <- s + o (+-o)
        
        input:
            `voxel_center`: voxel center with shape [N, 3] float
            
        output:
            coordinates with shape [N, 3] int
        """
        return ((voxel_center - (self.size[batch] / 2 / voxel_size)) * voxel_size / self.size[batch]).round().int()
    
    def compute_local_coordinates_offset(self, voxel_size: int, batch: int):
        """
        ### compute the offset from global coordinates (origin `0`) to local coordinates (origin `self.origin`)
        if add the offset, it means switch from local coordinates to global coordinates (l2g), 
        if minus the offset, it means switch from global coordinates to local coordinates (g2l).
        """
        return ((self.origin[batch]) * voxel_size / self.size[batch]).int()

    pass

def bitwise_xor(x: torch.Tensor, dim: int):
    assert dim < len(x.shape)
    if x.shape[dim] == 1: return x
    slices = torch.unbind(x, dim=dim)
    result = slices[0]
    for slice in slices[1:]:
        result = torch.bitwise_xor(result, slice)
    return result

def hash_query(coordinates: torch.Tensor, hash_table: torch.Tensor):
    """
    input:
        `x`: [N, 3] int
        `hash_table`: [C, T]
        
    output:
        [C, N]
    """
    PI_1, PI_2, PI_3 = 1, 2654435761, 805459861 # from Instant-NGP
    
    primes = torch.tensor([PI_1, PI_2, PI_3], device=coordinates.device)
    
    hash_index = bitwise_xor(coordinates * primes, dim=-1) % hash_table.shape[1] # (N)
    
    return hash_table[:, hash_index]
    
    
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
    coor_1 = coor_1.long()
    coor_2 = coor_2.long()
    
    coor_1_flat = coor_1[:, 0] + coor_1[:, 1] * 10000 + coor_1[:, 2] * 100000000
    coor_2_flat = coor_2[:, 0] + coor_2[:, 1] * 10000 + coor_2[:, 2] * 100000000
    
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
    d_grid = torch.stack((d_x, d_y, d_z), dim=-1).view(-1, 3) # (seg_times^3, 1, 3)
    
    result = (last_coordinates * seg_times).view(-1, 1, 3) + d_grid # (seg_times^3, N, 3)
    return result.view(-1, 3).unique(sorted=True, dim=0) # use unique to sort index


def create_gaussians_from_features(gaussian_features: torch.Tensor, coordinates: torch.Tensor, voxel_size: int, bbox: BoundingBox, batch: int) -> EncoderOutput:
    """
    input:
        gaussian_features: [C, N]
        coordinates: [N, 3]
        extrinsic: [V, 4, 4]
        far: 1
    """
    local_coordinates_offset = bbox.compute_local_coordinates_offset(voxel_size, batch)
    voxel_center = bbox.compute_voxel_center(coordinates + local_coordinates_offset, voxel_size, batch).permute(1, 0) # (3, N)
    b, n, dim, d_sh = 1, coordinates.shape[0], 3, SH_DEGREE ** 2
    
    means: torch.Tensor = gaussian_features[SLICE_DELTA_MEANS] + voxel_center # (3, N)
    covariances: torch.Tensor = build_covariance_from_scaling_rotation(
        gaussian_features[SLICE_SCALE], 
        1, 
        gaussian_features[SLICE_QUATERNION]) # (9, N)
    harmonics: torch.Tensor = gaussian_features[SLICE_SHS] # (d^2, N)
    opacities: torch.Tensor = gaussian_features[SLICE_OPACITY] # (1, N)
    
    gaussians = EncoderOutput(
        means=means.permute(1, 0).view(b, n, dim),
        covariances=covariances.permute(1, 0).reshape(b, n, dim, dim), 
        harmonics=harmonics.permute(1, 0).view(b, n, d_sh, 3).transpose(2, 3), # note that (d_sh, 3) in features
        opacities=opacities.permute(1, 0).view(b, n)
    )
    
    gaussians.others["scales"] = gaussian_features[SLICE_SCALE].permute(1, 0).view(b, n, dim) # (B, N, 3)
    return gaussians


def combine_batch_gaussians(batch_gaussians: list[EncoderOutput]) -> EncoderOutput:
    b, dim, d_sh = 1, 3, SH_DEGREE ** 2
    gaussian_size = 0
    means_list, covariance_list, harmonics_list, opacities_list = [], [], [], []
    scales_list, append_size_list = [], []
    for gaussian in batch_gaussians:
        if gaussian.opacities.shape[1] > gaussian_size:
            gaussian_size = gaussian.opacities.shape[1]
            
    for gaussian in batch_gaussians:
        append_size = gaussian_size - gaussian.opacities.shape[1]
        if append_size > 0: 
            gaussian.means = torch.cat((gaussian.means, torch.zeros(b, append_size, dim, device=gaussian.means.device)), dim=1)
            gaussian.covariances = torch.cat((gaussian.covariances, torch.zeros(b, append_size, dim, dim, device=gaussian.means.device)), dim=1)
            gaussian.harmonics = torch.cat((gaussian.harmonics, torch.zeros(b, append_size, 3, d_sh, device=gaussian.means.device)), dim=1)
            gaussian.opacities = torch.cat((gaussian.opacities, torch.zeros(b, append_size, device=gaussian.means.device)), dim=1)
        means_list.append(gaussian.means)
        covariance_list.append(gaussian.covariances)
        harmonics_list.append(gaussian.harmonics)
        opacities_list.append(gaussian.opacities)
        scales_list.append(gaussian.others["scales"])
        append_size_list.append(append_size)
        
    combined_gaussian = EncoderOutput(
        means = torch.cat(means_list, dim=0), 
        covariances = torch.cat(covariance_list, dim=0), 
        harmonics = torch.cat(harmonics_list, dim=0), 
        opacities = torch.cat(opacities_list, dim=0)
    )
    
    combined_gaussian.others["scales_list"] = scales_list # (B, N, 3)
    combined_gaussian.others["append_size_list"] = append_size_list
    return combined_gaussian

def voxel_down_sample(pcd: torch.Tensor, voxel_indices: torch.Tensor):
    """
    input:
        pcd: [N, C]
        voxel_indices: [N, 3(ijk)]
        
    output:
        downsampled_pcd: [N', C]
        unique_voxel_indices: [N', 3]
    """
    
    # Radix-sort-like method to sort pcd by 3d (ijk) indices.
    indices = None
    for i in reversed(range(voxel_indices.shape[1])):
        indices = voxel_indices[:, i].sort().indices
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
    pcd: PointCloudResult, 
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

    xyz = pcd.xyz_batches[batch_idx][:, :3]
    rgb = pcd.rgb_batches[batch_idx]
    xyzrgb = torch.cat((xyz, rgb), dim=1)

    downsampled_pcd_list = [] 
    
    for voxel_size in voxel_size_list:
        # move points to default voxel coordinates and execute downsampling
        local_coordinates_offset = bbox.compute_local_coordinates_offset(voxel_size, batch_idx)
        voxel_indices = bbox.compute_voxel_indices(xyz, voxel_size, batch_idx)
        voxel_indices -= local_coordinates_offset
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
    downsampled_pcds: list[torch.Tensor], 
    scale_idx: int, 
    local_coordinates: torch.Tensor, 
    gaussians: EncoderOutput):
    """
    ### Compute L_struct for a given scale.
    input:
        downsampled_pcds: [n1, 6(iiixyzrgb)] * 3
        classified_pcds: [n2, 6(iiixyzrgb)] * 3
        coordinates: [N, 3]
        
    output:
        existence_loss, offset_loss, color_loss
    
    #### Note that assume coordinates contains all points.
    """
    get_opacity = lambda mask: gaussians.opacities[mask.unsqueeze(0)]
    get_means = lambda mask: gaussians.means[mask.unsqueeze(0)] # (N, 3)
    get_color = lambda mask: SH2RGB(gaussians.harmonics[mask.unsqueeze(0)].reshape(-1, 3, SH_DEGREE ** 2)[..., 0])

    is_mapped_gaussians, is_mapped_points = isin_3d_coordinates(local_coordinates, downsampled_pcds[scale_idx][:, :3].int(), return_inverse=True)
    
    # compute existence loss, offset loss and color loss
    existence_loss, existence_n = 0., 0
    
    # for case 2
    existence_loss += (1. - get_opacity(is_mapped_gaussians)).sum()
    existence_n += is_mapped_gaussians.sum()
    
    predicted_means: torch.Tensor = get_means(is_mapped_gaussians)
        
    predicted_color = get_color(is_mapped_gaussians)
    
    offset_loss = (downsampled_pcds[scale_idx][is_mapped_points, 3:6] - predicted_means).norm(dim=1).sum() / math.sqrt(3)
    color_loss = (downsampled_pcds[scale_idx][is_mapped_points, 6:] - predicted_color).norm(p=1, dim=1).sum() / 3
    offset_n = color_n = is_mapped_points.sum()
    
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
        
    output: mask Tensor(N2) with type `bool`
    """
    # use int64 to avoid data overflow
    point_coordinates = point_coordinates.long()
    local_coordinates = local_coordinates.long()
    
    point_coordinates_flat = point_coordinates[:, 0] + point_coordinates[:, 1] * 10000 + point_coordinates[:, 2] * 100000000
    voxel_coordinates_flat = local_coordinates[:, 0] + local_coordinates[:, 1] * 10000 + local_coordinates[:, 2] * 100000000
    
    has_point_coordinates_flat, counts = point_coordinates_flat.unique(return_counts=True) # (N3)
    
    have_points_coordinates_flat = has_point_coordinates_flat[counts > 1]
    
    isin_mask = torch.isin(voxel_coordinates_flat, have_points_coordinates_flat) # (N2) True if > 1 points inside
    
    return torch.logical_not(isin_mask) # (N2) True if <= 1 points inside

class VoxelizedGaussianAdapterModule(nn.Module):

    def __init__(self, transformer: VoxelToPointTransformer, feature_channels=192, voxel_size_list=[32, 128, 512], patch_size_list=[3, 2, 1]) -> None:
        super().__init__()
        self.transformer = transformer
        self.voxel_size_count = len(voxel_size_list)
        self.voxel_size_list = voxel_size_list
        self.patch_size_list = patch_size_list
        assert len(patch_size_list) == len(voxel_size_list)

        self.gaussian_parameter_predictor = nn.Sequential(
            nn.Linear(in_features=feature_channels, out_features=64), 
            nn.Linear(in_features=64, out_features=GAUSSIAN_FEATURE_CHANNELS)
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        pass
    
        
    def forward(self, cnn_features: torch.Tensor, cas_module_result: CasMVSNetModuleResult, extrinsics: torch.Tensor, intrinsics: torch.Tensor, nears: torch.Tensor, fars: torch.Tensor):
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
        
        for batch in range(b):
            # for every batch the number of gaussian may be different (LoD)
            local_coordinates = None
            last_voxel_size = 0
            total_existence_loss, total_current_loss, total_offset_loss, total_color_loss = torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda")
            camera_center = extrinsics.inverse()[batch][:, :3, 3].mean(dim=0)
            gaussians = empty_encoder_output(d_sh=SH_DEGREE ** 2)
            gaussians.others["scales"] = torch.zeros(b, 0, 3, device=cnn_features.device)
            
            if is_trainning:
                downsampled_pcds = downsample_pcd(
                    pcd=cas_module_result.registed_pcd, 
                    voxel_size_list=self.voxel_size_list, 
                    bbox=bbox, 
                    batch_idx=batch
                )
                
            prob_pcd_xyz = prob_pcd.vertices[batch].permute(0, 2, 3, 1).reshape(-1, 4)[:, :3] # (N, 3)
            prob_pcd_xyz_local = prob_pcd_xyz - bbox.origin[batch]
            max_resolution_voxel_size = self.voxel_size_list[-1]
            max_resolution_prob_pcd, max_resolution_prob_pcd_indices = voxel_down_sample(
                prob_pcd_xyz_local, 
                bbox.compute_voxel_indices(
                    voxel_center=prob_pcd_xyz_local, 
                    voxel_size=max_resolution_voxel_size, # max resolution 
                    batch=batch
                )
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
                
                is_current_scale = identify_is_current_scale(
                    point_coordinates=point_coordinates, 
                    local_coordinates=local_coordinates
                )
                # update current local coordinates
                next_coordinates = local_coordinates[torch.logical_not(is_current_scale)]
                local_coordinates = local_coordinates[is_current_scale]
                n, _ = local_coordinates.shape
                # compute global coordinates
                offset = bbox.compute_local_coordinates_offset(voxel_size, batch)
                coordinates = local_coordinates + offset
                centers: torch.Tensor = bbox.compute_voxel_center(coordinates, voxel_size, batch) # (N, 3)
                
                prob_pcd_xyz = prob_pcd.vertices[batch][:, :3] # (V, 3, H, W)
                prob_pcd_ijk = bbox.compute_voxel_indices(prob_pcd_xyz, voxel_size, batch) # (V, 3, H, W)
                
                # for reduce memory consumption we apply transformer per view
                merged_feat = torch.zeros(c, n, device=cnn_features.device)
                for view_idx in range(v):
                    view_slicer = slice(view_idx, view_idx + 1)
                    voxel_feature: torch.Tensor = self.transformer(
                        cnn_features=cnn_features[batch, view_slicer], # (V, C, H, W)
                        extrinsics=extrinsics[batch, view_slicer], 
                        intrinsics=intrinsics[batch, view_slicer], 
                        point_xyz=prob_pcd_xyz[view_slicer], # (V, 3, H, W)
                        voxel_xyz=centers.transpose(0, 1).unsqueeze(0), # (V, 3, N)
                        point_ijk=prob_pcd_ijk[view_slicer], 
                        voxel_ijk=coordinates.transpose(0, 1).unsqueeze(0), # (V, 3, N)
                        confidences=prob_pcd.vertices_confidence[batch, view_slicer],  # (V, H, W)
                        voxel_length=2 * far / voxel_size, 
                        k=self.patch_size_list[scale_idx]
                    ) # (V, C, N)
                    merged_feat += voxel_feature.squeeze(0) # (c, n)
                    # remove unused variable
                    del voxel_feature
                
                merged_feat = (merged_feat / v).transpose(0, 1) # (N, C)
                
                gaussian_features = self.gaussian_parameter_predictor(merged_feat) # (N, 15)
                
                gaussian_features = gaussian_features.transpose(0, 1) # (15, N)
                activated_gaussian_features = activate_gaussians(gaussian_features, far, voxel_size)
                
                current_gaussians = create_gaussians_from_features(
                    gaussian_features=activated_gaussian_features, 
                    coordinates=local_coordinates, 
                    voxel_size=voxel_size, 
                    bbox=bbox, 
                    batch=batch
                )
                
                if is_trainning:
                    # compute losses
                    existence_loss, offset_loss, color_loss = compute_struct_loss(
                        downsampled_pcds=downsampled_pcds, 
                        scale_idx=scale_idx, 
                        local_coordinates=local_coordinates, 
                        gaussians=current_gaussians
                    )
                    total_existence_loss += existence_loss
                    total_offset_loss += offset_loss / (2 * far / voxel_size * math.sqrt(3)) # devide diagonal length to normalize
                    total_color_loss += color_loss
                
                # Append current gaussians
                gaussians.means = torch.cat((gaussians.means, current_gaussians.means), dim=1)
                gaussians.covariances = torch.cat((gaussians.covariances, current_gaussians.covariances), dim=1)
                gaussians.harmonics = torch.cat((gaussians.harmonics, current_gaussians.harmonics), dim=1)
                gaussians.opacities = torch.cat((gaussians.opacities, current_gaussians.opacities), dim=1)
                gaussians.others["scales"] = torch.cat((gaussians.others["scales"], current_gaussians.others["scales"]), dim=1)
                
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
        
        return combined_gaussians