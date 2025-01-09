import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
import math
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
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


def activate_hash_table(hash_table: torch.Tensor, far: torch.Tensor, voxel_size: int):
    activated_hash_table = torch.zeros_like(hash_table, device="cuda")
    activated_hash_table[SLICE_DELTA_MEANS] = delta_means_activation(hash_table[SLICE_DELTA_MEANS], far, voxel_size)
    activated_hash_table[SLICE_QUATERNION] = quaternion_activation(hash_table[SLICE_QUATERNION])
    activated_hash_table[SLICE_SCALE] = scaling_activation(hash_table[SLICE_SCALE], far, voxel_size)
    activated_hash_table[SLICE_OPACITY] = opacity_activation(hash_table[SLICE_OPACITY])
    activated_hash_table[SLICE_SHS] = hash_table[SLICE_SHS]
    activated_hash_table[SLICE_SHS_D1] = color_activation(hash_table[SLICE_SHS_D1])
    return activated_hash_table


def compute_voxel_center(coordinates: torch.Tensor, voxel_size: int, far: torch.Tensor) -> torch.Tensor:
    """
    ### Mapping cooordinates to voxel center with origin `0`.
    mapping (0, 0, 0) -> o
            (S-1, S-1, S-1) -> 2f - o
            (S, S, S) -> 2f + o
            
    
    input:
        `coordinates`: coordinates with shape [N, 3] int
        `camera_center`: [3]
        
    output:
        voxel center with shape [N, 3] float
    """
    return (coordinates / voxel_size * 2 * far) + (far / voxel_size)


def compute_voxel_indices(voxel_center: torch.Tensor, voxel_size: int, far: torch.Tensor):
    """
    ### The inverse function of `compute_voxel_center` with origin `0`
    mapping (0, 0, 0) <- o (+-o)
            (S-1, S-1, S-1) <- 2f - o (+-o)
            (S, S, S) <- 2f + o (+-o)
    
    input:
        `voxel_center`: voxel center with shape [N, 3] float
        `camera_center`: [3]
        
    output:
        coordinates with shape [N, 3] int
    """
    return ((voxel_center - (far / voxel_size)) * voxel_size / 2 / far).round().int()

def compute_local_coordinates_offset(camera_center: torch.Tensor, voxel_size: int, far: torch.Tensor):
    """
    ### compute the offset from global coordinates (origin `0`) to local coordinates (origin `mu - f`)
    if add the offset, it means switch from local coordinates to global coordinates (l2g), 
    if minus the offset, it means switch from global coordinates to local coordinates (g2l).
    """
    return ((camera_center.view(1, 3) - far) * voxel_size / 2 / far).int()

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
    
    
def isin_3d_coordinates(coor_1: torch.Tensor, coor_2: torch.Tensor):
    """
    ### Compute whether the element of `coor_1` is in `coor_2`. Ensure: 1. dtype=int; 2. max element < 1000;
    
    input:
        `coor_1`: [N, 3] int
        `coor_2`: [N, 3] int
    output:
        [N] bool
    """
    coor_1_flat = coor_1[:, 0] + coor_1[:, 1] * 1000 + coor_1[:, 2] * 1000000
    coor_2_flat = coor_2[:, 0] + coor_2[:, 1] * 1000 + coor_2[:, 2] * 1000000
    
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
    return result.view(-1, 3)


def create_gaussians_from_features(gaussian_features: torch.Tensor, coordinates: torch.Tensor, camera_center: torch.Tensor, voxel_size: int, far: torch.Tensor) -> EncoderOutput:
    """
    input:
        gaussian_features: [C, N]
        coordinates: [N, 3]
        extrinsic: [V, 4, 4]
        far: 1
    """
    local_coordinates_offset = compute_local_coordinates_offset(camera_center, voxel_size, far)
    voxel_center = compute_voxel_center(coordinates + local_coordinates_offset, voxel_size, far).permute(1, 0) # (3, N)
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
        harmonics=harmonics.permute(1, 0).view(b, n, 3, d_sh), 
        opacities=opacities.permute(1, 0).view(b, n)
    )
    
    return gaussians


def combine_batch_gaussians(batch_gaussians: list[EncoderOutput]) -> EncoderOutput:
    b, dim, d_sh = 1, 3, SH_DEGREE ** 2
    gaussian_size = 0
    means_list, covariance_list, harmonics_list, opacities_list = [], [], [], []
    for gaussian in batch_gaussians:
        if gaussian.opacities.shape[1] > gaussian_size:
            gaussian_size = gaussian.opacities.shape[1]
            
    for gaussian in batch_gaussians:
        append_size = gaussian_size - gaussian.opacities.shape[1]
        if append_size > 0: 
            gaussian.means = torch.cat((gaussian.means, torch.zeros(b, append_size, dim)), dim=1)
            gaussian.covariances = torch.cat((gaussian.covariances, torch.zeros(b, append_size, dim, dim)), dim=1)
            gaussian.harmonics = torch.cat((gaussian.harmonics, torch.zeros(b, append_size, 3, d_sh)), dim=1)
            gaussian.opacities = torch.cat((gaussian.opacities, torch.zeros(b, append_size)), dim=1)
        means_list.append(gaussian.means)
        covariance_list.append(gaussian.covariances)
        harmonics_list.append(gaussian.harmonics)
        opacities_list.append(gaussian.opacities)
        
    return EncoderOutput(
        means = torch.cat(means_list, dim=0), 
        covariances = torch.cat(covariance_list, dim=0), 
        harmonics = torch.cat(harmonics_list, dim=0), 
        opacities = torch.cat(opacities_list, dim=0)
    )

def voxel_down_sample(pcd: torch.Tensor, voxel_indices: torch.Tensor):
    """
    input:
        pcd: [N, 6(xyzrgb)]
        voxel_indices: [N, 3]
        
    output:
        downsampled_points: [N', 6(xyzrgb)]
        unique_voxels: [N' 3]
    """
    unique_voxels, inverse_indices = voxel_indices.unique(dim=0, return_inverse=True)
    
    downsampled_points = torch.zeros((unique_voxels.shape[0], 6), device=pcd.device)
    
    for i in range(unique_voxels.shape[0]):
        downsampled_points[i] = pcd[inverse_indices == i].mean(dim=0)
    
    return downsampled_points, unique_voxels


@torch.no_grad()
def downsample_and_classify_pcd(
    pcd: PointCloudResult, 
    voxel_size_list: list[int], 
    camera_center: torch.Tensor, 
    far: torch.Tensor, 
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
        local_coordinates_offset = compute_local_coordinates_offset(camera_center, voxel_size, far)
        voxel_indices = compute_voxel_indices(xyz, voxel_size, far)
        voxel_indices -= local_coordinates_offset
        downsampled_xyzrgb, downsampled_voxels = voxel_down_sample(xyzrgb, voxel_indices)
        downsampled_pcd_list.append(torch.cat((downsampled_voxels, downsampled_xyzrgb), dim=1))
    
    min_downsampled_pcd: torch.Tensor = downsampled_pcd_list[-1] # (n, iiixyzrgb)
    min_downsampled_xyz = min_downsampled_pcd[:, 3:6]
    
    distances, _, _ = knn_points(
        p1=min_downsampled_xyz.unsqueeze(0), 
        p2=min_downsampled_xyz.unsqueeze(0),
        K=2
    )
    min_dist = distances[0, :, 1] # (N)
    
    classified_pcd_list = [] # different voxel size points
    last_threshold, threshold = 0., 0.
    # calculate diagonal lengths of the different scale voxels to classify points.
    for i in range(len(voxel_size_list) - 1, 0, -1):
        threshold = (((2 * far / voxel_size_list[i]) + (2 * far / voxel_size_list[i-1])) / 2) * math.sqrt(3)
        mask = torch.logical_and(min_dist >= last_threshold, min_dist < threshold)
        classified_pcd_list.insert(0, min_downsampled_pcd[mask])
        last_threshold = threshold
    
    mask = min_dist >= threshold
    classified_pcd_list.insert(0, min_downsampled_pcd[mask]) # [(n, iiixyzrgb) * 3],
    
    return downsampled_pcd_list, classified_pcd_list # [(n, iiixyzrgb) * 3], [(n, iiixyzrgb) * 3] on GPU


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
    get_color = lambda mask: SH2RGB(gaussians.harmonics[mask.unsqueeze(0)].view(-1, 3 * SH_DEGREE ** 2)[..., :3])

    is_mapped_point = isin_3d_coordinates(local_coordinates, downsampled_pcds[scale_idx][:, :3].int())
    
    # compute existence loss, offset loss and color loss
    existence_loss, existence_n = 0., 0
    
    # for case 2
    existence_loss += (1. - get_opacity(is_mapped_point)).sum()
    existence_n += is_mapped_point.sum()
    
    predicted_means: torch.Tensor = get_means(is_mapped_point)
        
    predicted_color = get_color(is_mapped_point)
    
    offset_loss = (downsampled_pcds[scale_idx][:, 3:6] - predicted_means).norm(dim=1).sum() / math.sqrt(3)
    color_loss = (downsampled_pcds[scale_idx][:, 6:] - predicted_color).norm(p=1, dim=1).sum() / 3
    offset_n = color_n = downsampled_pcds[scale_idx].shape[0]
    
    # loss normalization
    if existence_n != 0: existence_loss /= existence_n
    if offset_n != 0: offset_loss /= offset_n
    if color_n != 0: color_loss /= color_n
        
    return existence_loss, offset_loss, color_loss
    
    

class VoxelizedGaussianAdapterModule(nn.Module):

    def __init__(self, transformer: VoxelToPointTransformer, feature_channels=192, voxel_size_list=[32, 128, 512], min_opacity=[0.1, 0.1, 0.1]) -> None:
        super().__init__()
        assert len(voxel_size_list) == 3
        self.transformer = transformer
        self.voxel_size_count = len(voxel_size_list)
        self.voxel_size_list = voxel_size_list
        self.min_opacity = min_opacity
        

        self.gaussian_parameter_predictor = nn.Sequential(
            nn.Linear(in_features=feature_channels, out_features=64), 
            nn.Linear(in_features=64, out_features=GAUSSIAN_FEATURE_CHANNELS)
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        pass
    
        
    def forward(self, cnn_features: torch.Tensor, cas_module_result: CasMVSNetModuleResult, extrinsics: torch.Tensor, intrinsics: torch.Tensor, fars: torch.Tensor):
        b, v, c, h, w = cnn_features.shape
        far = fars[0, 0]
        is_trainning = cnn_features.grad_fn != None
        batch_gaussians = []
        batch_losses = [[], [], [], []] # total_existence_loss, total_current_loss, total_offset_loss, total_color_loss
        prob_pcd = cas_module_result.registed_prob_pcd
        
        
        for batch in range(b):
            # for every batch the number of gaussian may be different (LoD)
            local_coordinates = None
            last_voxel_size = 0
            total_existence_loss, total_current_loss, total_offset_loss, total_color_loss = torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda")
            camera_center = extrinsics.inverse()[batch][:, :3, 3].mean(dim=0)
            gaussians = empty_encoder_output()
            
            with torch.no_grad():
                downsampled_pcds, classified_pcds = downsample_and_classify_pcd(
                    pcd=cas_module_result.registed_pcd, 
                    voxel_size_list=self.voxel_size_list, 
                    camera_center=camera_center, 
                    far=far, 
                    batch_idx=batch
                )
            
            # TODO: Create multi-scale voxel according to points.
            voxel_size = self.voxel_size_list[0]
            local_coordinates = create_local_coordinates(voxel_size, last_coordinates=None)
            offset = compute_local_coordinates_offset(camera_center, voxel_size, far)
            coordinates = local_coordinates + offset
            centers: torch.Tensor = compute_voxel_center(coordinates, voxel_size, far) # (N, 3)
            
            prob_pcd_xyz = prob_pcd.vertices[batch][:, :3] # (V, 3, H, W)
            prob_pcd_ijk = compute_voxel_indices(prob_pcd_xyz, voxel_size, far) # (V, 3, H, W)
            
            # for reduce memory consumption we uapply transformer per view
            voxel_feature_list = []
            for view_idx in range(v):
                view_slicer = slice(view_idx, view_idx + 1)
                voxel_feature = self.transformer(
                    cnn_features=cnn_features[batch, view_slicer], # (V, C, H, W)
                    extrinsics=extrinsics[batch, view_slicer], 
                    intrinsics=intrinsics[batch, view_slicer], 
                    point_xyz=prob_pcd_xyz[view_slicer], # (V, 3, H, W)
                    voxel_xyz=centers.transpose(0, 1).unsqueeze(0), # (V, 3, N)
                    point_ijk=prob_pcd_ijk[view_slicer], 
                    voxel_ijk=coordinates.transpose(0, 1).unsqueeze(0), # (V, 3, N)
                    confidences=prob_pcd.vertices_confidence[batch, view_slicer] # (V, H, W)
                ) # (V, C, N)
                voxel_feature_list.append(voxel_feature)
                
            voxel_features = torch.cat(voxel_feature_list, dim=0) # (V, C, N)
            
            merged_feat = torch.mean(voxel_features, dim=0).transpose(0, 1) # (N, C)
            
            gaussian_features = self.gaussian_parameter_predictor(merged_feat) # (N, 15)
            
            gaussian_features = gaussian_features.transpose(0, 1) # (15, N)
            activated_gaussian_features = activate_hash_table(gaussian_features, far, voxel_size)
            
            gaussians = create_gaussians_from_features(
                gaussian_features=activated_gaussian_features, 
                coordinates=local_coordinates, 
                camera_center=camera_center, 
                voxel_size=voxel_size, 
                far=far
            )
            
            if is_trainning:
                # compute losses
                existence_loss, offset_loss, color_loss = compute_struct_loss(
                    downsampled_pcds=downsampled_pcds, 
                    scale_idx=0, 
                    local_coordinates=local_coordinates, 
                    gaussians=gaussians
                )
                total_existence_loss += existence_loss
                total_offset_loss += offset_loss
                total_color_loss += color_loss
            
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