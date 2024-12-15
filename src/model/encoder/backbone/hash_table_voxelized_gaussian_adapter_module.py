import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
import math
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
from ..mvsnet.cas_mvsnet_module import CasMVSNetModuleResult, PointCloudResult
from ...types import EncoderOutput
from ....utils import build_covariance_from_scaling_rotation

GAUSSIAN_FEATURE_CHANNELS = 15

SLICE_DELTA_MEANS = slice(0, 3)
SLICE_QUATERNION = slice(3, 7)
SLICE_SCALE = slice(7, 10)
SLICE_SHS = slice(10, 13)
SLICE_OPACITY = slice(13, 14)
SLICE_CURRENT = slice(14, 15)

normalization = lambda x: (x - torch.mean(x)) / torch.std(x)

delta_means_activation = lambda x, far, size: normalization(x) * 2 * far / size / 6
scaling_activation = lambda x, far, size: torch.sigmoid(x) * 2 * far / size
quaternion_activation = lambda x: x
opacity_activation = lambda x: torch.sigmoid(x - 4) # sigmoid(-3 | -4) = 0.0474 | 0.018
color_activation = lambda x: torch.sigmoid(x)
current_activation = lambda x: torch.sigmoid(x)


def activate_hash_table(hash_table: torch.Tensor, far: torch.Tensor, voxel_size: int):
    activated_hash_table = torch.zeros_like(hash_table, device="cuda")
    activated_hash_table[SLICE_DELTA_MEANS] = delta_means_activation(hash_table[SLICE_DELTA_MEANS], far, voxel_size)
    activated_hash_table[SLICE_QUATERNION] = quaternion_activation(hash_table[SLICE_QUATERNION])
    activated_hash_table[SLICE_SCALE] = scaling_activation(hash_table[SLICE_SCALE], far, voxel_size)
    activated_hash_table[SLICE_SHS] = color_activation(hash_table[SLICE_SHS])
    activated_hash_table[SLICE_OPACITY] = opacity_activation(hash_table[SLICE_OPACITY])
    activated_hash_table[SLICE_CURRENT] = current_activation(hash_table[SLICE_CURRENT])
    return activated_hash_table


def compute_voxel_center(coordinates: torch.Tensor, camera_center: torch.Tensor, voxel_size: int, far: torch.Tensor) -> torch.Tensor:
    """
    mapping (0, 0, 0) -> mu - f + o
            (S-1, S-1, S-1) -> mu + f - o
            (S, S, S) -> mu + f + o
            
    
    input:
        `coordinates`: [N, 3] int
        `camera_center`: [3]
        
    output:
        [N, 3] float
    """
    
    return (coordinates / voxel_size * 2 * far) - far + camera_center.view(1, 3) + (far / voxel_size)


def compute_voxel_indices(x: torch.Tensor, camera_center: torch.Tensor, voxel_size: int, far: torch.Tensor):
    """
    ### The inverse function of `compute_voxel_center`
    mapping (0, 0, 0) <- mu - f + o (+-o)
            (S-1, S-1, S-1) <- mu + f - o
            (S, S, S) <- mu + f + o
    
    input:
        `x`: [N, 3]
        `camera_center`: [3]
        
    output:
        [N, 3] int
    """
    return ((x - (far / voxel_size) - camera_center.view(1, 3) + far) * voxel_size / 2 / far).round()



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
    

def create_coordinates(voxel_size: int, last_voxel_size: int = 0, last_coordinates: torch.Tensor = None):
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
    voxel_center = compute_voxel_center(coordinates, camera_center, voxel_size, far).permute(1, 0) # (3, N)
    b, n, dim, d_sh = 1, coordinates.shape[0], 3, 1
    
    means: torch.Tensor = gaussian_features[SLICE_DELTA_MEANS] + voxel_center # (3, N)
    covariances: torch.Tensor = build_covariance_from_scaling_rotation(
        gaussian_features[SLICE_SCALE], 
        1, 
        gaussian_features[SLICE_QUATERNION]) # (9, N)
    harmonics: torch.Tensor = gaussian_features[SLICE_SHS] # (3, N)
    opacities: torch.Tensor = gaussian_features[SLICE_OPACITY] # (1, N)
    
    gaussians = EncoderOutput(
        means=means.permute(1, 0).view(b, n, dim),
        covariances=covariances.permute(1, 0).reshape(b, n, dim, dim), 
        harmonics=harmonics.permute(1, 0).view(b, n, 3, d_sh), 
        opacities=opacities.permute(1, 0).view(b, n)
    )
    
    return gaussians


def combine_batch_gaussians(batch_gaussians: list[EncoderOutput]) -> EncoderOutput:
    b, dim, d_sh = 1, 3, 1
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
        voxel_indices = compute_voxel_indices(xyz, camera_center, voxel_size, far)
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
    
    max_downsampled_pcd_coordinates = max_downsampled_pcd[:, :3].floor() # (N, 3)
    
    return (max_downsampled_pcd_coordinates + d_coordinates).reshape(-1, 3).unique(dim=0) # (27, N, 3) -> (N', 3)
    


def compute_struct_loss(
    downsampled_pcds: list[torch.Tensor], 
    classified_pcds: list[torch.Tensor], 
    hash_table: torch.Tensor, 
    voxel_size_list: list[int], 
    scale_idx: int, 
    coordinates: torch.Tensor, 
    max_scale_voxel_existence_coordinates: torch.Tensor, 
    camera_center: torch.Tensor, 
    far: torch.Tensor):
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
    get_opacity = lambda coor: hash_query(coor, hash_table)[SLICE_OPACITY]
    get_delta_means = lambda coor: hash_query(coor, hash_table)[SLICE_DELTA_MEANS]
    get_color = lambda coor: hash_query(coor, hash_table)[SLICE_SHS]
    get_current = lambda coor: hash_query(coor, hash_table)[SLICE_CURRENT]

    is_mapped_point = torch.all(coordinates[:, None, :3] == downsampled_pcds[scale_idx][:, :3], dim=-1).any(dim=1)
    
    no_point_coordinates = coordinates[torch.logical_not(is_mapped_point)] # case 1
    has_point_coordinates = coordinates[is_mapped_point] # (N1, 3)

    is_mapped_and_correct_classified_point = torch.all(
        has_point_coordinates[:, None, :3] == classified_pcds[scale_idx][:, :3], dim=-1).any(dim=1)
    
    not_current_scale_point_coordinates = has_point_coordinates[torch.logical_not(is_mapped_and_correct_classified_point)] # case 3
    # TODO: check is_mapped_and_correct_classified_point.sum() == classified_pcd.shape[0]
    classified_pcd = classified_pcds[scale_idx] # case 2 (n2, iiixyzrgb)
    
    # compute existence loss, current loss, offset loss and color loss
    existence_loss, existence_n = 0., 0
    current_loss, current_n = 0., 0
    
    # for case 3
    existence_loss += (1. - get_opacity(not_current_scale_point_coordinates)).sum()
    existence_n += not_current_scale_point_coordinates.shape[0]
    current_loss += get_current(not_current_scale_point_coordinates).sum()
    current_n += not_current_scale_point_coordinates.shape[0]
    
    # for case 2
    existence_loss += (1. - get_opacity(classified_pcd[:, :3].int())).sum()
    existence_n += classified_pcd.shape[0]
    current_loss += (1. - get_current(classified_pcd[:, :3].int())).sum()
    current_n += classified_pcd.shape[0]
    
    predicted_means: torch.Tensor = \
        compute_voxel_center(classified_pcd[:, :3], camera_center, voxel_size_list[-1], far) + \
        get_delta_means(classified_pcd[:, :3].int()).permute(1, 0)
        
    predicted_color = get_color(classified_pcd[:, :3].int()).permute(1, 0)
    
    offset_loss = (classified_pcd[:, 3:6] - predicted_means).norm(dim=1).sum() / math.sqrt(3)
    color_loss = (classified_pcd[:, 6:] - predicted_color).norm(p=1, dim=1).sum() / 3
    offset_n = color_n = classified_pcd.shape[0]
    
    # for case 1
    # Find the coursest scale and decide: does it have a pseudo truth?
    max_scale_no_point_coordinates = (no_point_coordinates * (voxel_size_list[0] / voxel_size_list[scale_idx])).int()

    is_max_downsampled_may_exist_coordinates = torch.all(
        max_scale_no_point_coordinates[:, None, :3] == max_scale_voxel_existence_coordinates[:, :3], dim=-1).any(dim=1)
    
    max_downsampled_not_exist_coordinates = max_scale_no_point_coordinates[torch.logical_not(is_max_downsampled_may_exist_coordinates)]
    
    existence_loss += (get_opacity(max_downsampled_not_exist_coordinates)).sum()
    existence_n += max_downsampled_not_exist_coordinates.shape[0]
    
    # loss normalization
    if existence_n != 0: existence_loss /= existence_n
    if current_n != 0: current_loss /= current_n
    if offset_n != 0: offset_loss /= offset_n
    if color_n != 0: color_loss /= color_n
        
    return existence_loss, current_loss, offset_loss, color_loss
    
    

class HashTableVoxelizedGaussianAdapterModule(nn.Module):

    def __init__(self, voxel_size_list=[32, 128, 512], min_opacity=[0.1, 0.1, 0.1]) -> None:
        super().__init__()
        assert len(voxel_size_list) == 3
        self.voxel_size_count = len(voxel_size_list)
        self.voxel_size_list = voxel_size_list
        self.min_opacity = min_opacity
        pass
    
        
    def forward(self, multi_scale_hash_tables: tuple, cas_module_result: CasMVSNetModuleResult, extrinsics: torch.Tensor, fars: torch.Tensor):
        b = multi_scale_hash_tables[0].shape[0]
        far = fars[0, 0]
        is_trainning = multi_scale_hash_tables[0].grad_fn != None
        batch_gaussians = []
        batch_losses = [[], [], [], []] # total_existence_loss, total_current_loss, total_offset_loss, total_color_loss
        for batch in range(b):
            coordinates = None
            last_voxel_size = 0
            total_existence_loss, total_current_loss, total_offset_loss, total_color_loss = torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda")
            camera_center = extrinsics.inverse()[batch][:, :3, 3].mean(dim=0)
            dim, d_sh = 3, 1
            gaussians = EncoderOutput(
                means=torch.zeros(1, 0, dim, device="cuda"), 
                covariances=torch.zeros(1, 0, dim, dim, device="cuda"), 
                harmonics=torch.zeros(1, 0, 3, d_sh, device="cuda"), 
                opacities=torch.zeros(1, 0, device="cuda")
            )
            
            with torch.no_grad():
                downsampled_pcds, classified_pcds = downsample_and_classify_pcd(
                    pcd=cas_module_result.registed_pcd, 
                    voxel_size_list=self.voxel_size_list, 
                    camera_center=camera_center, 
                    far=far, 
                    batch_idx=batch
                )
                max_scale_voxel_existence_coordinates = compute_max_scale_voxel_existence_coordinates_by_pcd(downsampled_pcds[0])
            
            for scale_idx in range(self.voxel_size_count):
                voxel_size = self.voxel_size_list[scale_idx]
                hash_table = activate_hash_table(multi_scale_hash_tables[scale_idx][batch], far, voxel_size) # (C, T)
                
                pcd_voxel_coordinates = downsampled_pcds[scale_idx][:, :3].int()
                
                coordinates = create_coordinates(
                    voxel_size=voxel_size, 
                    last_voxel_size=last_voxel_size, 
                    last_coordinates=coordinates
                )
                
                coordinates = torch.cat((coordinates, pcd_voxel_coordinates), dim=0).unique(dim=0) # combine coordinates from last level and point cloud
                
                if is_trainning:
                    # compute losses
                    existence_loss, current_loss, offset_loss, color_loss = compute_struct_loss(
                        downsampled_pcds=downsampled_pcds, 
                        classified_pcds=classified_pcds, 
                        hash_table=hash_table, 
                        voxel_size_list=self.voxel_size_list, 
                        scale_idx=scale_idx, 
                        coordinates=coordinates, 
                        max_scale_voxel_existence_coordinates=max_scale_voxel_existence_coordinates, 
                        camera_center=camera_center, 
                        far=far
                    )
                    total_existence_loss += existence_loss
                    total_current_loss += current_loss
                    total_offset_loss += offset_loss
                    total_color_loss += color_loss
                
                # query and switch to next level
                gaussian_features = hash_query(coordinates=coordinates, hash_table=hash_table)
                existence_mask = gaussian_features[SLICE_OPACITY][0] >= self.min_opacity[scale_idx]
                current_mask = gaussian_features[SLICE_CURRENT][0] >= 0.5
                
                # select gaussians at current scale
                current_existence_mask = torch.logical_and(existence_mask, current_mask)
                if current_existence_mask.sum() > 0:
                    current_gaussian_features = gaussian_features[:, current_existence_mask]
                    current_coordinates = coordinates[current_existence_mask]
                    current_gaussians = create_gaussians_from_features(
                        gaussian_features=current_gaussian_features, 
                        coordinates=current_coordinates, 
                        camera_center=camera_center, 
                        voxel_size=self.voxel_size_list[scale_idx], 
                        far=far
                    )
                    
                    gaussians.means = torch.cat((gaussians.means, current_gaussians.means), dim=1)
                    gaussians.covariances = torch.cat((gaussians.covariances, current_gaussians.covariances), dim=1)
                    gaussians.harmonics = torch.cat((gaussians.harmonics, current_gaussians.harmonics), dim=1)
                    gaussians.opacities = torch.cat((gaussians.opacities, current_gaussians.opacities), dim=1)
                
                # next level
                coordinates = coordinates[torch.logical_and(existence_mask, torch.logical_not(current_mask))]
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