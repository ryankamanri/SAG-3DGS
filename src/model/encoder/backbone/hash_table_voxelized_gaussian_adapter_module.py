import torch
from torch import nn
import torch.nn.functional as F
from ...types import Gaussians
from ....utils import build_covariance_from_scaling_rotation

SLICE_DELTA_MEANS = slice(0, 3)
SLICE_QUATERNION = slice(3, 7)
SLICE_SCALE = slice(7, 10)
SLICE_SHS = slice(10, 13)
SLICE_OPACITY = slice(13, 14)

normalization = lambda x: (x - torch.mean(x)) / torch.std(x)

delta_means_activation = lambda x, far, size: normalization(x) * 2 * far / size / 3
scaling_activation = torch.exp
covariance_activation = build_covariance_from_scaling_rotation
opacity_activation = lambda x: torch.sigmoid(x - 4) # sigmoid(-3 | -4) = 0.0474 | 0.018


def compute_voxel_center(x: torch.Tensor, camera_center: torch.Tensor, voxel_size: int, far: torch.Tensor) -> torch.Tensor:
    """
    mapping (0, 0, 0) -> mu - f
            (N, N, N) -> mu + f
            
    
    input:
        `x`: [N, 3] int
        `camera_center`: [3]
        
    output:
        [3, N] float
    """
    
    return ((x / voxel_size * 2 * far) - far + camera_center.view(1, 3)).permute(1, 0)



def bitwise_xor(x: torch.Tensor, dim: int):
    assert dim < len(x.shape)
    if x.shape[dim] == 1: return x
    slices = torch.unbind(x, dim=dim)
    result = slices[0]
    for slice in slices[1:]:
        result = torch.bitwise_xor(result, slice)
    return result

def hash_query(x: torch.Tensor, hash_table: torch.Tensor):
    """
    input:
        `x`: [N, 3] int
        `hash_table`: [C, T]
        
    output:
        [C, N]
    """
    PI_1, PI_2, PI_3 = 1, 2654435761, 805459861 # from Instant-NGP
    
    primes = torch.tensor([PI_1, PI_2, PI_3], device=x.device)
    
    hash_index = bitwise_xor(x * primes, dim=-1) % hash_table.shape[1] # (N)
    
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
        index_candidates = torch.arange(voxel_size, dtype=torch.int).cuda()
        x, y, z = torch.meshgrid(index_candidates, index_candidates, index_candidates)
        return torch.stack((x, y, z), dim=-1).view(-1, 3)
    
    # create sparse voxel coordinates from last level
    assert voxel_size % last_voxel_size == 0
    seg_times = voxel_size // last_voxel_size
    seg_times_candidates = torch.arange(seg_times, dtype=torch.int).cuda()
    d_x, d_y, d_z = torch.meshgrid(seg_times_candidates, seg_times_candidates, seg_times_candidates)
    d_grid = torch.stack((d_x, d_y, d_z), dim=-1).view(-1, 3) # (seg_times^3, 1, 3)
    
    result = (last_coordinates * seg_times).view(-1, 1, 3) + d_grid # (seg_times^3, N, 3)
    return result.view(-1, 3)


def create_gaussians_from_features(gaussian_features: torch.Tensor, coordinates: torch.Tensor, extrinsic: torch.Tensor, voxel_size: int, far: torch.Tensor) -> Gaussians:
    """
    input:
        gaussian_features: [C, N]
        coordinates: [N, 3]
        extrinsic: [V, 4, 4]
        far: 1
    """
    camera_center = extrinsic[:, :3, 3].mean(dim=0)
    voxel_center = compute_voxel_center(coordinates, camera_center, voxel_size, far)
    b, n, dim, d_sh = 1, coordinates.shape[0], 3, 1
    
    means: torch.Tensor = delta_means_activation(gaussian_features[SLICE_DELTA_MEANS], far, voxel_size) + voxel_center # (3, N)
    covariances: torch.Tensor = covariance_activation(gaussian_features[SLICE_SCALE], 1, gaussian_features[SLICE_QUATERNION]) # (9, N)
    harmonics: torch.Tensor = gaussian_features[SLICE_SHS] # (3, N)
    opacities: torch.Tensor = opacity_activation(gaussian_features[SLICE_OPACITY]) # (1, N)
    
    gaussians = Gaussians(
        means=means.permute(1, 0).view(b, n, dim),
        covariances=covariances.permute(1, 0).reshape(b, n, dim, dim), 
        harmonics=harmonics.permute(1, 0).view(b, n, 3, d_sh), 
        opacities=opacities.permute(1, 0).view(b, n)
    )
    
    return gaussians


def combine_batch_gaussians(batch_gaussians: list[Gaussians]) -> Gaussians:
    b, dim, d_sh = 1, 3, 1
    gaussian_size = 0
    means_list, covariance_list, harmonics_list, opacities_list = [], [], [], []
    for gaussian in batch_gaussians:
        if gaussian.opacities.shape[1] > gaussian_size:
            gaussian_size = gaussian.opacities[1]
            
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
        
    return Gaussians(
        means = torch.cat(means_list, dim=0), 
        covariances = torch.cat(covariance_list, dim=0), 
        harmonics = torch.cat(harmonics_list, dim=0), 
        opacities = torch.cat(opacities_list, dim=0)
    )


class HashTableVoxelizedGaussianAdapterModule(nn.Module):

    def __init__(self, voxel_size_list=[32, 128, 512], min_opacity=[0.02, 0.05, 0.1]) -> None:
        super().__init__()
        assert len(voxel_size_list) == 3
        self.voxel_size_list = voxel_size_list
        self.min_opacity = min_opacity
        pass
    
        
    def forward(self, multi_scale_hash_tables: tuple, extrinsics: torch.Tensor, fars: torch.Tensor):
        b = multi_scale_hash_tables[0].shape[0]
        far = fars[0, 0]
        batch_gaussians = []
        for batch in range(b):
            coordinates = None
            last_voxel_size = 0
            for scale_idx in range(len(self.voxel_size_list)):
                voxel_size = self.voxel_size_list[scale_idx]
                hash_table = multi_scale_hash_tables[scale_idx][batch] # (C, T)
                coordinates = create_coordinates(voxel_size=voxel_size, last_voxel_size=last_voxel_size, last_coordinates=coordinates)
                gaussian_features = hash_query(x=coordinates, hash_table=hash_table)
                gaussian_mask = opacity_activation(gaussian_features[SLICE_OPACITY]) > self.min_opacity[scale_idx]
                coordinates = coordinates[gaussian_mask[0]]
                last_voxel_size = voxel_size
        
            gaussian_features = gaussian_features[:, gaussian_mask[0]]
            extrinsic = extrinsics[batch]
            gaussian = create_gaussians_from_features(gaussian_features, coordinates, extrinsic, self.voxel_size_list[-1], far)
            batch_gaussians.append(gaussian)
        
        
        return combine_batch_gaussians(batch_gaussians)