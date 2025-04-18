import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch3d.ops import knn_points

from ...encoder.mvsnet.cas_mvsnet_module import CasMVSNetModuleResult
from ....misc.slice_iterator import SliceIterator


class CostvolumeSampler(nn.Module):
    def __init__(
        self, 
        max_voxels_foreach_processing: int,
        costvolume_feature_channels: int
    ):
        super(CostvolumeSampler, self).__init__()
        self.max_voxels_foreach_processing = max_voxels_foreach_processing
        self.feature_channels = costvolume_feature_channels
        self.weight_predictor = nn.Sequential(
            nn.Linear(costvolume_feature_channels+3, costvolume_feature_channels),
            nn.GELU(),
            nn.Linear(costvolume_feature_channels, 16),
            nn.Linear(16, 8), 
            nn.GELU(), 
            nn.Linear(8, 4), 
            nn.Linear(4, 1), 
            nn.GELU()
        )
        
    def weight_features(
        self, 
        features: torch.Tensor, # (V, C, vox)
        point_xyz: torch.Tensor, # (3, vox)
        extrinsics: torch.Tensor, # (V, 4, 4)
        ):
        
        v, c, vox = features.shape
        cam_points = extrinsics[:, :3, 3].view(v, 3, 1) # (V, 3, 1)
        
        point_to_cam = cam_points - point_xyz.unsqueeze(0) # (V, 3, vox)
        point_to_cam = point_to_cam / torch.norm(point_to_cam, dim=1, keepdim=True) # (V, 3, vox)
        
        weighted_features = self.weight_predictor(
            torch.cat((features, point_to_cam), dim=1).permute(0, 2, 1).reshape(v*vox, c+3)
        ).reshape(v, vox, 1).permute(0, 2, 1) # (V, 1, vox)
        
        return weighted_features

    def forward(
        self,
        gaussian_means: torch.Tensor, # (Vox, 3)
        cas_module_result: CasMVSNetModuleResult, 
        extrinsic: torch.Tensor, # (V, 4, 4)
        intrinsic: torch.Tensor, # (V, 3, 3)
        near: torch.Tensor, # (V)
        far: torch.Tensor, # (V)
        batch_idx: int
    ):
        vox, _, = gaussian_means.shape
        _, v, _, h, w = cas_module_result.registed_prob_pcd.vertices.shape # (B, V, 4, H, W)
        stage_volumes, stage_near_far_invs = [], []
        for stage in range(3):
            stage_volumes.append([
                cas_module_result.ref_view_result_list[vi].backbone["stage{}".format(stage + 1)]["volume"][batch_idx]
                 for vi in range(v)]) # stage_volumes: [[(C, D, H, W) * V] * 3]
            stage_near_far_invs.append([
                cas_module_result.ref_view_result_list[vi].backbone["stage{}".format(stage + 1)]["depth_near_far_inv"][batch_idx]
                 for vi in range(v)]) # stage_near_far_invs: [[(2, H, W) * V] * 3]
            pass 
        
        merged_source_slice_list = []
        
        for si in SliceIterator(0, vox, self.max_voxels_foreach_processing):
            voxi = si.stop - si.start
            means_slice = gaussian_means[si].permute(1, 0).unsqueeze(0) # (1, 3, Voxi)
            means_cam_slice = torch.matmul(torch.linalg.inv(extrinsic), torch.cat((means_slice, torch.ones(1, 1, voxi).to(means_slice.device)), dim=1))[:, :3] # (1, 4, Vox) -> (1, 3, Vox)
            
            stage = 0 # only one stage
            volumes = torch.stack(stage_volumes[stage], dim=0) # (V, C, D, H, W)
            _, c, d, _, _ = volumes.shape # (V, C, D, H, W)
            near_inv, far_inv = 1.0 / near, 1.0 / far # (V, 2)
            prop = 1 / (2 ** (2 - stage))
            stage_intrinsics = intrinsic.clone()
            stage_intrinsics[:, :2] *= prop # (V, 3, 3) 4 -> 2 -> 1
            means_uvd_slice = torch.matmul(stage_intrinsics, means_cam_slice) # (V, 3, Voxi)
            means_uvd_slice = torch.stack((
                means_uvd_slice[:, 0] / means_uvd_slice[:, 2], 
                means_uvd_slice[:, 1] / means_uvd_slice[:, 2], 
                means_uvd_slice[:, 2]), dim=1) # (V, 3, Voxi)
            
            means_uv_invd_slice = means_uvd_slice.clone() # (V, 3, Voxi)
            means_uv_invd_slice[:, 2] = 1.0 / means_uvd_slice[:, 2] # the depth sample is not linear, so we need to inverse it for linear interpolation
            # because u, v, 1/d are all linear so we use uv_invd to sample features.
            # normalize
            means_uv_invd_norm_slice = torch.stack((
                (means_uv_invd_slice[:, 0] / ((w * prop - 1) / 2)) - 1, 
                (means_uv_invd_slice[:, 1] / ((h * prop - 1) / 2)) - 1, 
                ((means_uv_invd_slice[:, 2] - far_inv.view(v, 1)) / ((near_inv - far_inv).view(v, 1) / 2)) - 1), dim=1) # (V, 3, Voxi)

            sampled_feature = F.grid_sample(
                volumes.view(v, c, d, int(h*prop), int(w*prop)), 
                means_uv_invd_norm_slice.permute(0, 2, 1).view(v, 1, 1, voxi, 3),
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            ).view(v, c, voxi)

            weighted_features = self.weight_features(sampled_feature, means_slice.squeeze(0), extrinsic) # (V, 1, Voxi)
            sampled_feature = sampled_feature * torch.softmax(weighted_features, dim=0) # (V, C, Voxi)
            merged_source_slice_list.append(torch.sum(sampled_feature, dim=0)) # (C, Voxi)
            del sampled_feature
        
        return torch.cat(merged_source_slice_list, dim=1).transpose(0, 1) # (C, Vox) -> (Vox, C)
