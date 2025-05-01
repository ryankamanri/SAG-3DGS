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
        costvolume_feature_channels: int, 
        out_channels: int = 48,
    ):
        super(CostvolumeSampler, self).__init__()
        self.max_voxels_foreach_processing = max_voxels_foreach_processing
        self.feature_channels = costvolume_feature_channels
        self.out_channels = out_channels
        self.feature_enhancer = nn.Sequential(
            nn.Linear(costvolume_feature_channels, costvolume_feature_channels),
            nn.GELU(),
            nn.Linear(costvolume_feature_channels, costvolume_feature_channels),
            nn.GELU(),
            nn.Linear(costvolume_feature_channels, costvolume_feature_channels),
        )
        self.weight_predictor = nn.Sequential(
            nn.Linear(costvolume_feature_channels+3, costvolume_feature_channels),
            nn.GELU(),
            nn.Linear(costvolume_feature_channels, 8),
            nn.GELU(), 
            nn.Linear(8, 1), 
        )
        
        self.out_predictor = nn.Sequential(
            nn.Linear(costvolume_feature_channels, costvolume_feature_channels),
            nn.GELU(),
            nn.Linear(costvolume_feature_channels, out_channels)
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
        if vox == 0: return torch.zeros(vox, self.out_channels, device=gaussian_means.device)
        _, v, _, h, w = cas_module_result.registed_prob_pcd.vertices.shape # (B, V, 4, H, W)
        stage_volumes, stage_near_fars = [], []
        for stage in range(3):
            stage_volumes.append([
                cas_module_result.ref_view_result_list[vi].backbone["stage{}".format(stage + 1)]["volume"][batch_idx].unsqueeze(0)
                 for vi in range(v)]) # stage_volumes: [[(1, C, D, H, W) * V] * 3]
            stage_near_fars.append([
                cas_module_result.ref_view_result_list[vi].backbone["stage{}".format(stage + 1)]["depth_near_far"][batch_idx]
                 for vi in range(v)]) # stage_near_far_invs: [[(2, H, W) * V] * 3]
            pass 
        
        merged_source_slice_list = []
        
        for si in SliceIterator(0, vox, self.max_voxels_foreach_processing):
            voxi = si.stop - si.start
            means_slice = gaussian_means[si].permute(1, 0).unsqueeze(0) # (1, 3, Voxi)
            means_cam_slice = torch.matmul(torch.linalg.inv(extrinsic), torch.cat((means_slice, torch.ones(1, 1, voxi).to(means_slice.device)), dim=1))[:, :3] # (1, 4, Vox) -> (V, 3, Vox)
            
            stage = 0 # only one stage
            sampled_features = []
            # we handle volume individually cause it is too large
            for vidx in range(v):
                volume = stage_volumes[stage][vidx] # (V=1, C, D, H, W)
                vi, c, d, _, _ = volume.shape # (V=1, C, D, H, W)
                prop = 1 / (2 ** (2 - stage))
                stage_intrinsics = intrinsic[vidx].unsqueeze(0).clone() # V -> 1
                stage_intrinsics[:, :2] *= prop # (V, 3, 3) 4 -> 2 -> 1
                means_uvd_slice = torch.matmul(stage_intrinsics, means_cam_slice[vidx].unsqueeze(0)) # (V, 3, Voxi)
                means_uvd_slice = torch.stack((
                    means_uvd_slice[:, 0] / means_uvd_slice[:, 2], 
                    means_uvd_slice[:, 1] / means_uvd_slice[:, 2], 
                    means_uvd_slice[:, 2]), dim=1) # (V, 3, Voxi)
                
                # normalize
                means_uvd_norm_slice = torch.stack((
                    (means_uvd_slice[:, 0] / ((w * prop - 1) / 2)) - 1, 
                    (means_uvd_slice[:, 1] / ((h * prop - 1) / 2)) - 1, 
                    (((means_uvd_slice[:, 2] - near[vidx].view(vi, 1)) / ((far[vidx] - near[vidx]).view(vi, 1) / 2)) - 1)), dim=1) # (V, 3, Voxi)

                sampled_feature = F.grid_sample(
                    volume.view(vi, c, d, int(h*prop), int(w*prop)), 
                    means_uvd_norm_slice.permute(0, 2, 1).view(vi, 1, 1, voxi, 3),
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=True
                ).view(vi, c, voxi)
                sampled_features.append(sampled_feature)
            
            sampled_feature = torch.cat(sampled_features, dim=0) # (V, C, Voxi)
            sampled_feature = self.feature_enhancer(sampled_feature.permute(0, 2, 1)).permute(0, 2, 1) # (V, C, Voxi)
            weighted_features = self.weight_features(sampled_feature, means_slice.squeeze(0), extrinsic) # (V, 1, Voxi)
            sampled_feature = sampled_feature * torch.softmax(weighted_features, dim=0) # (V, C, Voxi)
            sampled_feature = self.out_predictor(sampled_feature.permute(0, 2, 1)).permute(0, 2, 1) # (V, C, Voxi)
            merged_source_slice_list.append(torch.sum(sampled_feature, dim=0)) # (C, Voxi)
            del sampled_feature
        
        return torch.cat(merged_source_slice_list, dim=1).transpose(0, 1) # (C, Vox) -> (Vox, C)
