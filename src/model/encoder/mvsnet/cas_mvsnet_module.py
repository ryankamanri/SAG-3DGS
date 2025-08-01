from dataclasses import dataclass
import torch
from torch import nn
from ..mvsnet import CascadeMVSNet, generate_depth_map_based_point_cloud, generate_geometric_mask
import torch.nn.functional as F

@dataclass
class ReferenceViewResult:
    img: torch.Tensor # (B, C, H, W)
    pretrained: dict[str, object]
    backbone: dict[str, object]
    
def empty_reference_view_result():
    return ReferenceViewResult(torch.tensor(0), {}, {})

@dataclass
class PointCloudResult:
    """
    # A data class stored batches of xyz1 homogeneous coordinates `[(n_points, 4) * B]` and rgb colors `[(n_points, 3) * B]`.
    """
    xyz_batches: list[torch.Tensor]
    rgb_batches: list[torch.Tensor]

def empty_point_cloud_result():
    return PointCloudResult([], [])

@dataclass
class ViewBasedPointCloudResult:
    """
        vertices: Tensor(B, V, 4, H, W)
        vertices_confidence: Tensor(B, V, H, W)
        vertices_geometry_mask: Tensor(B, V, H, W)
    """
    vertices: torch.Tensor
    vertices_confidence: torch.Tensor
    vertices_geometry_mask: torch.Tensor
    
def empty_view_based_point_cloud_result():
    return ViewBasedPointCloudResult(torch.tensor(0), torch.tensor(0), torch.tensor(0))

@dataclass
class CasMVSNetModuleResult:
    ref_view_result_list: list[ReferenceViewResult]
    registed_pcd: dict[str, ViewBasedPointCloudResult]
    registed_prob_pcd: dict[str, ViewBasedPointCloudResult]
    
def empty_cas_mvsnet_module_result():
    return CasMVSNetModuleResult([], {}, {})


class CasMVSNetModule(nn.Module):

    def __init__(self, feat_scales, cas_mvsnet_ckpt_path, ndepths=[48, 32, 8], cr_base_chs=[32, 16, 8], in_channels=[64, 48, 32], geo_max_dist=0.001, geo_max_depth_diff=0.001, use_backbone=True, load_to_backbone=False) -> None:
        super().__init__()
        self.feat_scales = feat_scales
        self.ndepths = ndepths
        self.geo_max_dist = geo_max_dist
        self.geo_max_depth_diff = geo_max_depth_diff
        self.use_backbone = use_backbone
        self.refine = False
        assert len(feat_scales) == len(ndepths) == len(cr_base_chs), "feat_scales, ndepths and cr_base_chs must have the same length."
        self.feat_scales = feat_scales
        self.stages = [f"stage{i+1}" for i in range(len(feat_scales))]
        # print(f"loading checkpoint from {cas_mvsnet_ckpt_path}...")
        # initialize pretrained mvsnet
        # state_dict = torch.load(cas_mvsnet_ckpt_path)
        
        if use_backbone:
            # TODO: remove pretrained_cas_mvsnet
            # self.pretrained_cas_mvsnet = CascadeMVSNet(refine=False, ndepths=ndepths, return_photometric_confidence=True)
            self.backbone_cas_mvsnet = CascadeMVSNet(
                use_dot_similarity=False, 
                cr_base_chs=cr_base_chs, 
                in_channels=in_channels, 
                refine=self.refine, 
                ndepths=ndepths, 
                return_volume=True, 
                return_photometric_confidence=True)
        # else:
        #     self.pretrained_cas_mvsnet = CascadeMVSNet(refine=False, ndepths=ndepths, return_volume=True, return_photometric_confidence=True)
            
        # self.pretrained_cas_mvsnet.load_state_dict(state_dict["model"])
        # self.pretrained_cas_mvsnet.eval()
        
        # if use_backbone and load_to_backbone:
        #     self.backbone_cas_mvsnet.load_state_dict(state_dict["model"])
            
            
        
    def preprocess(self, imgs: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, nears: torch.Tensor, fars: torch.Tensor, ndepths = 192):
        b, v, c, h, w = imgs.shape
        
        # Initialize projection matrix dictionary
        proj_mat = {}
        
        # proj_matrices (B, V, 2(intr & extr), 4, 4)
        # Base projection matrix (highest resolution stage)
        base_proj = torch.zeros(b, v, 2, 4, 4, device="cuda")
        base_proj[..., 0, :, :] = extrinsics.inverse()  # extr inv
        base_proj[..., 1, :3, :3] = intrinsics.clone() # origin intr
        
        # Create projection matrices for each scale
        for stage, scale in zip(self.stages, self.feat_scales):
            scaled_proj = base_proj.clone()
            # Adjust the internal parameter matrix to adapt to scaling
            scaled_proj[..., 1, :2, :] = base_proj[..., 1, :2, :] / scale
            proj_mat[stage] = scaled_proj
        
        # the intrinsics adapts depth map (w, h), not (w / 4, h / 4)
        # intrinsics[..., :2, :] *= 4
        
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = (1.0 / fars).view(b*v, 1)
        max_depth = (1.0 / nears).view(b*v, 1)
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, ndepths).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        )
        depth_values = 1 / depth_candi_curr # (B*V, ndepths)
        depth_values = depth_values.reshape(b, v, ndepths) # (B, V, ndepths)
        depth_values = depth_values.flip(dims=(2,)) # start from near to far.
        
        return proj_mat, depth_values
        
    def forward(self, context, imgs, img_masks, extrinsics, intrinsics, nears, fars, is_training, outer_features=None):
        proj_mat, depth_values = self.preprocess(imgs, extrinsics, intrinsics, nears, fars)
        near_fars = torch.stack([nears, fars], dim=-1) # (B, V, 2)
        b, v, c, h, w = imgs.shape
        is_training = is_training
        result = empty_cas_mvsnet_module_result()
        
        def empty_stage_list():
            res = {}
            for stage in self.stages:
                res[stage] = []
            return res
        
        pretrained_depths_est = empty_stage_list()  # depth map list
        pretrained_photometric_confidences = empty_stage_list() # photometric confidence map list
        pretrained_geo_masks = empty_stage_list() # geometric mask list
        backbone_depths_est = empty_stage_list()
        backbone_photometric_confidences = empty_stage_list()
        backbone_geo_masks = empty_stage_list()
        
        
        if self.use_backbone:
            backbone_outputs_list = self.backbone_cas_mvsnet.forward(imgs, proj_mat, depth_values, outer_features)
            
        stages = self.stages
            
        # for every reference image, the mvsnet will generate a depth map and a photometric confidence map
        for vi in range(v):
            pretrained_outputs = {}
            # if is_training:
            #     # pretrained_outputs = pretrained_outputs_list[vi]
            #     for stage, idx in zip(stages, range(len(stages))):
            #         prop = 1.0 / self.feat_scales[idx]
            #         pretrained_depths_est[stage].append(F.interpolate(context["depth"][:, vi].unsqueeze(1), scale_factor=prop, mode="bilinear").squeeze(1))
            #         pretrained_photometric_confidences[stage].append(F.interpolate(context["depth_mask"][:, vi].unsqueeze(1), scale_factor=prop, mode="bilinear").squeeze(1))
                
            backbone_outputs = backbone_outputs_list[vi]
            for stage, idx in zip(stages, range(len(stages))):
                prop = 1.0 / self.feat_scales[idx]
                backbone_depths_est[stage].append(F.interpolate(backbone_outputs["depth"].unsqueeze(1), scale_factor=prop, mode="bilinear").squeeze(1))
                backbone_photometric_confidences[stage].append(F.interpolate(backbone_outputs["photometric_confidence"].unsqueeze(1), scale_factor=prop, mode="bilinear").squeeze(1))
                
            result.ref_view_result_list.append(ReferenceViewResult(imgs[:, vi], pretrained_outputs, backbone_outputs))
        
        # if is_training:
        #     with torch.no_grad():            
        #         for stage, idx in zip(stages, range(len(stages))):
        #             result.registed_pcd[stage] = ViewBasedPointCloudResult(
        #                 vertices=generate_depth_map_based_point_cloud(pretrained_depths_est[stage], extrinsics, proj_mat[stage][..., 1, :3, :3]), 
        #                 vertices_confidence=torch.stack(pretrained_photometric_confidences[stage], dim=1),
        #                 vertices_geometry_mask=torch.stack(pretrained_geo_masks[stage], dim=1) if len(pretrained_geo_masks[stage]) > 0 else torch.tensor(0))
            
        
        if False:
            assert b == 1
            import open3d
            masks = torch.logical_and(img_masks, torch.stack(backbone_geo_masks, dim=1)) if len(backbone_geo_masks) > 0 else img_masks
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(prob_vertices.permute(0, 1, 3, 4, 2)[masks][..., :3].detach().cpu())
            pcd.colors = open3d.utility.Vector3dVector(imgs.permute(0, 1, 3, 4, 2)[masks].detach().cpu())
            open3d.visualization.draw_geometries([pcd])      
        for stage, idx in zip(stages, range(len(stages))):
            result.registed_prob_pcd[stage] = ViewBasedPointCloudResult(
                vertices=generate_depth_map_based_point_cloud(backbone_depths_est[stage], extrinsics, proj_mat[stage][..., 1, :3, :3]), 
                vertices_confidence=torch.stack(backbone_photometric_confidences[stage], dim=1), 
                vertices_geometry_mask=torch.stack(backbone_geo_masks[stage], dim=1) if len(backbone_geo_masks[stage]) > 0 else torch.tensor(0))
        
        return result