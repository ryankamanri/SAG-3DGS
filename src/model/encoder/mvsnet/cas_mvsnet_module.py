from dataclasses import dataclass
import torch
from torch import nn
from ..mvsnet import CascadeMVSNet, generate_point_cloud_from_depth_maps, generate_depth_map_based_point_cloud



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
    """
    vertices: torch.Tensor
    vertices_confidence: torch.Tensor
    
def empty_probalility_point_cloud_result():
    return ViewBasedPointCloudResult(torch.tensor(0), torch.tensor(0))

@dataclass
class CasMVSNetModuleResult:
    ref_view_result_list: list[ReferenceViewResult]
    registed_pcd: PointCloudResult
    registed_prob_pcd: ViewBasedPointCloudResult
    
def empty_cas_mvsnet_module_result():
    return CasMVSNetModuleResult([], empty_point_cloud_result(), empty_probalility_point_cloud_result())

class CasMVSNetModule(nn.Module):

    def __init__(self, cas_mvsnet_ckpt_path, ndepths=[48, 32, 8], use_backbone=True, load_to_backbone=False) -> None:
        super().__init__()
        self.ndepths = ndepths
        self.use_backbone = use_backbone
        print(f"loading checkpoint from {cas_mvsnet_ckpt_path}...")
        # initialize pretrained mvsnet
        state_dict = torch.load(cas_mvsnet_ckpt_path)
        
        if use_backbone:
            self.pretrained_cas_mvsnet = CascadeMVSNet(ndepths=ndepths, return_photometric_confidence=True)
            self.backbone_cas_mvsnet = CascadeMVSNet(ndepths=ndepths, return_prob_volume=True, return_photometric_confidence=True)
        else:
            self.pretrained_cas_mvsnet = CascadeMVSNet(ndepths=ndepths, return_prob_volume=True, return_photometric_confidence=True)
            
        self.pretrained_cas_mvsnet.load_state_dict(state_dict["model"])
        self.pretrained_cas_mvsnet.eval()
        
        if use_backbone and load_to_backbone:
            self.backbone_cas_mvsnet.load_state_dict(state_dict["model"])
        
    def preprocess(self, imgs: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, nears: torch.Tensor, fars: torch.Tensor, ndepths = 192):
        b, v, c, h, w = imgs.shape
        # make the intrinsic mat adapt to feature map (w / 4, w / 4)
        cloned_intrinsics = intrinsics.clone()
        cloned_intrinsics[..., :2, :] /= 4
        
        # multi-stage proj_mats
        # proj_matrices (B, V, 2(intr & extr), 4, 4)
        proj_matrices = torch.zeros(b, v, 2, 4, 4, device="cuda")
        proj_matrices[..., 0, :, :] = extrinsics
        proj_matrices[..., 1, :3, :3] = cloned_intrinsics
        
        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 2
        stage3_pjmats = proj_matrices.clone()
        stage3_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 4

        proj_mat = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }
        
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
        
    def forward(self, imgs, extrinsics, intrinsics, nears, fars, is_trainning: bool):
        proj_mat, depth_values = self.preprocess(imgs, extrinsics, intrinsics, nears, fars)
        b, v, c, h, w = imgs.shape
        
        result = empty_cas_mvsnet_module_result()
        
        pretrained_depths_est = [] # depth map list
        pretrained_photometric_confidences = []
        backbone_depths_est = []
        backbone_photometric_confidences = []
        
        pretrained_outputs_list = []
        if is_trainning:
            with torch.no_grad(): # necessary to reduce the memory
                pretrained_outputs_list = self.pretrained_cas_mvsnet(imgs, proj_mat, depth_values) # depth and photometric_confidence
        
        if self.use_backbone:
            backbone_outputs_list = self.backbone_cas_mvsnet(imgs, proj_mat, depth_values)
        else:
            backbone_outputs_list = pretrained_outputs_list
            
        # for every reference image, the mvsnet will generate a depth map and a photometric confidence map
        for vi in range(v):
            if is_trainning:
                pretrained_outputs = pretrained_outputs_list[vi]
                pretrained_depths_est.append(pretrained_outputs["depth"])
                pretrained_photometric_confidences.append(pretrained_outputs["photometric_confidence"])
                
            backbone_outputs = backbone_outputs_list[vi]
            backbone_depths_est.append(backbone_outputs["depth"])
            backbone_photometric_confidences.append(backbone_outputs["photometric_confidence"])
            
            result.ref_view_result_list.append(ReferenceViewResult(imgs[:, vi], pretrained_outputs, backbone_outputs))
        
        vertices, vertices_color = [], []
        if is_trainning:
            with torch.no_grad():            
                vertices, vertices_color = generate_point_cloud_from_depth_maps(imgs, extrinsics, intrinsics, pretrained_depths_est, pretrained_photometric_confidences)
            
        prob_vertices = generate_depth_map_based_point_cloud(backbone_depths_est, extrinsics, intrinsics)
        
        if False:
            assert b == 1
            import open3d
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(prob_vertices.permute(0, 1, 3, 4, 2).reshape(b*v*h*w, 4)[..., :3].detach().cpu())
            pcd.colors = open3d.utility.Vector3dVector(imgs.permute(0, 1, 3, 4, 2).reshape(b*v*h*w, 3).detach().cpu())
            open3d.visualization.draw_geometries([pcd])      
        
        result.registed_pcd = PointCloudResult(xyz_batches=vertices, rgb_batches=vertices_color)
        result.registed_prob_pcd = ViewBasedPointCloudResult(
            vertices=prob_vertices, 
            vertices_confidence=torch.stack(backbone_photometric_confidences, dim=1))
        
        return result