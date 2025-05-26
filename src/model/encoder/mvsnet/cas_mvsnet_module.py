from dataclasses import dataclass
import torch
from torch import nn
from ..mvsnet import CascadeMVSNet, generate_depth_map_based_point_cloud, generate_geometric_mask
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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
    registed_pcd: ViewBasedPointCloudResult
    registed_prob_pcd: ViewBasedPointCloudResult
    
def empty_cas_mvsnet_module_result():
    return CasMVSNetModuleResult([], empty_view_based_point_cloud_result(), empty_view_based_point_cloud_result())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cameras(t_pred_aligned, t_gt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t_pred_aligned[:,0], t_pred_aligned[:,1], t_pred_aligned[:,2], c='r', label='Aligned Pred')
    ax.scatter(t_gt[:,0], t_gt[:,1], t_gt[:,2], c='b', label='GT')
    ##################################
    labels = range(t_pred_aligned.shape[0])
    for l in labels:
        # 在点的右上方添加序号（偏移量可调整）
        ax.text(t_pred_aligned[l][0] + 0.01, t_pred_aligned[l][1] + 0.01, t_pred_aligned[l][2], str(l), 
                fontsize=10, color='red', 
                ha='left', va='bottom')  # 水平对齐：左，垂直对齐：底
        ax.text(t_gt[l][0] + 0.01, t_gt[l][1] + 0.01, t_gt[l][2], str(l), 
                fontsize=10, color='blue', 
                ha='left', va='bottom')  # 水平对齐：左，垂直对齐：底
    #####################################
    ax.legend()
    plt.show()
    plt.waitforbuttonpress()


def umeyama_alignment_with_scale_batch(t_pred, t_gt):
    """
    带尺度的 Umeyama 算法（批量处理）
    Args:
        t_pred: (B, N, 3) 预测的平移向量
        t_gt:   (B, N, 3) 真实的平移向量
    Returns:
        R:      (B, 3, 3) 旋转矩阵
        t:      (B, 3)    平移向量
        s:      (B,)      尺度因子
    """
    B, N, _ = t_pred.shape
    
    # 计算质心
    mu_pred = torch.mean(t_pred, dim=1, keepdim=True)  # (B, 1, 3)
    mu_gt = torch.mean(t_gt, dim=1, keepdim=True)      # (B, 1, 3)
    
    # 去中心化
    X = t_pred - mu_pred  # (B, N, 3)
    Y = t_gt - mu_gt     # (B, N, 3)
    
    # 协方差矩阵 H = X^T @ Y
    H = torch.matmul(X.transpose(1, 2), Y)  # (B, 3, 3)
    
    # SVD 分解
    U, S, V = torch.linalg.svd(H)  # (B, 3, 3), (B, 3), (B, 3, 3)
    
    # 计算旋转矩阵 R = V @ U^T，并校正行列式符号
    R = torch.matmul(V, U.transpose(1, 2))  # (B, 3, 3)
    # visualize_cameras((R @ X.transpose(1, 2)).transpose(1, 2).reshape(-1, 3).cpu(), Y.reshape(-1, 3).cpu())
    # def angle(vec1, vec2):
    #     return torch.arccos((vec1 * vec2).sum() / vec1.norm() / vec2.norm()) * 180 / (3.1415926)
    det = torch.det(R)                      # (B,)
    sign = torch.sign(det)
    V_corrected = V * sign.view(-1, 1, 1)   # 校正符号
    R = torch.matmul(V_corrected, U.transpose(1, 2))
    
    # 计算尺度因子 s = tr(Σ) / tr(X^T X)
    sigma = S.sum(dim=1)  # tr(Σ) = sum(S的对角线)
    X_var = torch.sum(X ** 2, dim=(1, 2))                 # tr(X^T X)
    s = sigma / X_var                                     # (B,)
    
    # 计算平移向量 t = μ_gt - s * R @ μ_pred
    mu_pred_squeezed = mu_pred.squeeze(1)  # (B, 3)
    R_mu_pred = torch.matmul(R, mu_pred_squeezed.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    t = mu_gt.squeeze(1) - s.unsqueeze(-1) * R_mu_pred  # (B, 3)
    
    return R, t, s

def align_pred_to_gt_batch(T_pred, T_gt, depth_pred):
    """
    批量对齐预测的外参和深度（带尺度估计）
    Args:
        T_pred:    (B, N, 4, 4) 预测的外参矩阵
        T_gt:      (B, N, 4, 4)    真实的外参矩阵
        depth_pred: (B, N, H, W) 预测的深度图
    Returns:
        T_aligned: (B, N, 4, 4) 对齐后的外参矩阵
        depth_scaled: (B, N, H, W) 缩放后的深度图
    """
    B, N, H, W = depth_pred.shape
    
    t_gt = T_gt[:, :, :3, 3]  # (B, N, 3)
    
    # Step 1: 联合估计 R, t, s
    R, t, s = umeyama_alignment_with_scale_batch(T_pred[:, :, :3, 3], t_gt)
    
    # Step 2: 对齐外参
    R_pred = T_pred[:, :, :3, :3]  # (B, N, 3, 3)
    t_pred = T_pred[:, :, :3, 3]   # (B, N, 3)
    
    # 应用旋转和缩放：R_aligned = R @ R_pred, t_aligned = s * (R @ t_pred) + t
    R_aligned = torch.matmul(R.unsqueeze(1), R_pred)  # (B, N, 3, 3)
    t_aligned = s.unsqueeze(-1).unsqueeze(1) * torch.matmul(R.unsqueeze(1), t_pred.unsqueeze(-1)).squeeze(-1) + t.unsqueeze(1)
    
    # 构建对齐后的外参矩阵
    T_aligned = torch.eye(4, device=T_pred.device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
    T_aligned[:, :, :3, :3] = R_aligned
    T_aligned[:, :, :3, 3] = t_aligned
    
    # Step 3: 缩放深度
    depth_scaled = depth_pred * s.view(B, 1, 1, 1)  # (B, N, H, W)
    
    return T_aligned, depth_scaled

def adapt_size(target_size: tuple, imgs: torch.Tensor, pad_value=1.0):
    b, v, c, h, w = imgs.shape
    target_h, target_w = target_size
    target_imgs = imgs.clone()
    if w < target_w:
        target_imgs = torch.nn.functional.pad(target_imgs, ((target_w - w) // 2, (target_w - w) // 2), value=pad_value)
    else:
        start_x = (w - target_w) // 2
        target_imgs = target_imgs[:, :, :, :, start_x:start_x + target_w]
        
    if h < target_h:
        target_imgs = torch.nn.functional.pad(target_imgs, (0, 0, (target_h - h) // 2, (target_h - h) // 2), value=pad_value)
    else:
        start_y = (h - target_h) // 2
        target_imgs = target_imgs[:, :, :, start_y:start_y + target_h, :]
        
    assert target_imgs.shape[3] == target_h and target_imgs.shape[4] == target_w, f"target_imgs shape error: {target_imgs.shape}"
    return target_imgs

class CasMVSNetModule(nn.Module):

    def __init__(self, cas_mvsnet_ckpt_path, ndepths=[48, 32, 8], geo_max_dist=0.001, geo_max_depth_diff=0.001, use_backbone=True, load_to_backbone=False) -> None:
        super().__init__()
        self.ndepths = ndepths
        self.geo_max_dist = geo_max_dist
        self.geo_max_depth_diff = geo_max_depth_diff
        self.use_backbone = use_backbone
        self.refine = False
        print(f"loading checkpoint from {cas_mvsnet_ckpt_path}...")
        # initialize pretrained mvsnet
        state_dict = torch.load(cas_mvsnet_ckpt_path)
        
        if use_backbone:
            self.pretrained_cas_mvsnet = CascadeMVSNet(refine=False, ndepths=ndepths, return_photometric_confidence=True)
            self.backbone_cas_mvsnet = CascadeMVSNet(use_dot_similarity=False, cr_base_chs=[32, 16, 8], refine=self.refine, ndepths=ndepths, return_volume=True, return_photometric_confidence=True)
        else:
            self.pretrained_cas_mvsnet = CascadeMVSNet(refine=False, ndepths=ndepths, return_volume=True, return_photometric_confidence=True)
            
        self.pretrained_cas_mvsnet.load_state_dict(state_dict["model"])
        self.pretrained_cas_mvsnet.eval()
        
        if use_backbone and load_to_backbone:
            self.backbone_cas_mvsnet.load_state_dict(state_dict["model"])
            
        # VGGT initial
        self.use_vggt = False
        if self.use_vggt:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vggt = VGGT()
            self.vggt.load_state_dict(torch.load("pretrained/vggt/model.pt"))
            self.vggt.eval()
            
        
    def preprocess(self, imgs: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, nears: torch.Tensor, fars: torch.Tensor, ndepths = 192):
        b, v, c, h, w = imgs.shape
        # make the intrinsic mat adapt to feature map (w / 4, w / 4)
        cloned_intrinsics = intrinsics.clone()
        cloned_intrinsics[..., :2, :] /= 4
        
        # multi-stage proj_mats
        # proj_matrices (B, V, 2(intr & extr), 4, 4)
        proj_matrices = torch.zeros(b, v, 2, 4, 4, device="cuda")
        proj_matrices[..., 0, :, :] = extrinsics.inverse()
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
        
        # to be compatible with VGGT
        # pad images to 518x518
        target_size = (518, 518)
        vggt_imgs = adapt_size(target_size, imgs, pad_value=1.0)
        
        return proj_mat, depth_values, vggt_imgs
        
    def forward(self, context, imgs, img_masks, extrinsics, intrinsics, nears, fars, outer_features=None):
        proj_mat, depth_values, vggt_imgs = self.preprocess(imgs, extrinsics, intrinsics, nears, fars)
        near_fars = torch.stack([nears, fars], dim=-1) # (B, V, 2)
        b, v, c, h, w = imgs.shape
        
        result = empty_cas_mvsnet_module_result()
        
        pretrained_depths_est = [] # depth map list
        pretrained_photometric_confidences = [] # photometric confidence map list
        pretrained_geo_masks = [] # geometric mask list
        backbone_depths_est = []
        backbone_photometric_confidences = []
        backbone_geo_masks = []
        
        pretrained_outputs_list = []
        if False:
            with torch.no_grad(): # necessary to reduce the memory
                pretrained_outputs_list = self.pretrained_cas_mvsnet(imgs, proj_mat, depth_values) # depth and photometric_confidence
        
        if self.use_backbone:
            backbone_outputs_list = self.backbone_cas_mvsnet.forward(imgs, proj_mat, depth_values, outer_features)
        elif not self.training:
            backbone_outputs_list = self.pretrained_cas_mvsnet(imgs, proj_mat, depth_values)
        else:
            backbone_outputs_list = pretrained_outputs_list
            
        if self.use_vggt and self.training:
            with torch.inference_mode():
                predictions = self.vggt(vggt_imgs)
                vggt_extrinsics_3x4, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], build_intrinsics=False)
                vggt_extrinsics = torch.eye(4, device=imgs.device).unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1)
                vggt_extrinsics[:, :, :3, :4] = vggt_extrinsics_3x4
                vggt_extrinsics_aligned, depth_values = align_pred_to_gt_batch(vggt_extrinsics, extrinsics, predictions["depth"].squeeze(-1))
                
                # visualize_cameras(vggt_extrinsics_aligned[:, :, :3, 3].reshape(-1, 3).cpu(), extrinsics[:, :, :3, 3].reshape(-1, 3).cpu())
                context["depth"] = adapt_size(imgs.shape[3:], depth_values.unsqueeze(2), pad_value=0.0).squeeze(2)
                # TODO: handle abnormal depth_mask values (>1)
                context["depth_mask"] = adapt_size(imgs.shape[3:], predictions["depth_conf"].unsqueeze(2), pad_value=0.0).squeeze(2)
            
        # for every reference image, the mvsnet will generate a depth map and a photometric confidence map
        for vi in range(v):
            pretrained_outputs = {}
            if self.training:
                # pretrained_outputs = pretrained_outputs_list[vi]
                pretrained_depths_est.append(context["depth"][:, vi])
                pretrained_photometric_confidences.append(context["depth_mask"][:, vi])
                
            backbone_outputs = backbone_outputs_list[vi]
            backbone_depths_est.append(backbone_outputs["depth"])
            backbone_photometric_confidences.append(backbone_outputs["photometric_confidence"])
            
            result.ref_view_result_list.append(ReferenceViewResult(imgs[:, vi], pretrained_outputs, backbone_outputs))
        
        if self.training:
            with torch.no_grad():            
                vertices = generate_depth_map_based_point_cloud(pretrained_depths_est, extrinsics, intrinsics)
                # for vi in range(v):
                #     pretrained_geo_mask, _ = generate_geometric_mask(imgs, extrinsics, intrinsics, pretrained_depths_est, near_fars,
                #                                                      ref_idx=vi, max_dist=self.geo_max_dist, max_depth_diff=self.geo_max_depth_diff)
                #     backbone_geo_mask, _ = generate_geometric_mask(imgs, extrinsics, intrinsics, backbone_depths_est, near_fars, 
                #                                                 ref_idx=vi, max_dist=self.geo_max_dist, max_depth_diff=self.geo_max_depth_diff)
                #     pretrained_geo_masks.append(pretrained_geo_mask)
                #     backbone_geo_masks.append(backbone_geo_mask)
                    
                result.registed_pcd = ViewBasedPointCloudResult(
                    vertices=vertices, 
                    vertices_confidence=torch.stack(pretrained_photometric_confidences, dim=1),
                    vertices_geometry_mask=torch.stack(pretrained_geo_masks, dim=1) if len(pretrained_geo_masks) > 0 else torch.tensor(0))
            
        prob_vertices = generate_depth_map_based_point_cloud(backbone_depths_est, extrinsics, intrinsics)
        
        if False:
            assert b == 1
            import open3d
            masks = torch.logical_and(img_masks, torch.stack(backbone_geo_masks, dim=1)) if len(backbone_geo_masks) > 0 else img_masks
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(prob_vertices.permute(0, 1, 3, 4, 2)[masks][..., :3].detach().cpu())
            pcd.colors = open3d.utility.Vector3dVector(imgs.permute(0, 1, 3, 4, 2)[masks].detach().cpu())
            open3d.visualization.draw_geometries([pcd])      
        
        result.registed_prob_pcd = ViewBasedPointCloudResult(
            vertices=prob_vertices, 
            vertices_confidence=torch.stack(backbone_photometric_confidences, dim=1), 
            vertices_geometry_mask=torch.stack(backbone_geo_masks, dim=1) if len(backbone_geo_masks) > 0 else torch.tensor(0))
        
        return result