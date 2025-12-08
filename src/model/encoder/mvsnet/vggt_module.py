from typing import Any, Mapping
import torch
from torch import nn
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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
    
    return T_aligned, depth_pred, s

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


def compute_bounds(depths: torch.Tensor, min_percentile=0.0, max_percentile=0.97, min_depth=0.1, max_depth=400.0):
    """
    #### Use percentile to compute near and far bounds from depth maps.
    
    input:
        depths: (B, V, H, W)
    output:
        near and far bounds
    """
    b, v, h, w = depths.shape
    depths = depths.reshape(b * v, h * w)
    sorted_flat_depth = depths.sort(dim=-1).values # (B*V, H*W)
    nears = sorted_flat_depth[:, int(h * w * min_percentile)].view(b, v) # (B, V)
    nears = torch.clamp(nears, min=min_depth)  # avoid too small near values (0) may be devided by zero in later calculations
    fars = sorted_flat_depth[:, int(h * w * max_percentile)].view(b, v) # (B, V)
    fars = torch.clamp(fars, max=max_depth)  # avoid too large far values (100) may cause overflow in later calculations
    return nears, fars

class VGGTModule():
    def __init__(self, pth_path="pretrained/vggt/model.pt", inference_mode=True, device=torch.device("cuda")):
        """
        VGGTModule is a wrapper for the VGGT model. 
        Note that we don't extend nn.Module here because we want to use the inference mode of VGGT. VGGT module will not be managed by pytorch_lightning.
        Args:
            pth_path: path to the pretrained model
            inference_mode: whether to use inference mode
            device: device to run the model on
        """
        super().__init__()
        self.vggt = VGGT().to(device)
        self.vggt.load_state_dict(torch.load(pth_path))
        self.vggt.eval()
        self.inference_mode = inference_mode
         
    def forward(self, imgs: torch.Tensor, extrinsics: torch.Tensor):
        """
        imgs: (B, V, C, H, W)
        extrinsics: (B, V, 4, 4)
        intrinsics: (B, V, 3, 3)
        """
        b, v, c, h, w = imgs.shape
        # to be compatible with VGGT
        # pad images to 518x518
        target_size = (518, 518)
        vggt_imgs = adapt_size(target_size, imgs, pad_value=1.0)
        with torch.inference_mode(mode=self.inference_mode):
            predictions = self.vggt(vggt_imgs)
            vggt_extrinsics_3x4, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], build_intrinsics=False)
            vggt_extrinsics = torch.eye(4, device=imgs.device).unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1)
            vggt_extrinsics[:, :, :3, :4] = vggt_extrinsics_3x4
            vggt_extrinsics_aligned, depth_values, s = align_pred_to_gt_batch(vggt_extrinsics, extrinsics, predictions["depth"].squeeze(-1))
            
            # visualize_cameras(vggt_extrinsics_aligned[:, :, :3, 3].reshape(-1, 3).cpu(), extrinsics[:, :, :3, 3].reshape(-1, 3).cpu())
            depths = adapt_size(imgs.shape[3:], depth_values.unsqueeze(2), pad_value=0.0).squeeze(2)
            nears, fars = compute_bounds(depths)
            
        return depths.clone(), nears.clone(), fars.clone(), s
    
    
          