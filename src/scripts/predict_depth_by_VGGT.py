import os
from pathlib import Path
import torch
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT().to(device)
model.load_state_dict(torch.load("pretrained/vggt/model.pt"))
model.eval()

# META_DIR = Path("path_to_meta_dir")
# IMAGE_DIR = Path("path_to_image_dir")
# OUTPUT_DIR = Path("path_to_output_dir")
META_DIR = Path("C:/Users/97448/plus/repos/datasets/RealEstate10K/data/RealEstate10K/RealEstate10K")
IMAGE_DIR = Path("C:/Users/97448/plus/repos/mvsplat/outputs/predict_depth")
OUTPUT_DIR = Path("C:/Users/97448/plus/repos/mvsplat/outputs/predict_depth_output")


for stage in ["train", "test"]:
    os.makedirs(OUTPUT_DIR / stage, exist_ok=True)
    for scene in tqdm(os.listdir(IMAGE_DIR / stage), desc=f"stage {stage}"): # image based
        scene_img_dir = IMAGE_DIR / stage / scene
        w2cs = []
        image_dirs = []
        image_names = []
        # TODO: load extrs
        with open(META_DIR / stage / f"{scene}.txt") as f:
            content = f.readlines()
            for line in content[1:]:
                meta = line.rstrip().split(" ")
                image_names.append(meta[0])
                image_dirs.append(scene_img_dir / f"{meta[0]}.jpg")
                w2c = torch.tensor([
                    [float(i) for i in meta[7:11]], 
                    [float(i) for i in meta[11:15]], 
                    [float(i) for i in meta[15:19]], 
                    [0., 0., 0., 1.]], device=device)
                w2cs.append(w2c)
                pass
            pass
        # TODO: add codes below
        # Load and preprocess example images (replace with your own image paths)
        # image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
        image_size = (256, 256)
        images = load_and_preprocess_images(image_dirs).to(device)
        # print(images.shape)
        images = images.unsqueeze(0) # (B=1, V, C, H, W)
        b, v, c, h, w = images.shape
        extrinsics = torch.stack(w2cs).inverse().unsqueeze(0) # (B, V, 4, 4)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images)
                pass
            pass
        vggt_extrinsics_3x4, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], build_intrinsics=False)
        vggt_extrinsics = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1)
        vggt_extrinsics[:, :, :3, :4] = vggt_extrinsics_3x4
        vggt_extrinsics_aligned, depth_values = align_pred_to_gt_batch(vggt_extrinsics, extrinsics, predictions["depth"].squeeze(-1))
        aligned_depths = adapt_size(image_size, depth_values.unsqueeze(2), pad_value=0.0).squeeze(2) # (B, V, H, W)
        depth_confs = adapt_size(image_size, predictions["depth_conf"].unsqueeze(2), pad_value=0.0).squeeze(2)
        result = {}
        for vi in range(v):
            result[image_names[vi]] = {
                "depth": aligned_depths[0, vi], 
                "depth_conf": depth_confs[0, vi]
            }
            # print(result[image_names[vi]]["depth"].shape)
        # print(result)
        torch.save(result, str(OUTPUT_DIR / stage / f"{scene}.pt"))
        pass # for scene in os.listdir(IMAGE_DIR): # image based
    pass


