import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusionBlock(nn.Module):
    """多尺度特征融合块，处理单个分辨率级别"""
    def __init__(self, feature_dim, fusion_mode='attention', scale_level=0):
        super().__init__()
        self.scale_level = scale_level
        self.feature_dim = feature_dim
        
        # 跨尺度连接处理
        if scale_level > 0:
            self.upsample = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.skip_conv = nn.Conv2d(feature_dim, feature_dim, 1)
        
        
        # 特征精炼模块
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim + 3, feature_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
        )

    def forward(self, images, target_features, warped_features, target_depths, prev_fused=None):
        """
        :param target_features: 目标视图原始特征 [B, V, C, H, W]
        :param warped_features: 重投影特征 [B, V, N, C, H, W] (N=源视图数)
        :param target_depths: 深度图 [B, V, H, W]
        :param prev_fused: 上一尺度融合特征 [B, V, C, H//2, W//2] (更低分辨率)
        :return: 融合特征 [B, V, C, H, W]
        """
        B, V, N, C, H, W = warped_features.shape
        
        # 1. 跨尺度特征融合 (从低分辨率到高分辨率)
        if self.scale_level > 0 and prev_fused is not None:
            # 上采样前一尺度特征 (更低分辨率 -> 当前分辨率)
            upsampled = self.upsample(prev_fused.view(B*V, C, H//2, W//2))
            upsampled = upsampled.view(B, V, C, H, W)
            
            # 跳跃连接处理
            skip = self.skip_conv(target_features.view(B*V, C, H, W))
            skip = skip.view(B, V, C, H, W)
            
            # 特征增强 (结合低分辨率上下文)
            enhanced_target = 0.7 * skip + 0.3 * upsampled
        else:
            enhanced_target = target_features
        
        # 2. 多视图特征融合 (当前尺度)
        # 点积融合 (保留多视图一致性)
        dot_weight = torch.softmax((warped_features * target_features.unsqueeze(2)).sum(dim=3) / (torch.tensor(C) ** 0.5), dim=2) # (B, V, N, H, W)
        fused = (warped_features * dot_weight.unsqueeze(dim=3)).sum(dim=2)
        
        # 3. 与增强后的目标特征融合
        combined = 0.3 * fused + 0.7 * enhanced_target
        
        # 4. 特征精炼
        refined = self.refine(torch.cat((
            combined.view(B*V, C, H, W), 
            images.view(B*V, 3, H, W),
        ), dim=1))
        return refined.view(B, V, C, H, W)


class DepthFuseNet(nn.Module):
    """修正的多尺度多视图特征融合网络（输入金字塔从低到高）"""
    def __init__(self, feature_dims=[64, 64, 64], fusion_mode='attention'):
        """
        :param feature_dims: 各尺度特征维度（索引0=最低分辨率）
        :param fusion_mode: 融合模式 ('weighted_sum', 'variance', 'attention')
        """
        super().__init__()
        self.num_scales = len(feature_dims)
        self.fusion_mode = fusion_mode
        
        # 深度/焦距特征编码器
        self.depth_intr_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 1)
        )
        
        # 创建多尺度融合模块（从低分辨率到高分辨率）
        self.fusion_blocks = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        for i, dim in enumerate(feature_dims):  # i=0: 最低分辨率
            self.fusion_blocks.append(
                MultiScaleFusionBlock(
                    feature_dim=dim,
                    fusion_mode=fusion_mode,
                    scale_level=i  # 0=最低分辨率
                )
            )
            self.embeddings.append(
                nn.Sequential(
                    nn.Conv2d(dim + 16, dim * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim * 2, dim * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim * 2, dim, 3, padding=1),
                )
            )
        
            

    def forward(self, images_pyramid, features_pyramid, depths_pyramid, pdf_max_pyramid, intrinsics, extrinsics):
        """
        :param features_pyramid: 多尺度特征金字塔 [尺度0, 尺度1, ...] 
                                其中尺度0=最低分辨率，尺度N=最高分辨率
        :param depths_pyramid: 多尺度深度图（同特征金字塔顺序）
        :param pdf_max_pyramid: 多尺度pdf_max图（同特征金字塔顺序）
        :param intrinsics: 相机内参 [B, V, 3, 3]（原始分辨率）
        :param extrinsics: 相机外参 [B, V, 4, 4] (World to Camera)
        :return: 多尺度融合特征金字塔 [B, V, C, H, W] * num_scales（同输入顺序）
        """
        # 0. 参数检查
        assert len(features_pyramid) == self.num_scales
        assert len(depths_pyramid) == self.num_scales
        
        # 1. 构建内参金字塔（按分辨率缩放）
        intrinsics_pyramid = self.create_intrinsics_pyramid(intrinsics)
        
        # 2. 从最低分辨率开始处理（尺度0），逐级向高分辨率处理
        fused_pyramid = []
        prev_fused = None  # 存储上一尺度的融合结果（用于跨尺度连接）
        
        for scale_idx in range(self.num_scales):  # 0=最低分辨率
            # 获取当前尺度数据
            features = features_pyramid[scale_idx]
            depths = depths_pyramid[scale_idx]
            intrinsics_scaled = intrinsics_pyramid[scale_idx]
            
            # 计算当前尺度的重投影特征
            warped_features = self.compute_warped_features(
                features, depths, intrinsics_scaled, extrinsics
            )
            
            
            # 应用当前尺度的融合块
            fused = self.fusion_blocks[scale_idx](
                images=images_pyramid[f"stage{scale_idx+1}"],
                target_features=features,
                warped_features=warped_features,
                target_depths=depths,
                prev_fused=prev_fused  # 传入上采样后的低分辨率特征
            )
            
            # 存储当前尺度结果
            fused_pyramid.append(fused)
            prev_fused = fused  # 为下一尺度准备
            
        for scale_idx in range(self.num_scales):
            # depth/intr embedding & color residual
            fused = fused_pyramid[scale_idx]
            depths = depths_pyramid[scale_idx]
            max_pdf = pdf_max_pyramid[scale_idx]
            intrinsics_inv_scaled = intrinsics_pyramid[scale_idx].inverse()
            b, v, hi, wi = depths.shape
            
            depth_intr = torch.cat((
                depths.view(b*v, 1, hi, wi) * (intrinsics_inv_scaled[:, :, 0, 0]).view(b*v, 1, 1, 1), 
                depths.view(b*v, 1, hi, wi) * (intrinsics_inv_scaled[:, :, 1, 1]).view(b*v, 1, 1, 1)
            ), dim=1)
            
            depth_intr_embedding = self.depth_intr_encoder(
                torch.cat((
                    torch.log(torch.clamp(depth_intr, min=1e-6)),
                    max_pdf.view(b*v, 1, hi, wi)
                ), dim=1)
            ).view(b, v, -1, hi, wi)
            
            fused_pyramid[scale_idx] = self.embeddings[scale_idx](
                torch.cat([
                    fused, 
                    depth_intr_embedding,
                ], dim=2).view(-1, fused.shape[2] + 16, fused.shape[3], fused.shape[4])
            ).view(*fused.shape)
            
            pass
        
        return fused_pyramid

    def compute_warped_features(self, features, depths, intrinsics, extrinsics):
        """计算当前尺度的重投影特征"""
        B, V, C, H, W = features.shape
        
        # 生成所有视图对
        all_views = torch.arange(V, device=features.device)
        target_indices = all_views.repeat_interleave(V - 1)
        source_indices = torch.cat([all_views[all_views != i] for i in all_views])
        
        # 向量化重投影
        warped_features = self.batch_reproject(
            features, depths, intrinsics, extrinsics,
            target_indices, source_indices
        )  # [B*V*(V-1), C, H, W]
        
        # 重塑为 [B, V, V-1, C, H, W]
        return warped_features.view(B, V, V-1, C, H, W)

    def create_intrinsics_pyramid(self, intrinsics):
        """创建多尺度内参金字塔（索引0=最低分辨率）"""
        intrinsics_pyramid = []
        
        # 从最高分辨率开始缩放（原始输入是最高分辨率）
        # 但我们的金字塔索引0是最低分辨率，所以需要反转
        scales = [1.0 / (2**i) for i in range(self.num_scales)]
        scales = scales[::-1]  # 反转：[最低分辨率缩放, ..., 最高分辨率缩放]
        
        for scale in scales:
            scaled_intrinsics = intrinsics.clone()
            scaled_intrinsics[:, :, :2, :] *= scale  # 缩放焦距和主点
            intrinsics_pyramid.append(scaled_intrinsics)
        
        return intrinsics_pyramid

    def batch_reproject(self, features, depths, intrinsics, extrinsics, 
                      target_indices, source_indices):
        """批量重投影特征（完整实现）"""
        B, V, C, H, W = features.shape
        num_pairs = len(target_indices)
        
        # 展平批次和视图维度
        flat_features = features.reshape(B*V, C, H, W)
        flat_depths = depths.reshape(B*V, H, W)
        flat_intrinsics = intrinsics.reshape(B*V, 3, 3)
        flat_extrinsics = extrinsics.reshape(B*V, 4, 4)
        
        # 选择目标视图参数
        target_depths = flat_depths.view(B, V, H, W)[:, target_indices].reshape(B*num_pairs, H, W)
        target_intrinsics = flat_intrinsics.view(B, V, 3, 3)[:, target_indices].reshape(B*num_pairs, 3, 3)
        target_extrinsics = flat_extrinsics.view(B, V, 4, 4)[:, target_indices].reshape(B*num_pairs, 4, 4)
        
        # 选择源视图参数
        source_features = flat_features.view(B, V, C, H, W)[:, source_indices].reshape(B*num_pairs, C, H, W)
        source_intrinsics = flat_intrinsics.view(B, V, 3, 3)[:, source_indices].reshape(B*num_pairs, 3, 3)
        source_extrinsics = flat_extrinsics.view(B, V, 4, 4)[:, source_indices].reshape(B*num_pairs, 4, 4)
        
        # 批量生成点云
        points = self.batch_depth_to_pointcloud(
            target_depths, target_intrinsics
        )  # [B*num_pairs, H, W, 3]
        
        # 批量投影到源视图
        grid = self.batch_project_points(
            points, 
            target_extrinsics, 
            source_extrinsics, 
            source_intrinsics
        )  # [B*num_pairs, H, W, 2]
        
        # 批量重采样特征
        warped_features = F.grid_sample(
            source_features, 
            grid, 
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_features

    def batch_depth_to_pointcloud(self, depths, intrinsics):
        """批量深度图转点云"""
        B, H, W = depths.shape
        device = depths.device
        
        # 创建像素网格
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        
        # 扩展为批处理
        grid_u = grid_u.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_v = grid_v.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        
        # 内参分解
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)
        
        # 归一化坐标
        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        
        # 转换为3D点
        points = torch.stack([
            x * depths,
            y * depths,
            depths
        ], dim=-1)  # [B, H, W, 3]
        
        return points

    def batch_project_points(self, points, target_extrinsics, source_extrinsics, source_intrinsics, min_clamp=1e-3):
        """批量3D点投影到源视图"""
        B, H, W, _ = points.shape
        device = points.device
        
        # 添加齐次坐标
        ones = torch.ones(B, H, W, 1, device=device)
        points_homo = torch.cat([points, ones], dim=-1)  # [B, H, W, 4]
        
        # 转换目标外参为世界坐标系 (求逆)
        target_inv = self.batch_inverse_extrinsic(target_extrinsics)
        
        # 变换到源视图坐标系
        # P_source = source_extrinsics @ target_inv @ P_target
        transform = torch.matmul(source_extrinsics, target_inv)
        
        # 应用变换 [B, 4, 4] @ [B, H, W, 4, 1] -> [B, H, W, 4]
        points_homo = points_homo.view(B, H*W, 4, 1)
        points_src = torch.matmul(transform.view(B, 1, 4, 4), points_homo).squeeze(-1)
        points_src = points_src.view(B, H, W, 4)
        
        torch.clamp_(points_src[..., 2], min=0.1) # avoid division by zero!
        
        # 投影到图像平面
        x = points_src[..., 0] / points_src[..., 2].clamp(min=min_clamp)
        y = points_src[..., 1] / points_src[..., 2].clamp(min=min_clamp)
        
        # 内参分解
        fx = source_intrinsics[:, 0, 0].view(B, 1, 1)
        fy = source_intrinsics[:, 1, 1].view(B, 1, 1)
        cx = source_intrinsics[:, 0, 2].view(B, 1, 1)
        cy = source_intrinsics[:, 1, 2].view(B, 1, 1)
        
        # 转换为像素坐标
        u = x * fx + cx
        v = y * fy + cy
        
        # 归一化到[-1, 1]
        grid_u = 2.0 * u / (W - 1) - 1.0
        grid_v = 2.0 * v / (H - 1) - 1.0
        
        return torch.stack([grid_u, grid_v], dim=-1)  # [B, H, W, 2]
    
    def batch_inverse_extrinsic(self, extrinsics):
        """批量计算外参矩阵的逆"""
        R = extrinsics[:, :3, :3]
        t = extrinsics[:, :3, 3]
        
        # 计算逆旋转
        R_inv = R.transpose(1, 2)
        
        # 计算逆平移
        t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
        
        # 构建逆矩阵
        inv = torch.zeros_like(extrinsics)
        inv[:, :3, :3] = R_inv
        inv[:, :3, 3] = t_inv
        inv[:, 3, 3] = 1.0
        
        return inv