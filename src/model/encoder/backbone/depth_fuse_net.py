import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusionBlock(nn.Module):
    """Multi-scale feature fusion block for processing a single resolution level"""
    def __init__(self, feature_dim, fusion_mode='attention', scale_level=0):
        super().__init__()
        self.scale_level = scale_level
        self.feature_dim = feature_dim
        
        # Cross-scale connection processing
        if scale_level > 0:
            self.upsample = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            self.skip_conv = nn.Conv2d(feature_dim, feature_dim, 1)
        
        
        # Feature refinement module
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim + 3, feature_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
        )

    def forward(self, images, target_features, warped_features, target_depths, prev_fused=None):
        """
        :param target_features: Target view original features [B, V, C, H, W]
        :param warped_features: Reprojected features [B, V, N, C, H, W] (N=number of source views)
        :param target_depths: Depth maps [B, V, H, W]
        :param prev_fused: Previous scale fused features [B, V, C, H//2, W//2] (lower resolution)
        :return: Fused features [B, V, C, H, W]
        """
        B, V, N, C, H, W = warped_features.shape
        
        # 1. Cross-scale feature fusion (from low resolution to high resolution)
        if self.scale_level > 0 and prev_fused is not None:
            # Upsample previous scale features (lower resolution -> current resolution)
            upsampled = self.upsample(prev_fused.view(B*V, C, H//2, W//2))
            upsampled = upsampled.view(B, V, C, H, W)
            
            # Skip connection processing
            skip = self.skip_conv(target_features.view(B*V, C, H, W))
            skip = skip.view(B, V, C, H, W)
            
            # Feature enhancement (combining low-resolution context)
            enhanced_target = 0.7 * skip + 0.3 * upsampled
        else:
            enhanced_target = target_features
        
        # 2. Multi-view feature fusion (current scale)
        # Dot product fusion (preserving multi-view consistency)
        dot_weight = torch.softmax((warped_features * target_features.unsqueeze(2)).sum(dim=3) / (torch.tensor(C) ** 0.5), dim=2) # (B, V, N, H, W)
        fused = (warped_features * dot_weight.unsqueeze(dim=3)).sum(dim=2)
        
        # 3. Fusion with enhanced target features
        combined = 0.3 * fused + 0.7 * enhanced_target
        
        # 4. Feature refinement
        refined = self.refine(torch.cat((
            combined.view(B*V, C, H, W), 
            images.view(B*V, 3, H, W),
        ), dim=1))
        return refined.view(B, V, C, H, W)


class DepthFuseNet(nn.Module):
    """Revised multi-scale multi-view feature fusion network (input pyramid from low to high resolution)"""
    def __init__(self, feature_dims=[64, 64, 64], fusion_mode='attention'):
        """
        :param feature_dims: Feature dimensions for each scale (index 0 = lowest resolution)
        :param fusion_mode: Fusion mode ('weighted_sum', 'variance', 'attention')
        """
        super().__init__()
        self.num_scales = len(feature_dims)
        self.fusion_mode = fusion_mode
        
        # Depth/focal feature encoder
        self.depth_intr_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 1)
        )
        
        # Create multi-scale fusion modules (from low to high resolution)
        self.fusion_blocks = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        for i, dim in enumerate(feature_dims):  # i=0: lowest resolution
            self.fusion_blocks.append(
                MultiScaleFusionBlock(
                    feature_dim=dim,
                    fusion_mode=fusion_mode,
                    scale_level=i  # 0=lowest resolution
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
        :param features_pyramid: Multi-scale feature pyramid [scale0, scale1, ...] 
                                where scale0=lowest resolution, scaleN=highest resolution
        :param depths_pyramid: Multi-scale depth maps (same order as feature pyramid)
        :param pdf_max_pyramid: Multi-scale pdf_max maps (same order as feature pyramid)
        :param intrinsics: Camera intrinsics [B, V, 3, 3] (original resolution)
        :param extrinsics: Camera extrinsics [B, V, 4, 4] (World to Camera)
        :return: Multi-scale fused feature pyramid [B, V, C, H, W] * num_scales (same order as input)
        """
        # 0. Parameter validation
        assert len(features_pyramid) == self.num_scales
        assert len(depths_pyramid) == self.num_scales
        
        # 1. Build intrinsics pyramid (scaled by resolution)
        intrinsics_pyramid = self.create_intrinsics_pyramid(intrinsics)
        
        # 2. Process from lowest resolution (scale 0), progressively to higher resolutions
        fused_pyramid = []
        prev_fused = None  # Store previous scale fusion result (for cross-scale connections)
        
        for scale_idx in range(self.num_scales):  # 0=lowest resolution
            # Get current scale data
            features = features_pyramid[scale_idx]
            depths = depths_pyramid[scale_idx]
            intrinsics_scaled = intrinsics_pyramid[scale_idx]
            
            # Compute reprojected features at current scale
            warped_features = self.compute_warped_features(
                features, depths, intrinsics_scaled, extrinsics
            )
            
            
            # Apply current scale fusion block
            fused = self.fusion_blocks[scale_idx](
                images=images_pyramid[f"stage{scale_idx+1}"],
                target_features=features,
                warped_features=warped_features,
                target_depths=depths,
                prev_fused=prev_fused  # Pass upsampled low-resolution features
            )
            
            # Store current scale result
            fused_pyramid.append(fused)
            prev_fused = fused  # Prepare for next scale
            
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
        """Compute reprojected features for current scale"""
        B, V, C, H, W = features.shape
        
        # Generate all view pairs
        all_views = torch.arange(V, device=features.device)
        target_indices = all_views.repeat_interleave(V - 1)
        source_indices = torch.cat([all_views[all_views != i] for i in all_views])
        
        # Vectorized reprojection
        warped_features = self.batch_reproject(
            features, depths, intrinsics, extrinsics,
            target_indices, source_indices
        )  # [B*V*(V-1), C, H, W]
        
        # Reshape to [B, V, V-1, C, H, W]
        return warped_features.view(B, V, V-1, C, H, W)

    def create_intrinsics_pyramid(self, intrinsics):
        """Create multi-scale intrinsics pyramid (index 0 = lowest resolution)"""
        intrinsics_pyramid = []
        
        # Start scaling from highest resolution (original input is highest resolution)
        # But our pyramid index 0 is lowest resolution, so we need to reverse
        scales = [1.0 / (2**i) for i in range(self.num_scales)]
        scales = scales[::-1]  # Reverse: [lowest resolution scale, ..., highest resolution scale]
        
        for scale in scales:
            scaled_intrinsics = intrinsics.clone()
            scaled_intrinsics[:, :, :2, :] *= scale  # Scale focal length and principal point
            intrinsics_pyramid.append(scaled_intrinsics)
        
        return intrinsics_pyramid

    def batch_reproject(self, features, depths, intrinsics, extrinsics, 
                      target_indices, source_indices):
        """Batch reprojection of features (full implementation)"""
        B, V, C, H, W = features.shape
        num_pairs = len(target_indices)
        
        # Flatten batch and view dimensions
        flat_features = features.reshape(B*V, C, H, W)
        flat_depths = depths.reshape(B*V, H, W)
        flat_intrinsics = intrinsics.reshape(B*V, 3, 3)
        flat_extrinsics = extrinsics.reshape(B*V, 4, 4)
        
        # Select target view parameters
        target_depths = flat_depths.view(B, V, H, W)[:, target_indices].reshape(B*num_pairs, H, W)
        target_intrinsics = flat_intrinsics.view(B, V, 3, 3)[:, target_indices].reshape(B*num_pairs, 3, 3)
        target_extrinsics = flat_extrinsics.view(B, V, 4, 4)[:, target_indices].reshape(B*num_pairs, 4, 4)
        
        # Select source view parameters
        source_features = flat_features.view(B, V, C, H, W)[:, source_indices].reshape(B*num_pairs, C, H, W)
        source_intrinsics = flat_intrinsics.view(B, V, 3, 3)[:, source_indices].reshape(B*num_pairs, 3, 3)
        source_extrinsics = flat_extrinsics.view(B, V, 4, 4)[:, source_indices].reshape(B*num_pairs, 4, 4)
        
        # Batch point cloud generation
        points = self.batch_depth_to_pointcloud(
            target_depths, target_intrinsics
        )  # [B*num_pairs, H, W, 3]
        
        # Batch projection to source views
        grid = self.batch_project_points(
            points, 
            target_extrinsics, 
            source_extrinsics, 
            source_intrinsics
        )  # [B*num_pairs, H, W, 2]
        
        # Batch feature resampling
        warped_features = F.grid_sample(
            source_features, 
            grid, 
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped_features

    def batch_depth_to_pointcloud(self, depths, intrinsics):
        """Batch depth map to point cloud conversion"""
        B, H, W = depths.shape
        device = depths.device
        
        # Create pixel grid
        u = torch.arange(W, device=device).float()
        v = torch.arange(H, device=device).float()
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        
        # Expand for batch processing
        grid_u = grid_u.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        grid_v = grid_v.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        
        # Intrinsics decomposition
        fx = intrinsics[:, 0, 0].view(B, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1)
        
        # Normalized coordinates
        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        
        # Convert to 3D points
        points = torch.stack([
            x * depths,
            y * depths,
            depths
        ], dim=-1)  # [B, H, W, 3]
        
        return points

    def batch_project_points(self, points, target_extrinsics, source_extrinsics, source_intrinsics, min_clamp=1e-3):
        """Batch 3D point projection to source views"""
        B, H, W, _ = points.shape
        device = points.device
        
        # Add homogeneous coordinates
        ones = torch.ones(B, H, W, 1, device=device)
        points_homo = torch.cat([points, ones], dim=-1)  # [B, H, W, 4]
        
        # Convert target extrinsics to world coordinate system (inverse)
        target_inv = self.batch_inverse_extrinsic(target_extrinsics)
        
        # Transform to source view coordinate system
        # P_source = source_extrinsics @ target_inv @ P_target
        transform = torch.matmul(source_extrinsics, target_inv)
        
        # Apply transformation [B, 4, 4] @ [B, H, W, 4, 1] -> [B, H, W, 4]
        points_homo = points_homo.view(B, H*W, 4, 1)
        points_src = torch.matmul(transform.view(B, 1, 4, 4), points_homo).squeeze(-1)
        points_src = points_src.view(B, H, W, 4)
        
        torch.clamp_(points_src[..., 2], min=0.1) # avoid division by zero!
        
        # Project to image plane
        x = points_src[..., 0] / points_src[..., 2].clamp(min=min_clamp)
        y = points_src[..., 1] / points_src[..., 2].clamp(min=min_clamp)
        
        # Intrinsics decomposition
        fx = source_intrinsics[:, 0, 0].view(B, 1, 1)
        fy = source_intrinsics[:, 1, 1].view(B, 1, 1)
        cx = source_intrinsics[:, 0, 2].view(B, 1, 1)
        cy = source_intrinsics[:, 1, 2].view(B, 1, 1)
        
        # Convert to pixel coordinates
        u = x * fx + cx
        v = y * fy + cy
        
        # Normalize to [-1, 1]
        grid_u = 2.0 * u / (W - 1) - 1.0
        grid_v = 2.0 * v / (H - 1) - 1.0
        
        return torch.stack([grid_u, grid_v], dim=-1)  # [B, H, W, 2]
    
    def batch_inverse_extrinsic(self, extrinsics):
        """Batch computation of extrinsic matrix inverse"""
        R = extrinsics[:, :3, :3]
        t = extrinsics[:, :3, 3]
        
        # Compute inverse rotation
        R_inv = R.transpose(1, 2)
        
        # Compute inverse translation
        t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)
        
        # Construct inverse matrix
        inv = torch.zeros_like(extrinsics)
        inv[:, :3, :3] = R_inv
        inv[:, :3, 3] = t_inv
        inv[:, 3, 3] = 1.0
        
        return inv