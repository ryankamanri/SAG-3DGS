from typing import Any, List, Tuple

import torch
import torch.nn as nn
import taichi as ti
import numpy as np
from torch.autograd import Function

# 初始化 Taichi
ti.init(arch=ti.gpu, default_fp=ti.f32, debug=False)



class VoxelAttentionTaichiFunction(Function):
    """
    Voxel Attention Function Implemented with Taichi (Optimized Version) 
    Assumes that input points are sorted by voxel index, and the number of points per voxel is known.
    """
    @staticmethod
    def forward(ctx, projected_q, projected_k, projected_v,
                voxel_point_counts, voxel_start_indices, num_voxels):
        """
        Params:
            projected_q: [M, H]
            projected_k: [N, H]
            projected_v: [N, H]
            voxel_point_counts: [M]
            voxel_start_indices: [M]
            num_voxels: M
        """
        # save inputs for backward population
        ctx.save_for_backward(projected_q, projected_k, projected_v,
                              voxel_point_counts, voxel_start_indices)
        ctx.num_voxels = num_voxels

        # get shape info
        num_points, hidden_dim = projected_k.shape
        max_voxel_point_count = voxel_point_counts.max().item()
        
        # define the output and attn_weights tensor
        output = torch.zeros((num_voxels, hidden_dim), device=projected_q.device, dtype=projected_q.dtype)
        attn_weights = torch.zeros((num_points,), device=projected_q.device, dtype=projected_q.dtype)

        # compute attention in Taichi
        compute_attention_forward_optimized(
            projected_q.contiguous(), projected_k.contiguous(), projected_v.contiguous(),
            voxel_point_counts.contiguous(), voxel_start_indices.contiguous(),
            output.contiguous(), attn_weights.contiguous(),
            num_points, num_voxels, hidden_dim
        )

        # save for backward
        ctx.max_voxel_point_count = max_voxel_point_count
        ctx.attn_weights = attn_weights

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # get saved tensors
        projected_q, projected_k, projected_v, point_counts, start_indices = ctx.saved_tensors
        attn_weights = ctx.attn_weights
        num_voxels = ctx.num_voxels

        # get shape info
        num_points, hidden_dim = projected_k.shape
        
        # define the grad_q, grad_k, grad_v tensor
        grad_q = torch.zeros((num_voxels, hidden_dim), device=projected_q.device, dtype=projected_q.dtype)
        grad_k = torch.zeros((num_points, hidden_dim), device=projected_k.device, dtype=projected_k.dtype)
        grad_v = torch.zeros((num_points, hidden_dim), device=projected_v.device, dtype=projected_v.dtype)

        # compute attention grad backward in Taichi
        compute_attention_backward_optimized(
            grad_output.contiguous(), attn_weights.contiguous(),
            projected_q.contiguous(), projected_k.contiguous(), projected_v.contiguous(),
            point_counts.contiguous(), start_indices.contiguous(),
            grad_q.contiguous(), grad_k.contiguous(), grad_v.contiguous(),
            num_points, num_voxels, hidden_dim
        )

        return grad_q, grad_k, grad_v, None, None, None


@ti.kernel
def compute_attention_forward_optimized(
        features_q: ti.types.ndarray(), # readonly,  [num_voxels, hidden_dim]
        features_k: ti.types.ndarray(), # readonly,  [num_points, hidden_dim]
        features_v: ti.types.ndarray(), # readonly, [num_points, hidden_dim]
        point_counts: ti.types.ndarray(), # readonly, [num_voxels]
        start_indices: ti.types.ndarray(), # readonly, [num_voxels]
        output: ti.types.ndarray(), # write-only, [num_voxels, hidden_dim]
        attn_weights: ti.types.ndarray(), # write-only, [num_points]
        num_points: ti.i32,
        num_voxels: ti.i32,
        hidden_dim: ti.i32
):
    # parallel over each voxel
    for voxel_idx in range(num_voxels):
        max_score = -1e9
        point_count = point_counts[voxel_idx]
        start_idx = start_indices[voxel_idx]

        if point_count == 0:
            continue
        
        # compute max score for numerical stability
        # compute attention scores for all points in the voxel
        for local_j in range(point_count):
            global_j = start_idx + local_j
            # compute dot product between Q_voxel and K_j
            score = 0.0
            for h in range(hidden_dim):
                score += features_q[voxel_idx, h] * features_k[global_j, h]
            score /= ti.sqrt(hidden_dim)

            # save the score and update max_score
            attn_weights[global_j] = score
            if score > max_score:
                max_score = score

        # compute softmax
        exp_sum = 0.0
        for local_j in range(point_count):
            global_j = start_idx + local_j
            exp_val = ti.exp(attn_weights[global_j] - max_score)
            attn_weights[global_j] = exp_val
            exp_sum += exp_val

        # normalize
        for local_j in range(point_count):
            global_j = start_idx + local_j
            attn_weights[global_j] /= exp_sum

        # compute the weighted sum of values
        for h in range(hidden_dim):
            weighted_val = 0.0
            for local_j in range(point_count):
                global_j = start_idx + local_j
                weighted_val += attn_weights[global_j] * features_v[global_j, h]

            # save the weighted value to output
            output[voxel_idx, h] = weighted_val


@ti.kernel
def compute_attention_backward_optimized(
        grad_output: ti.types.ndarray(),  # readonly, [num_voxels, hidden_dim]
        attn_weights: ti.types.ndarray(),  # readonly, [num_points]
        features_q: ti.types.ndarray(),  # readonly, [num_voxels, hidden_dim]
        features_k: ti.types.ndarray(),  # readonly, [num_points, hidden_dim]
        features_v: ti.types.ndarray(),  # readonly, [num_points, hidden_dim]
        point_counts: ti.types.ndarray(),  # readonly, [num_voxels]
        start_indices: ti.types.ndarray(),  # readonly, [num_voxels]
        grad_features_q: ti.types.ndarray(),  # write-only, [num_voxels, hidden_dim]
        grad_features_k: ti.types.ndarray(),  # write-only, [num_points, hidden_dim]
        grad_features_v: ti.types.ndarray(),  # write-only, [num_points, hidden_dim]
        num_points: ti.i32,
        num_voxels: ti.i32,
        hidden_dim: ti.i32
):
    """Taichi-based attention backprop (per-voxel query vector version)."""
    # parallel over each voxel
    for voxel_idx in range(num_voxels):
        point_count = point_counts[voxel_idx]
        start_idx = start_indices[voxel_idx]

        if point_count == 0:
            continue

        # compute gradients w.r.t. V
        for local_j in range(point_count):
            global_j = start_idx + local_j
            for h in range(hidden_dim):
                # ∂L/∂V_j = ∂L/∂output * ∂output/∂V_j = grad_output * attn_weights
                grad_features_v[global_j, h] += grad_output[voxel_idx, h] * attn_weights[global_j]

        # compute gradients w.r.t. Q and K
        for local_j in range(point_count):
            global_j = start_idx + local_j
            # ∂L/∂attn_weights_j = ∂L/∂output * ∂output/∂attn_weights_j = grad_output · V_j
            grad_attn_j = 0.0
            for h in range(hidden_dim):
                grad_attn_j += grad_output[voxel_idx, h] * features_v[global_j, h]

            # compute gradients w.r.t. scores (softmax)
            for local_i in range(point_count):
                global_i = start_idx + local_i
                # ∂attn_weights_j/∂score_i = attn_weights_j * (δ_ij - attn_weights_i)
                grad_score_i = attn_weights[global_j] * (
                        (local_i == local_j) - attn_weights[global_i]
                ) * grad_attn_j

                # compute key's gradient
                for h in range(hidden_dim):
                    # ∂score_i/∂K_i = Q_voxel / sqrt(d)
                    grad_features_k[global_i, h] += grad_score_i * features_q[voxel_idx, h] / ti.sqrt(hidden_dim)

                # compute query's gradient
                for h in range(hidden_dim):
                    # ∂score_i/∂Q_voxel = K_i / sqrt(d)
                    grad_features_q[voxel_idx, h] += grad_score_i * features_k[global_i, h] / ti.sqrt(hidden_dim)

class VoxelAttentionTaichi(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels=None, stages=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or in_channels
        self.stages = stages

        self.query_proj = nn.Sequential(
            nn.Linear(1, hidden_channels // 4), 
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )  
        self.key_proj = nn.Linear(in_channels, hidden_channels)
        self.value_proj = nn.Linear(in_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, self.out_channels) if hidden_channels != self.out_channels else None

        self.position_encoding = nn.Linear(3, hidden_channels)

    def forward(self, points, features, voxel_centers, voxel_point_counts, voxel_start_indices, num_voxels, voxel_size, stage_idx):
        """
        Forward propagation (optimized version)

        Args:
            features: Point features [N, C], sorted by voxel index
            voxel_centers: Corresponding voxel center coordinates [N, C]
            voxel_point_counts: Number of points in each voxel [M]
            voxel_start_indices: Start index of each voxel in the sorted array [M]
            num_voxels: Total number of voxels M
        """
        voxel_size = torch.tensor([voxel_size], device=points.device)  # (1,)

        # Project query, key, and value
        projected_q: torch.Tensor = self.query_proj(voxel_size)  # (H, )
        projected_q = projected_q.unsqueeze(0).repeat(voxel_point_counts.shape[0], 1)  # (M, H)
        projected_k = self.key_proj(features)
        projected_v = self.value_proj(features)

        # Compute relative position encoding & add to key
        relative_position_encoding = self.position_encoding(points - voxel_centers)  # (N, C)
        projected_k = projected_k + relative_position_encoding

        # Compute attention using Taichi
        aggregated = VoxelAttentionTaichiFunction.apply(
            projected_q, projected_k, projected_v,
            voxel_point_counts, voxel_start_indices, num_voxels
        )

        # Apply output projection
        if self.out_proj is not None:
            aggregated = self.out_proj(aggregated)

        return aggregated



# Helper function: prepare sorted point cloud data
def prepare_sorted_pointcloud(points, point_features, voxel_size):
    """
    Prepare sorted point cloud data

    Input:
        points: (N, 3)
        point_features: (N, C)
        voxel_size: float

    Returns:
        sorted_points: Point positions sorted by voxel index (N, 3)
        sorted_features: Point features sorted by voxel index (N, C)
        sorted_voxel_centers: Voxel centers sorted by voxel index (N, C)
        voxel_point_counts: Number of points in each voxel (Vox,)
        voxel_start_indices: Start index of each voxel in the sorted array (Vox,)
        num_voxels: Total number of voxels
    """
    # Compute voxel coordinates
    voxel_coords = torch.floor(points / voxel_size).long()

    # Compute voxel centers
    voxel_centers = (voxel_coords + 0.5) * voxel_size

    # Convert 3D voxel coordinates to 1D hashed indices
    min_coords = voxel_coords.min(dim=0)[0]
    max_coords = voxel_coords.max(dim=0)[0]
    voxel_grid_dims = max_coords - min_coords + 1

    hashed_indices = (
        (voxel_coords[:, 0] - min_coords[0]) * voxel_grid_dims[1] * voxel_grid_dims[2] +
        (voxel_coords[:, 1] - min_coords[1]) * voxel_grid_dims[2] +
        (voxel_coords[:, 2] - min_coords[2])
    )

    # Get unique voxels and point counts
    unique_hashed, inverse_indices, counts = torch.unique(
        hashed_indices, return_inverse=True, return_counts=True
    )
    num_voxels = unique_hashed.shape[0]

    # Sort by voxel index
    sorted_indices = torch.argsort(inverse_indices)
    sorted_points = points[sorted_indices]
    sorted_features = point_features[sorted_indices]
    sorted_voxel_centers = voxel_centers[sorted_indices]

    # Compute start index of each voxel in the sorted array
    start_indices = torch.zeros(num_voxels, dtype=torch.int, device=points.device)
    start_indices[1:] = torch.cumsum(counts, dim=0)[:-1]

    return sorted_points, sorted_features, sorted_voxel_centers, counts.int(), start_indices, num_voxels



# Usage Example
if __name__ == "__main__":
    # Simulated data
    num_points = 1000
    in_channels = 32
    hidden_channels = 64
    voxel_size = 0.1

    # Randomly generate point cloud and features
    points = torch.rand(num_points, 3, device="cuda")  # Random point coordinates (N, 3)
    point_features = torch.rand(num_points, in_channels, device="cuda")  # Random point features (N, C)

    # Prepare sorted point cloud data
    sorted_points, sorted_features, sorted_voxel_centers, point_counts, start_indices, num_voxels = prepare_sorted_pointcloud(
        points, point_features, voxel_size
    )

    # Create attention module
    attention = VoxelAttentionTaichi(in_channels, hidden_channels).to("cuda")

    for i in range(3):
        # Forward pass
        voxel_features = attention.forward(
            sorted_points,
            sorted_features,
            sorted_voxel_centers,
            point_counts,
            start_indices,
            num_voxels,
            voxel_size, i)

        print(f"Number of input points: {num_points}")
        print(f"Number of output voxels: {num_voxels}")
        print(f"Shape of voxel features: {voxel_features.shape}")

        # Test gradient computation
        loss = voxel_features.mean()
        loss.backward()

        print("Gradient computed successfully!")
