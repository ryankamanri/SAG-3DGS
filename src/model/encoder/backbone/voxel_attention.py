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
    使用 Taichi 实现的体素注意力函数（优化版）
    假设输入点已按体素索引排序，并且知道每个体素中的点数
    """
    @staticmethod
    def forward(ctx, projected_q, projected_k, projected_v,
                voxel_point_counts, voxel_start_indices, num_voxels, ti_fields):
        """
        前向传播（优化版）

        参数:
            projected_q: 投影后的查询向量 [M, H]
            projected_k: 投影后的键向量 [N, H]，已按体素索引排序
            projected_v: 投影后的值向量 [N, H]，已按体素索引排序
            voxel_point_counts: 每个体素中的点数 [M]
            voxel_start_indices: 每个体素在排序后数组中的起始索引 [M]
            num_voxels: 体素数量 M
        """
        # 保存输入用于反向传播
        ctx.save_for_backward(projected_q, projected_k, projected_v,
                              voxel_point_counts, voxel_start_indices)
        ctx.num_voxels = num_voxels
        ctx.ti_fields = ti_fields

        # 获取形状信息
        num_points, hidden_dim = projected_k.shape
        max_voxel_point_count = voxel_point_counts.max().item()

        # 初始化 Taichi 字段
        ti_fields.features_q_ti = ti.field(ti.f32, shape=(num_voxels, hidden_dim))  # (M, H)
        ti_fields.features_k_ti = ti.field(ti.f32, shape=(num_points, hidden_dim))  # (N, H)
        ti_fields.features_v_ti = ti.field(ti.f32, shape=(num_points, hidden_dim))  # (N, H)
        ti_fields.point_counts_ti = ti.field(ti.i32, shape=num_voxels)
        ti_fields.start_indices_ti = ti.field(ti.i32, shape=num_voxels)
        ti_fields.output_ti = ti.field(ti.f32, shape=(num_voxels, hidden_dim))
        ti_fields.attn_weights_ti = ti.field(ti.f32, shape=(num_points,))
        ti_fields.grad_output_ti = ti.field(ti.f32, shape=(num_voxels, hidden_dim))
        ti_fields.grad_q_ti = ti.field(ti.f32, shape=(num_voxels, hidden_dim))
        ti_fields.grad_k_ti = ti.field(ti.f32, shape=(num_points, hidden_dim))
        ti_fields.grad_v_ti = ti.field(ti.f32, shape=(num_points, hidden_dim))

        # 将数据复制到 Taichi
        ti_fields.features_q_ti.from_torch(projected_q.contiguous())
        ti_fields.features_k_ti.from_torch(projected_k.contiguous())
        ti_fields.features_v_ti.from_torch(projected_v.contiguous())
        ti_fields.point_counts_ti.from_torch(voxel_point_counts.contiguous())
        ti_fields.start_indices_ti.from_torch(voxel_start_indices.contiguous())

        # 在 Taichi 中计算注意力
        compute_attention_forward_optimized(
            ti_fields.features_q_ti, ti_fields.features_k_ti, ti_fields.features_v_ti,
            ti_fields.point_counts_ti, ti_fields.start_indices_ti,
            ti_fields.output_ti, ti_fields.attn_weights_ti,
            num_points, num_voxels, hidden_dim
        )

        # 将结果复制回 PyTorch
        result = ti_fields.output_ti.to_torch(device=projected_q.device)

        # 保存注意力权重用于反向传播
        ctx.max_voxel_point_count = max_voxel_point_count
        ctx.attn_weights = ti_fields.attn_weights_ti.to_torch(device=projected_q.device)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播（优化版）
        """
        # 获取保存的输入和中间结果
        projected_q, projected_k, projected_v, point_counts, start_indices = ctx.saved_tensors
        attn_weights = ctx.attn_weights
        num_voxels = ctx.num_voxels
        ti_fields = ctx.ti_fields

        # 获取形状信息
        num_points, hidden_dim = projected_k.shape

        # 将数据复制到 Taichi
        ti_fields.features_q_ti.from_torch(projected_q.contiguous())
        ti_fields.features_k_ti.from_torch(projected_k.contiguous())
        ti_fields.features_v_ti.from_torch(projected_v.contiguous())
        ti_fields.grad_output_ti.from_torch(grad_output.contiguous())
        ti_fields.attn_weights_ti.from_torch(attn_weights.contiguous())
        ti_fields.point_counts_ti.from_torch(point_counts.contiguous())
        ti_fields.start_indices_ti.from_torch(start_indices.contiguous())

        # 在 Taichi 中计算梯度
        compute_attention_backward_optimized(
            ti_fields.grad_output_ti, ti_fields.attn_weights_ti,
            ti_fields.features_q_ti, ti_fields.features_k_ti, ti_fields.features_v_ti,
            ti_fields.point_counts_ti, ti_fields.start_indices_ti,
            ti_fields.grad_q_ti, ti_fields.grad_k_ti, ti_fields.grad_v_ti,
            num_points, num_voxels, hidden_dim
        )

        # 将梯度复制回 PyTorch
        grad_q = ti_fields.grad_q_ti.to_torch(device=projected_q.device)
        grad_k = ti_fields.grad_k_ti.to_torch(device=projected_k.device)
        grad_v = ti_fields.grad_v_ti.to_torch(device=projected_v.device)

        return grad_q, grad_k, grad_v, None, None, None, None


@ti.kernel
def compute_attention_forward_optimized(
        features_q: ti.template(),
        features_k: ti.template(),
        features_v: ti.template(),
        point_counts: ti.template(),
        start_indices: ti.template(),
        output: ti.template(),
        attn_weights: ti.template(),
        num_points: ti.i32,
        num_voxels: ti.i32,
        hidden_dim: ti.i32
):
    """在 Taichi 中计算注意力前向传播（优化版）"""
    max_score = -1e9
    # 对每个体素并行处理
    for voxel_idx in range(num_voxels):
        point_count = point_counts[voxel_idx]
        start_idx = start_indices[voxel_idx]

        if point_count == 0:
            continue

        # 计算注意力分数
        # 计算所有注意力分数并找到最大值
        for local_j in range(point_count):
            global_j = start_idx + local_j
            # 计算点i和点j之间的注意力分数
            score = 0.0
            for h in range(hidden_dim):
                score += features_q[voxel_idx, h] * features_k[global_j, h]
            score /= ti.sqrt(hidden_dim)

            # 保存分数并找到最大值
            attn_weights[global_j] = score
            if score > max_score:
                max_score = score

        # 计算softmax
        exp_sum = 0.0
        for local_j in range(point_count):
            global_j = start_idx + local_j
            exp_val = ti.exp(attn_weights[global_j] - max_score)
            attn_weights[global_j] = exp_val
            exp_sum += exp_val

        # 归一化
        for local_j in range(point_count):
            global_j = start_idx + local_j
            attn_weights[global_j] /= exp_sum

        # 计算加权和
        for h in range(hidden_dim):
            weighted_val = 0.0
            for local_j in range(point_count):
                global_j = start_idx + local_j
                weighted_val += attn_weights[global_j] * features_v[global_j, h]

            # 存储加权值（平均）
            output[voxel_idx, h] = weighted_val


@ti.kernel
def compute_attention_backward_optimized(
        grad_output: ti.template(),  # 形状: [num_voxels, hidden_dim]
        attn_weights: ti.template(),  # 形状: [num_points]
        features_q: ti.template(),  # 形状: [num_voxels, hidden_dim]
        features_k: ti.template(),  # 形状: [num_points, hidden_dim]
        features_v: ti.template(),  # 形状: [num_points, hidden_dim]
        point_counts: ti.template(),  # 形状: [num_voxels]
        start_indices: ti.template(),  # 形状: [num_voxels]
        grad_features_q: ti.template(),  # 形状: [num_voxels, hidden_dim]
        grad_features_k: ti.template(),  # 形状: [num_points, hidden_dim]
        grad_features_v: ti.template(),  # 形状: [num_points, hidden_dim]
        num_points: ti.i32,
        num_voxels: ti.i32,
        hidden_dim: ti.i32
):
    """在 Taichi 中计算注意力反向传播（每个体素独立查询向量版本）"""
    # 对每个体素并行处理
    for voxel_idx in range(num_voxels):
        point_count = point_counts[voxel_idx]
        start_idx = start_indices[voxel_idx]

        if point_count == 0:
            continue

        # 计算值的梯度
        for local_j in range(point_count):
            global_j = start_idx + local_j
            for h in range(hidden_dim):
                # ∂L/∂V_j = ∂L/∂output * ∂output/∂V_j = grad_output * attn_weights
                grad_features_v[global_j, h] += grad_output[voxel_idx, h] * attn_weights[global_j]

        # 计算注意力权重的梯度
        for local_j in range(point_count):
            global_j = start_idx + local_j
            # 计算 ∂L/∂attn_weights_j = ∂L/∂output * ∂output/∂attn_weights_j = grad_output · V_j
            grad_attn_j = 0.0
            for h in range(hidden_dim):
                grad_attn_j += grad_output[voxel_idx, h] * features_v[global_j, h]

            # 计算softmax的梯度
            for local_i in range(point_count):
                global_i = start_idx + local_i
                # ∂attn_weights_j/∂score_i = attn_weights_j * (δ_ij - attn_weights_i)
                grad_score_i = attn_weights[global_j] * (
                        (local_i == local_j) - attn_weights[global_i]
                ) * grad_attn_j

                # 计算键的梯度
                for h in range(hidden_dim):
                    # ∂score_i/∂K_i = Q_voxel / sqrt(d)
                    grad_features_k[global_i, h] += grad_score_i * features_q[voxel_idx, h] / ti.sqrt(hidden_dim)

                # 计算查询的梯度（每个体素独立，无需原子操作）
                for h in range(hidden_dim):
                    # ∂score_i/∂Q_voxel = K_i / sqrt(d)
                    grad_features_q[voxel_idx, h] += grad_score_i * features_k[global_i, h] / ti.sqrt(hidden_dim)
# 包装成模块
class VoxelAttentionTaichi(nn.Module):
    """
    使用 Taichi 的体素注意力模块（优化版）
    """

    def __init__(self, in_channels, hidden_channels, out_channels=None, stages=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or in_channels
        self.stages = stages

        # 定义投影层
        self.query_proj = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),  # 输入是体素大小（标量）
            nn.ReLU(),
            nn.Linear(hidden_channels // 4, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )  # 体素大小到查询向量的投影网络
        self.key_proj = nn.Linear(in_channels, hidden_channels)
        self.value_proj = nn.Linear(in_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, self.out_channels) if hidden_channels != self.out_channels else None

        self.position_encoding = nn.Linear(3, hidden_channels)

        # define taichi fields
        class TaichiFields:
            features_q_ti = ti.field(ti.f32, shape=())  # (M, H)
            features_k_ti = ti.field(ti.f32, shape=())  # (N, H)
            features_v_ti = ti.field(ti.f32, shape=())  # (N, H)
            point_counts_ti = ti.field(ti.i32, shape=())
            start_indices_ti = ti.field(ti.i32, shape=())

            output_ti = ti.field(ti.f32, shape=())
            attn_weights_ti = ti.field(ti.f32, shape=())

            grad_output_ti = ti.field(ti.f32, shape=())

            grad_q_ti = ti.field(ti.f32, shape=())
            grad_k_ti = ti.field(ti.f32, shape=())
            grad_v_ti = ti.field(ti.f32, shape=())

        self.stage_ti_fields = [TaichiFields() for _ in range(self.stages)]

    def forward(self, points, features, voxel_centers, voxel_point_counts, voxel_start_indices, num_voxels, voxel_size, stage_idx):
        """
        前向传播（优化版）

        参数:
            features: 点特征 [N, C]，已按体素索引排序
            voxel_centers: 对应体素中心点 [N, C]
            voxel_point_counts: 每个体素中的点数 [M]
            voxel_start_indices: 每个体素在排序后数组中的起始索引 [M]
            num_voxels: 体素数量 M
        """
        voxel_size = torch.tensor([voxel_size], device=points.device)  # (1,)

        # 投影查询、键和值
        projected_q: torch.Tensor = self.query_proj(voxel_size)  # (H, )
        projected_q = projected_q.unsqueeze(0).repeat(voxel_point_counts.shape[0], 1) # (M, H)
        projected_k = self.key_proj(features)
        projected_v = self.value_proj(features)

        # compute relative position encoding & add to K
        relative_position_encoding = self.position_encoding(points - voxel_centers)  # (N, C)
        projected_k = projected_k + relative_position_encoding

        # 使用 Taichi 计算注意力
        aggregated = VoxelAttentionTaichiFunction.apply(
            projected_q, projected_k, projected_v,
            voxel_point_counts, voxel_start_indices, num_voxels, self.stage_ti_fields[stage_idx]
        )

        # 应用输出投影
        if self.out_proj is not None:
            aggregated = self.out_proj(aggregated)

        return aggregated


# 辅助函数：准备排序后的点云数据
def prepare_sorted_pointcloud(points, point_features, voxel_size):
    """
    准备排序后的点云数据

    input:
        points: (N, 3)
        point: features: (N, C)
        voxel_size: float

    返回:
        sorted_points: 按体素索引排序的点位置 (N, 3)
        sorted_features: 按体素索引排序的点特征 (N, C)
        sorted_voxel_centers: 按体素索引排序的体素中心 (N, C)
        voxel_point_counts: 每个体素中的点数 (Vox,)
        voxel_start_indices: 每个体素在排序后数组中的起始索引 (Vox,)
        num_voxels: 体素数量
    """
    # 计算体素坐标
    voxel_coords = torch.floor(points / voxel_size).long()

    # compute voxel_center
    voxel_centers = (voxel_coords + 0.5) * voxel_size

    # 将三维坐标转换为一维哈希
    min_coords = voxel_coords.min(dim=0)[0]
    max_coords = voxel_coords.max(dim=0)[0]
    voxel_grid_dims = max_coords - min_coords + 1

    hashed_indices = (
            (voxel_coords[:, 0] - min_coords[0]) * voxel_grid_dims[1] * voxel_grid_dims[2] +
            (voxel_coords[:, 1] - min_coords[1]) * voxel_grid_dims[2] +
            (voxel_coords[:, 2] - min_coords[2])
    )

    # 获取唯一体素和计数
    unique_hashed, inverse_indices, counts = torch.unique(
        hashed_indices, return_inverse=True, return_counts=True
    )
    num_voxels = unique_hashed.shape[0]

    # 按体素索引排序
    sorted_indices = torch.argsort(inverse_indices)
    sorted_points = points[sorted_indices]
    sorted_features = point_features[sorted_indices]
    sorted_voxel_centers = voxel_centers[sorted_indices]

    # 计算每个体素的起始索引
    start_indices = torch.zeros(num_voxels, dtype=torch.int, device=points.device)
    start_indices[1:] = torch.cumsum(counts, dim=0)[:-1]

    return sorted_points, sorted_features, sorted_voxel_centers, counts.int(), start_indices, num_voxels


# 使用示例
if __name__ == "__main__":
    # 模拟数据
    num_points = 1000
    in_channels = 32
    hidden_channels = 64
    voxel_size = 0.1

    # 随机生成点云和特征
    points = torch.rand(num_points, 3, device="cuda")  # 随机点坐标 (N, 3)
    point_features = torch.rand(num_points, in_channels, device="cuda")  # 随机点特征 (N, C)

    # 准备排序后的点云数据
    sorted_points, sorted_features, sorted_voxel_centers, point_counts, start_indices, num_voxels = prepare_sorted_pointcloud(
        points, point_features, voxel_size
    )

    # 创建注意力模块
    attention = VoxelAttentionTaichi(in_channels, hidden_channels).to("cuda")

    for i in range(2):
        # 前向传播
        voxel_features = attention.forward(
            sorted_points,
            sorted_features,
            sorted_voxel_centers,
            point_counts,
            start_indices,
            num_voxels,
            voxel_size)


        print(f"输入点数量: {num_points}")
        print(f"输出体素数量: {num_voxels}")
        print(f"体素特征形状: {voxel_features.shape}")

        # 测试梯度
        loss = voxel_features.mean()
        loss.backward()

        print("梯度计算成功!")