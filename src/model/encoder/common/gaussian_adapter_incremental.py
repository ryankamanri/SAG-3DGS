from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class GaussianAdapterCfg:
    opacity_mapping: OpacityMappingCfg
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
            
    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        raw_gaussians: Float[Tensor, "*#batch _"],
        means: Float[Tensor, "*#batch 3"],   # 粗略值
        global_step: int, 
        eps: float = 1e-8,
    ) -> Gaussians:

        # 拆分 raw_gaussians：scales(3), rotations(4), mean_offset(3), densities, sh(3*d_sh)
        scales, rotations, mean_offset, densities, sh = raw_gaussians.split(
            (3, 4, 3, 1, 3 * self.d_sh), dim=-1
        )

        # Scale 特征映射到有效范围
        # scale_min = self.cfg.gaussian_scale_min
        # scale_max = self.cfg.gaussian_scale_max
        # scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        
        # refer AnySplat
        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)

        # 归一化四元数
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # Spherical harmonics
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh * self.sh_mask

        # 世界坐标系协方差
        covariances = build_covariance(scales, rotations)

        # ✅ 修正后的 means
        final_means = means + mean_offset * 0.001
        
        # compute the opacities
        densities: torch.Tensor = torch.sigmoid(densities)
        opacities = self.map_pdf_to_opacity(densities, global_step).view(-1)

        return Gaussians(
            means=final_means,
            covariances=covariances,
            harmonics=sh,
            opacities=opacities,
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 3 + 4 + 3 + 1 + 3 * self.d_sh
