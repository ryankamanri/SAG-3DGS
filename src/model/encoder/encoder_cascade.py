from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult


# @dataclass
# class OpacityMappingCfg:
#     initial: float
#     final: float
#     warm_up: int


@dataclass
class EncoderCascadeCfg:
    name: Literal["cascade"]
    cas_mvsnet_ckpt_path: str
    # d_feature: int
    # num_depth_candidates: int
    # num_surfaces: int
    # visualizer: EncoderVisualizerCostVolumeCfg
    # gaussian_adapter: GaussianAdapterCfg
    # opacity_mapping: OpacityMappingCfg
    # gaussians_per_pixel: int
    # unimatch_weights_path: str | None
    # downscale_factor: int
    # shim_patch_size: int
    # multiview_trans_attn_split: int
    # costvolume_unet_feat_dim: int
    # costvolume_unet_channel_mult: List[int]
    # costvolume_unet_attn_res: List[int]
    # depth_unet_feat_dim: int
    # depth_unet_attn_res: List[int]
    # depth_unet_channel_mult: List[int]
    # wo_depth_refine: bool
    # wo_cost_volume: bool
    # wo_backbone_cross_attn: bool
    # wo_cost_volume_refine: bool
    # use_epipolar_trans: bool


class EncoderCascade(Encoder[EncoderCascadeCfg]):
    cas_mvsnet_module: CasMVSNetModule
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cas_mvsnet_module = CasMVSNetModule(cfg.cas_mvsnet_ckpt_path)

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        ndepths = 192
    ) -> Gaussians:
        cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module(context)
        
        print(True)
        
        return Gaussians(torch.zeros(0, 0, 0), torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 3, 0), torch.zeros(0, 0))

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
