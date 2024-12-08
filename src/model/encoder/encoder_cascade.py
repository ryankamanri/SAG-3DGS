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
from .mvsnet.cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult
from ..encodings.positional_encoding import camera_positional_encoding
from .backbone.multi_costvolume_transformer_module import MultiCostVolumeTransformerModule
from .backbone.hash_table_voxelized_gaussian_adapter_module import HashTableVoxelizedGaussianAdapterModule


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
    multi_costvolume_transformer_module: MultiCostVolumeTransformerModule
    gaussian_adapter_module: HashTableVoxelizedGaussianAdapterModule
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cas_mvsnet_module = CasMVSNetModule(cfg.cas_mvsnet_ckpt_path)
        self.multi_costvolume_transformer_module = MultiCostVolumeTransformerModule(
            num_transformer_layers=2 * 2, # need to be 2n
            input_channels=75,
            out_channels=14, # 3(means)+7(quaternion+scale)+3*1(shs)+1(opacity)
            num_head=1, # TODO: fix abnormal code of multi-head attention.
            ffn_dim_expansion=4,
            no_cross_attn=False
        )
        
        self.multiview_trans_attn_split = 16
        
        self.gaussian_adapter_module = HashTableVoxelizedGaussianAdapterModule([32, 128, 512])

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
        cam_poses = camera_positional_encoding(context["extrinsics"], context["intrinsics"], num_frequencies=2)
        hash_tables = self.multi_costvolume_transformer_module(cas_module_result, cam_poses, self.multiview_trans_attn_split)
        gaussians = self.gaussian_adapter_module(hash_tables, context["extrinsics"], context["far"])
        
        return gaussians

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
