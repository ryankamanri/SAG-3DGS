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
from ..types import EncoderOutput
from .backbone import (
    BackboneMultiview,
)
from ..types import EncoderOutput, empty_encoder_output
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .mvsnet.cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult
from .backbone.feature_extractor import CNNFeatureExtractor
from ..encodings.positional_encoding import camera_positional_encoding
from .backbone.multi_costvolume_transformer_module import MultiCostVolumeTransformerModule
from .backbone.voxelized_gaussian_adapter_module import VoxelizedGaussianAdapterModule, GAUSSIAN_FEATURE_CHANNELS
from .backbone.voxel_to_point_cross_attn_transformer import VoxelToPointTransformer



@dataclass
class EncoderCascadeCfg:
    name: Literal["cascade"]
    cas_mvsnet_ckpt_path: str
    cas_mvsnet_use_backbone: bool
    cas_mvsnet_load_to_backbone: bool
    cas_mvsnet_ndepth: list[int]
    positional_encoding_num_frequencies: int
    feature_channels: int
    transformer_layers: int
    transformer_num_head: int
    no_ffn: bool
    ffn_dim_expansion: int
    voxel_size_list: list[int]
    min_opacity_list: list[float]


class EncoderCascade(Encoder[EncoderCascadeCfg]):
    cas_mvsnet_module: CasMVSNetModule
    multi_costvolume_transformer_module: MultiCostVolumeTransformerModule
    gaussian_adapter_module: VoxelizedGaussianAdapterModule
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cas_mvsnet_module = CasMVSNetModule(
            cas_mvsnet_ckpt_path=cfg.cas_mvsnet_ckpt_path, 
            ndepths=cfg.cas_mvsnet_ndepth, 
            use_backbone=cfg.cas_mvsnet_use_backbone, 
            load_to_backbone=cfg.cas_mvsnet_load_to_backbone
            )
        
        self.feature_channels = cfg.feature_channels
        self.feature_extractor = CNNFeatureExtractor(out_channels=self.feature_channels)
        
        self.transformer = VoxelToPointTransformer(
            num_layers=cfg.transformer_layers, 
            d_model=self.feature_channels, 
            nhead=cfg.transformer_num_head, 
            no_ffn=cfg.no_ffn, 
            ffn_dim_expansion=cfg.ffn_dim_expansion
        )
        
        self.gaussian_adapter_module = VoxelizedGaussianAdapterModule(
            transformer=self.transformer, 
            feature_channels=self.feature_channels, 
            voxel_size_list=cfg.voxel_size_list, 
            min_opacity=cfg.min_opacity_list
        )
        
        print(cfg)
        
    def preprocess(self, context):
        imgs : torch.Tensor = context["image"] # (B, V, C, H, W), or get the origin size image by context["origin_image"]
        c2w_extrinsics : torch.Tensor = context["extrinsics"] # (B, V, 4, 4)
        normalized_intrinsics : torch.Tensor = context["intrinsics"] # (B, V, 3, 3), or get the origin size image by context["origin_intrinsics"]
        nears, fars = context["near"], context["far"] # (B, V)
        b, v, c, h, w = imgs.shape
        # crop image to adapt to mvsnet and swin transformer (h and w can be devided by 16)
        if h % 16 != 0:
            imgs = imgs[..., :-(h % 16), :]
        if w % 16 != 0:
            imgs = imgs[..., :-(w % 16)]
        b, v, c, h, w = imgs.shape # update h and w
        
        # convert extrinsics c2w -> w2c
        extrinsics = c2w_extrinsics.inverse()
        
        # intrinsics adapt to img size
        intrinsics = normalized_intrinsics.clone()
        intrinsics[..., 0, :] *= w
        intrinsics[..., 1, :] *= h
        
        return imgs, extrinsics, intrinsics, nears, fars

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        ndepths = 192
    ) -> EncoderOutput:
        imgs, extrinsics, intrinsics, nears, fars = self.preprocess(context)
        features = self.feature_extractor(imgs) # (B, V, C, H, W)
        cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module(imgs, extrinsics, intrinsics, nears, fars)
        gaussians: EncoderOutput = self.gaussian_adapter_module(features, cas_module_result, extrinsics, intrinsics, fars)
        gaussians.others["cas_module_result"] = cas_module_result
        return gaussians

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
