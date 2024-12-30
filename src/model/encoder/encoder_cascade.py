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
from ..types import EncoderOutput
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .mvsnet.cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult
from ..encodings.positional_encoding import camera_positional_encoding
from .backbone.multi_costvolume_transformer_module import MultiCostVolumeTransformerModule
from .backbone.hash_table_voxelized_gaussian_adapter_module import HashTableVoxelizedGaussianAdapterModule, GAUSSIAN_FEATURE_CHANNELS




@dataclass
class EncoderCascadeCfg:
    name: Literal["cascade"]
    cas_mvsnet_ckpt_path: str
    cas_mvsnet_use_backbone: bool
    cas_mvsnet_load_to_backbone: bool
    cas_mvsnet_ndepth: list[int]
    positional_encoding_num_frequencies: int
    multi_costvolume_transformer_layers: int
    multi_costvolume_transformer_num_head: int
    voxel_size_list: list[int]
    min_opacity_list: list[float]


class EncoderCascade(Encoder[EncoderCascadeCfg]):
    cas_mvsnet_module: CasMVSNetModule
    multi_costvolume_transformer_module: MultiCostVolumeTransformerModule
    gaussian_adapter_module: HashTableVoxelizedGaussianAdapterModule
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cas_mvsnet_module = CasMVSNetModule(
            cas_mvsnet_ckpt_path=cfg.cas_mvsnet_ckpt_path, 
            ndepths=cfg.cas_mvsnet_ndepth, 
            use_backbone=cfg.cas_mvsnet_use_backbone, 
            load_to_backbone=cfg.cas_mvsnet_load_to_backbone
            )
        
        self.positional_encoding_num_frequencies = cfg.positional_encoding_num_frequencies
        self.rgb_channels = 3
        
        self.multi_costvolume_transformer_module = MultiCostVolumeTransformerModule(
            num_transformer_layers=cfg.multi_costvolume_transformer_layers, 
            input_channels=(self.rgb_channels + cfg.cas_mvsnet_ndepth[-1] + 32 * self.positional_encoding_num_frequencies),
            out_channels=GAUSSIAN_FEATURE_CHANNELS, # 3(means)+7(quaternion+scale)+3*1(shs)+1(opacity)+1(current)
            num_head=cfg.multi_costvolume_transformer_num_head
        )
        
        self.multiview_trans_attn_split = 16
        
        self.gaussian_adapter_module = HashTableVoxelizedGaussianAdapterModule(
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
        
        cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module(imgs, extrinsics, intrinsics, nears, fars)
        cam_poses = camera_positional_encoding(extrinsics, intrinsics, num_frequencies=self.positional_encoding_num_frequencies)
        hash_tables = self.multi_costvolume_transformer_module(cas_module_result, cam_poses, self.multiview_trans_attn_split)
        gaussians: EncoderOutput = self.gaussian_adapter_module(hash_tables, cas_module_result, extrinsics, fars)
        
        gaussians.others["cas_module_result"] = cas_module_result
        return gaussians

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
