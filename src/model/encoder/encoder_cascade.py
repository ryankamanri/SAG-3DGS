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
from ...global_cfg import get_cfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .mvsnet.cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult
from .backbone.feature_extractor import FeatureNet
from ..encodings.positional_encoding import camera_positional_encoding
from .backbone.multi_costvolume_transformer_module import MultiCostVolumeTransformerModule
from .backbone.voxelized_gaussian_adapter_module import VoxelizedGaussianAdapterModule
from .backbone.voxel_to_point_cross_attn_transformer import VoxelToPointTransformer
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .mvsnet import generate_depth_map_based_point_cloud, generate_geometric_mask


@dataclass
class EncoderCascadeCfg:
    name: Literal["cascade"]
    cas_mvsnet_ckpt_path: str
    cas_mvsnet_use_backbone: bool
    cas_mvsnet_load_to_backbone: bool
    cas_mvsnet_ndepth: list[int]
    cas_mvsnet_geo_max_dist: float
    cas_mvsnet_geo_max_depth_diff: float
    positional_encoding_num_frequencies: int
    feature_channels: int
    transformer_layers: int
    transformer_num_head: int
    no_ffn: bool
    ffn_dim_expansion: int
    max_voxels_foreach_processing: int
    voxel_size_list: list[int]
    patch_size_list: list[int]
    predict_sh_degree: int
    # params for multi-view depth predictor
    downscale_factor: int
    num_depth_candidates: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: list[int]
    costvolume_unet_attn_res: list
    gaussians_per_pixel: int
    depth_unet_feat_dim: int
    depth_unet_attn_res: list
    depth_unet_channel_mult: list[int]
    multiview_trans_attn_split: int
    
    


class EncoderCascade(Encoder[EncoderCascadeCfg]):
    cas_mvsnet_module: CasMVSNetModule
    multi_costvolume_transformer_module: MultiCostVolumeTransformerModule
    gaussian_adapter_module: VoxelizedGaussianAdapterModule
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.cas_mvsnet_module = CasMVSNetModule(
            cas_mvsnet_ckpt_path=cfg.cas_mvsnet_ckpt_path, 
            ndepths=cfg.cas_mvsnet_ndepth, 
            geo_max_dist=cfg.cas_mvsnet_geo_max_dist, 
            geo_max_depth_diff=cfg.cas_mvsnet_geo_max_depth_diff, 
            use_backbone=cfg.cas_mvsnet_use_backbone, 
            load_to_backbone=cfg.cas_mvsnet_load_to_backbone
            )
        
        self.feature_channels = cfg.feature_channels
        self.feature_extractor = FeatureNet()
        
        # from mvsplat
        self.backbone = BackboneMultiview(
            feature_channels=cfg.feature_channels,
            downscale_factor=cfg.downscale_factor
        )
        
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.feature_channels,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.feature_channels,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )
        
        
        self.transformer = VoxelToPointTransformer(
            num_layers=cfg.transformer_layers, 
            d_model=self.feature_channels, 
            nhead=cfg.transformer_num_head, 
            no_ffn=cfg.no_ffn, 
            ffn_dim_expansion=cfg.ffn_dim_expansion, 
            max_voxels_foreach_processing=cfg.max_voxels_foreach_processing
        )
        
        self.gaussian_adapter_module = VoxelizedGaussianAdapterModule(
            transformer=self.transformer, 
            feature_channels=self.feature_channels, 
            voxel_size_list=cfg.voxel_size_list, 
            patch_size_list=cfg.patch_size_list, 
            sh_degree=cfg.predict_sh_degree
        )
        
        print(cfg)
        print("Do NOT forget to register the lr for every module in `configure_optimizers`!")
        
    def preprocess(self, context):
        imgs : torch.Tensor = context["image"] # (B, V, C, H, W), or get the origin size image by context["origin_image"]
        alphas: torch.Tensor = context["alpha"] # (B, V, H, W)
        c2w_extrinsics : torch.Tensor = context["extrinsics"] # (B, V, 4, 4)
        normalized_intrinsics : torch.Tensor = context["intrinsics"] # (B, V, 3, 3), or get the origin size image by context["origin_intrinsics"]
        nears, fars = context["near"], context["far"] # (B, V)
        b, v, c, h, w = imgs.shape
        # crop image to adapt to mvsnet and swin transformer (h and w can be devided by 32)
        if h % 32 != 0:
            imgs = imgs[..., :-(h % 32), :]
            alphas = alphas[..., :-(h % 32), :]
        if w % 32 != 0:
            imgs = imgs[..., :-(w % 32)]
            alphas = alphas[..., :-(w % 32)]
        b, v, c, h, w = imgs.shape # update h and w
        
        # intrinsics adapt to img size
        intrinsics = normalized_intrinsics.clone()
        intrinsics[..., 0, :] *= w
        intrinsics[..., 1, :] *= h
        
        masks = alphas > 0.9
        return imgs, masks, c2w_extrinsics, intrinsics, nears, fars

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        ndepths = 192
    ) -> EncoderOutput:
        imgs, masks, extrinsics, intrinsics, nears, fars = self.preprocess(context)
        b, v, c, h, w = imgs.shape
        # features = self.feature_extractor(imgs) # (B, V, C, H, W)
        cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module(imgs, masks, extrinsics, intrinsics, nears, fars)
        ################################################### from mvsplat
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=None,
        )

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        ) # (B, V, H*W, 1, 1), (B, V, H*W, 1, 1), (B, V, H*W, C)
        
        
        features = rearrange(raw_gaussians, "b v (h w) c -> b v c h w", h=h, w=w)
        depths = rearrange(depths, "b v (h w) 1 1 -> b v h w", h=h, w=w)
        depths = list(torch.unbind(depths, dim=1)) # (B, H, W) * V
        vertices = generate_depth_map_based_point_cloud(depths, extrinsics, intrinsics) # (B, V, 4, H, W)
        near_fars = torch.stack([nears, fars], dim=-1) # (B, V, 2)
        geo_mask = []
        for vi in range(v):
            geo_mask.append(generate_geometric_mask(imgs, extrinsics, intrinsics, depths, near_fars, 
                                                    ref_idx=vi, max_depth_diff=self.cfg.cas_mvsnet_geo_max_depth_diff,
                                                    max_dist=self.cfg.cas_mvsnet_geo_max_dist)[0])
        
        cas_module_result.registed_prob_pcd.vertices = vertices
        cas_module_result.registed_prob_pcd.vertices_confidence = torch.ones(b, v, h, w, device=imgs.device) # (B, V, H, W)
        cas_module_result.registed_prob_pcd.vertices_geometry_mask = torch.stack(geo_mask, dim=1) if len(geo_mask) > 0 else torch.tensor(0) # (B, V, H, W)
        
        
        ##########################################################
        gaussians: EncoderOutput = self.gaussian_adapter_module(imgs, features, cas_module_result, masks, extrinsics, intrinsics, nears, fars)
        gaussians.others["cas_module_result"] = cas_module_result
        gaussians.others["nears"] = nears
        gaussians.others["fars"] = fars
        gaussians.others["depths"] = depths
        return gaussians

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
    
    def configure_optimizers(self, cfg):
        return [
            {'params': self.feature_extractor.parameters(), 'lr': cfg.lr}, 
            {'params': self.cas_mvsnet_module.parameters(), 'lr': cfg.lr}, 
            {'params': self.backbone.parameters(), 'lr': cfg.lr},
            {'params': self.depth_predictor.parameters(), 'lr': cfg.lr}, 
            {'params': self.transformer.parameters(), 'lr': cfg.lr}
        ] + self.gaussian_adapter_module.configure_optimizers(cfg)
