from dataclasses import dataclass
from typing import Callable, Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ..decoder.decoder import DecoderOutput

from ...misc.execution_timer import ExecutionTimer

from .mvsnet.vggt_module import VGGTModule

from .mvsnet.cas_mvsnet_module import CasMVSNetModule, CasMVSNetModuleResult

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import EncoderOutput
from .backbone import (
    BackboneMultiviewIncremental,
)
from .backbone.depth_fuse_net import DepthFuseNet
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeIncrementalCfg:
    name: Literal["incremental"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    unet_output_scales: list[int]
    cas_mvsnet_ckpt_path: str
    cas_mvsnet_use_backbone: bool
    cas_mvsnet_load_to_backbone: bool
    cas_mvsnet_ndepth: list[int]
    cas_mvsnet_cr_base_channels: list[int]
    cas_mvsnet_in_channels: list[int]
    cas_mvsnet_geo_max_dist: float
    cas_mvsnet_geo_max_depth_diff: float
    cas_mvsnet_use_out_features: bool


class EncoderCostVolumeIncremental(Encoder[EncoderCostVolumeIncrementalCfg]):
    backbone: BackboneMultiviewIncremental
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter
    render_callback: Optional[Callable[[EncoderOutput, int], DecoderOutput]] = None

    def __init__(self, cfg: EncoderCostVolumeIncrementalCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg()[get_cfg().mode].num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiviewIncremental(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                # is_strict_loading = not cfg.wo_backbone_cross_attn
                # self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        # self.depth_predictor = DepthPredictorMultiView(
        #     feature_channels=cfg.d_feature * 2,
        #     upscale_factor=cfg.downscale_factor,
        #     num_depth_candidates=cfg.num_depth_candidates,
        #     costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
        #     costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
        #     costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
        #     gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
        #     gaussians_per_pixel=cfg.gaussians_per_pixel,
        #     num_views=get_cfg()[get_cfg().mode].num_context_views,
        #     depth_unet_feat_dim=cfg.depth_unet_feat_dim,
        #     depth_unet_attn_res=cfg.depth_unet_attn_res,
        #     depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        #     wo_depth_refine=cfg.wo_depth_refine,
        #     wo_cost_volume=cfg.wo_cost_volume,
        #     wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        # )
        
        assert cfg.unet_output_scales[-1] == 1, "The last scale of unet_output_scales must be 1, which is the original size of the image."
        self.stages = [f"stage{i+1}" for i in range(len(cfg.unet_output_scales))]
        
        self.cas_mvsnet_module = CasMVSNetModule(
            feat_scales=cfg.unet_output_scales,
            cas_mvsnet_ckpt_path=cfg.cas_mvsnet_ckpt_path, 
            ndepths=cfg.cas_mvsnet_ndepth, 
            cr_base_chs=cfg.cas_mvsnet_cr_base_channels,
            in_channels=cfg.cas_mvsnet_in_channels,
            geo_max_dist=cfg.cas_mvsnet_geo_max_dist, 
            geo_max_depth_diff=cfg.cas_mvsnet_geo_max_depth_diff, 
            use_backbone=cfg.cas_mvsnet_use_backbone, 
            load_to_backbone=cfg.cas_mvsnet_load_to_backbone
            )
        
        self.depth_fuse_net = DepthFuseNet(
            feature_dims=self.backbone.get_feature_dims,
            fusion_mode="weighted_sum", 
        )
            
        self.use_vggt = True
        self.vggt_module = VGGTModule() if self.use_vggt else nn.Module()

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
    
    def preprocess(self, context):
        imgs : torch.Tensor = context["image"] # (B, V, C, H, W), or get the origin size image by context["origin_image"]
        b, v, c, h, w = imgs.shape

        alphas: torch.Tensor = context["alpha"] # (B, V, H, W)
        c2w_extrinsics : torch.Tensor = context["extrinsics"] # (B, V, 4, 4)
        normalized_intrinsics : torch.Tensor = context["intrinsics"] # (B, V, 3, 3), or get the origin size image by context["origin_intrinsics"]
        nears, fars = context["near"], context["far"] # (B, V)
        
        # check the shape of image to adapt to mvsnet and swin transformer (h and w can be devided by 32)
        assert h % 32 == 0 and w % 32 == 0, "The height and width of the image must be divisible by 32"
        
        stage_imgs = {}
        stage_masks = {}
        stage_intrinsics = {}
        
        # 对每个尺度进行处理
        for i, scale in enumerate(self.cfg.unet_output_scales, start=1):
            scale_key = f"stage{i}"
            
            # 图像缩放
            if scale == 1:
                stage_imgs[scale_key] = imgs
            else:
                scale_factor = 1.0 / scale
                stage_imgs[scale_key] = torch.nn.functional.interpolate(
                    imgs.view(b*v, c, h, w), 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=False
                ).view(b, v, c, h // scale, w // scale)
            
            # 内参调整
            stage_intrinsics[scale_key] = torch.stack([
                normalized_intrinsics[..., 0, :] * w / scale,
                normalized_intrinsics[..., 1, :] * h / scale,
                normalized_intrinsics[..., 2, :]  # 第三行保持不变
            ], dim=-2)
            
            # 掩码处理
            if scale == 1:
                stage_masks[scale_key] = alphas > 0.9
            else:
                stage_masks[scale_key] = torch.nn.functional.interpolate(
                    alphas.view(b*v, 1, h, w),
                    scale_factor=scale_factor,
                    mode='nearest'
                ).view(b, v, h // scale, w // scale) > 0.9

        return stage_imgs, stage_masks, c2w_extrinsics, stage_intrinsics, nears, fars


    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> EncoderOutput:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            epipolar_kwargs=epipolar_kwargs,
        )
        
        stage_features = {}
        for i in range(len(self.cfg.unet_output_scales)):
            _, _, c, hn, wn = trans_features[i].shape
            stage_features[f"stage{i+1}"] = trans_features[i].view(b, v, c, hn, wn) # (B, V, C, Hn, Wn)
        
        stage_imgs, stage_masks, extrinsics, stage_intrinsics, nears, fars = self.preprocess(context)
        imgs, intrinsics = stage_imgs[self.stages[-1]], stage_intrinsics[self.stages[-1]] # origin size images and intrinsics
        is_training = self.training

        cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module.forward(
            context, imgs, stage_masks, extrinsics, intrinsics, nears, fars, is_training, outer_features=stage_features if self.cfg.cas_mvsnet_use_out_features else None)

        
        if is_training and self.use_vggt:
            depths, _, _ = self.vggt_module.forward(imgs, extrinsics)
            context["depth"] = depths
            context["depth_mask"][torch.logical_or(depths < nears.view(b, v, 1, 1), depths > fars.view(b, v, 1, 1))] = 0.0 # remove those too big or small values.
            # Important note: `nears`, `fars` here covered those loaded from DataLoader, 
            # which will decide the candidate depths in the CasMVSNetModule and voxel range in VoxelizedGaussianAdapterModule.
            # and context["depth"], context["depth_mask"], which will be used in the loss function.
            # you can overwrite context["near"], context["far"] to ensure exact camera rendering (though it wonld not happen during training).
        else:
            context["depth"] = torch.ones((b, v, h, w), device=imgs.device) * fars.view(b, v, 1, 1)  # dummy depth map
            context["depth_mask"] = torch.ones((b, v, h, w), device=imgs.device)  # dummy depth mask
        
        # # Sample depths from the resulting features.
        # in_feats = trans_features[0]
        # extra_info = {}
        # extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        # extra_info["scene_names"] = scene_names
        # gpp = self.cfg.gaussians_per_pixel
        # depths, densities, raw_gaussians = self.depth_predictor(
        #     in_feats,
        #     context["intrinsics"],
        #     context["extrinsics"],
        #     context["near"],
        #     context["far"],
        #     gaussians_per_pixel=gpp,
        #     deterministic=deterministic,
        #     extra_info=extra_info,
        #     cnn_features=cnn_features[0],
        # )
        
        # depth (1, 3, 65536, 1, 1)
        # densities (1, 3, 65536, 1, 1)
        # raw_gaussians (1, 3, 65536, 84)
        
        stage_depths = [[cas_module_result.ref_view_result_list[vi].backbone[stage]["depth"] for vi in range(v)] for stage in self.stages] # [[(B, H, W) * V] * S]
        stage_depths = [torch.stack(depths, dim=1) for depths in stage_depths] # [(B, V, H, W) * S]
        
        # fuse with depth
        trans_features = self.depth_fuse_net.forward(trans_features, stage_depths, intrinsics, extrinsics.inverse())
        
        # multi-stage render
        stage_renders: dict = {}
        for idx, stage in enumerate(self.stages):
            # if we are not on training, we only reander the last stage.
            if not self.training and idx != len(self.stages) - 1: continue
            
            _, _, hi, wi = stage_depths[idx].shape
            depths = stage_depths[idx].view(b, v, hi*wi, 1, 1) # (B, V, H, W) -> (B, V, H*W, 1, 1)
        
            gaussian_channels = (self.gaussian_adapter.d_in + 2)
            raw_gaussians: torch.Tensor = trans_features[idx][:, :, :gaussian_channels, :, :]
            raw_gaussians = raw_gaussians.permute(0, 1, 3, 4, 2).view(b, v, hi*wi, gaussian_channels)
        
            densities: torch.Tensor = torch.sigmoid(trans_features[idx][:, :, gaussian_channels:gaussian_channels+1, :, :])
            densities = densities.permute(0, 1, 3, 4, 2).view(b, v, hi*wi, 1, 1) # (B, V, H*W, 1, 1)

            # Convert the features and depths into Gaussians.
            xy_ray, _ = sample_image_grid((hi, wi), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            gaussians = rearrange(
                raw_gaussians,
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / torch.tensor((wi, hi), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
            gpp = self.cfg.gaussians_per_pixel
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step) / gpp,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (hi, wi),
            )


            # Optionally apply a per-pixel opacity.
            opacity_multiplier = 1

            res = EncoderOutput(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.scales, 
                    "b v r srf spp xyz -> b (v r srf spp) xyz"
                ), 
                rearrange(
                    gaussians.rotations, 
                    "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                ), 
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    opacity_multiplier * gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
            
            if self.training:
                stage_renders[stage] = self.render_callback(res, self.cfg.unet_output_scales[idx])
                
        res.others["cas_module_result"] = cas_module_result
        res.others["nears"] = nears
        res.others["fars"] = fars
        res.others["stages"] = self.stages
        res.others["scales"] = self.cfg.unet_output_scales
        res.others["stage_renders"] = stage_renders
        
        return res

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
    
    def configure_optimizers(self, cfg):
        return [
            {'params': self.parameters(), 'lr': cfg.lr}
        ]
