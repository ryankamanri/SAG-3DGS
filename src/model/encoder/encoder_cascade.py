from dataclasses import dataclass
from typing import Callable, Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ..decoder.decoder import DecoderOutput

from .mvsnet.vggt_module import VGGTModule
from .backbone.costvolume_sampler import CostvolumeSampler

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
from ..encodings.positional_encoding import camera_positional_encoding
from .backbone.multi_costvolume_transformer_module import MultiCostVolumeTransformerModule
from .backbone.voxelized_gaussian_adapter_module import VoxelizedGaussianAdapterModule
from .backbone.voxel_to_point_cross_attn_transformer import VoxelToPointTransformer
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .mvsnet import generate_depth_map_based_point_cloud, generate_geometric_mask
from .costvolume.ldm_unet.unet import UNetModel
from ...misc.execution_timer import ExecutionTimer
from .backbone.feature_extractor import FeatureUNet

@dataclass
class EncoderCascadeCfg:
    name: Literal["cascade"]
    unet_output_scales: list[int]
    cas_mvsnet_ckpt_path: str
    cas_mvsnet_use_backbone: bool
    cas_mvsnet_load_to_backbone: bool
    cas_mvsnet_ndepth: list[int]
    cas_mvsnet_cr_base_channels: list[int]
    cas_mvsnet_base_channel: int
    cas_mvsnet_geo_max_dist: float
    cas_mvsnet_geo_max_depth_diff: float
    cas_mvsnet_use_out_features: bool
    use_vggt: bool
    positional_encoding_num_frequencies: int
    feature_channels: int
    transformer_layers: int
    transformer_num_head: int
    no_ffn: bool
    ffn_dim_expansion: int
    max_voxels_foreach_processing: int
    voxel_size_list: list[int]
    voxel_size_begin_steps: list[int]
    patch_size_list: list[int]
    predict_sh_degree: int
    min_thresholds: list[float]
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
    render_callback: Optional[Callable[[EncoderOutput, int], DecoderOutput]] = None  # render low resolution for split voxel.
    
    def __init__(self, cfg: EncoderCascadeCfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.voxel_size_list = cfg.voxel_size_list
        self.voxel_size_begin_steps = cfg.voxel_size_begin_steps
        self.current_idx = 0
        assert cfg.unet_output_scales[-1] == 1, "The last scale of unet_output_scales must be 1, which is the original size of the image."
        self.stages = [f"stage{i+1}" for i in range(len(cfg.unet_output_scales))]
        
        self.use_vggt = self.cfg.use_vggt and get_cfg().mode == "train"
        self.vggt_module = VGGTModule() if self.use_vggt else nn.Module()
        
        self.cas_mvsnet_module = CasMVSNetModule(
            feat_scales=cfg.unet_output_scales,
            cas_mvsnet_ckpt_path=cfg.cas_mvsnet_ckpt_path, 
            ndepths=cfg.cas_mvsnet_ndepth, 
            cr_base_chs=cfg.cas_mvsnet_cr_base_channels,
            base_channel=cfg.cas_mvsnet_base_channel,
            geo_max_dist=cfg.cas_mvsnet_geo_max_dist, 
            geo_max_depth_diff=cfg.cas_mvsnet_geo_max_depth_diff, 
            use_backbone=cfg.cas_mvsnet_use_backbone, 
            load_to_backbone=cfg.cas_mvsnet_load_to_backbone
            )
        
        self.feature_channels = cfg.feature_channels
        self.do_enhance_feat = True
        self.timer_switch = False
        
        self.feat_extractor = FeatureUNet(
            output_scales=cfg.unet_output_scales,
            out_channels=cfg.feature_channels,
        )
        
        # from mvsplat
        self.backbone = BackboneMultiview(
            feature_channels=cfg.feature_channels * 4,
            downscale_factor=4, 
            num_head=cfg.transformer_num_head * 4
        )
        
        self.upsamplerx2 = nn.Sequential(
            nn.ConvTranspose2d(cfg.feature_channels * 4, cfg.feature_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(cfg.feature_channels * 2)
        )
        
        self.upsamplerx4 = nn.Sequential(     
            nn.ConvTranspose2d(cfg.feature_channels * 2, cfg.feature_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(cfg.feature_channels)
        )
        
        self.feat_enhancer = nn.ModuleDict({
            "stage_unifying": nn.ModuleDict({
                "stage1": nn.Sequential(
                    nn.Conv2d(cfg.feature_channels * 4, cfg.feature_channels * 4, kernel_size=1, stride=1, padding=0),
                    nn.GELU(), 
                    nn.Conv2d(cfg.feature_channels * 4, cfg.feature_channels, kernel_size=1, stride=1, padding=0),
                ), 
                "stage2": nn.Sequential(
                    nn.Conv2d(cfg.feature_channels * 2, cfg.feature_channels * 2, kernel_size=1, stride=1, padding=0),
                    nn.GELU(), 
                    nn.Conv2d(cfg.feature_channels * 2, cfg.feature_channels, kernel_size=1, stride=1, padding=0),
                ),
                "stage3": nn.Sequential(
                    nn.Conv2d(cfg.feature_channels, cfg.feature_channels, kernel_size=1, stride=1, padding=0),
                    nn.GELU(), 
                    nn.Conv2d(cfg.feature_channels, cfg.feature_channels, kernel_size=1, stride=1, padding=0),
                ), 
            }), 
            "conv1": nn.Sequential(
                nn.Conv2d(self.feature_channels+3, self.feature_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=1, stride=1, padding=0),
            ), # merge rgb features
            "conv2": nn.Sequential(
                nn.Conv2d(self.feature_channels+3, self.feature_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=1, stride=1, padding=0),
            ), # merge rgb features
            "unet": UNetModel(
                image_size=None, 
                in_channels=self.feature_channels,
                model_channels=self.feature_channels,
                out_channels=self.feature_channels,
                num_res_blocks=1,
                attention_resolutions=cfg.depth_unet_attn_res,
                channel_mult=cfg.depth_unet_channel_mult,
                num_head_channels=self.feature_channels // 2, 
                dims=2,
                postnorm=True, 
                num_frames=get_cfg()[get_cfg().mode].num_context_views, 
                use_cross_view_self_attn=True
            ),
        }) 
        
        
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.feature_channels,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.feature_channels,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg()[get_cfg().mode].num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
        )
        
        
        self.transformer = nn.ModuleList([VoxelToPointTransformer(
            num_layers=cfg.transformer_layers, 
            d_model=self.feature_channels, 
            nhead=cfg.transformer_num_head, 
            no_ffn=cfg.no_ffn, 
            ffn_dim_expansion=cfg.ffn_dim_expansion, 
            max_voxels_foreach_processing=cfg.max_voxels_foreach_processing
        ) for vi in self.voxel_size_list])
        
        self.costvolume_sampler = CostvolumeSampler(
            max_voxels_foreach_processing=cfg.max_voxels_foreach_processing,
            costvolume_feature_channels=(cfg.feature_channels if cfg.cas_mvsnet_use_out_features else cfg.cas_mvsnet_base_channel) * 4 * 2,  # 8 * 4 * 2
            out_channels=cfg.feature_channels,
        )
        
        self.gaussian_adapter_module = VoxelizedGaussianAdapterModule(
            scales=cfg.unet_output_scales,
            stages=self.stages,
            transformer=self.transformer, 
            costvolume_sampler=self.costvolume_sampler,
            feature_channels=self.feature_channels, 
            voxel_size_list=cfg.voxel_size_list, 
            patch_size_list=cfg.patch_size_list, 
            sh_degree=cfg.predict_sh_degree, 
            max_voxels_foreach_processing=cfg.max_voxels_foreach_processing,
            min_thresholds=cfg.min_thresholds,
        )
        
        print(cfg)
        print("Do NOT forget to register the lr for every module in `configure_optimizers`!")
        
    def enhance_features(
        self, 
        stage_imgs: dict[str, torch.Tensor], # (B, V, 3, H, W)
        stage_features: dict[str, torch.Tensor], # (B, V, C, H, W)
        ):
        stage_enhanced_features = {}
        for stage in ("stage1", "stage2", "stage3"):
            imgs = stage_imgs[stage] # (B, V, 3, Hn, Wn)
            features = stage_features[stage] # (B, V, Cn, Hn, Wn)
            b, v, c, h, w = features.shape
            
            features = self.feat_enhancer["stage_unifying"][stage](features.view(b*v, c, h, w)) # (B*V, C, Hn, Wn)
            unified_c = features.shape[1] # update c after unifying
            
            enhanced_features = self.feat_enhancer["conv1"](
                torch.cat((features, imgs.reshape(b*v, 3, h, w)), dim=1)
            )
            enhanced_features = self.feat_enhancer["unet"](enhanced_features)
            enhanced_features = self.feat_enhancer["conv2"](
                torch.cat((enhanced_features, imgs.reshape(b*v, 3, h, w)), dim=1)
            )
            stage_enhanced_features[stage] = enhanced_features.view(b, v, unified_c, h, w) # (B, V, C, Hn, Wn)
        
        return stage_enhanced_features
        
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
        ndepths = 192
    ) -> EncoderOutput:
        is_training = self.training
        stage_imgs, stage_masks, extrinsics, stage_intrinsics, nears, fars = self.preprocess(context)
        imgs, intrinsics = stage_imgs[self.stages[-1]], stage_intrinsics[self.stages[-1]] # origin size images and intrinsics
        b, v, c, h, w = imgs.shape
        while global_step >= self.voxel_size_begin_steps[self.current_idx]:
            self.current_idx += 1

        with ExecutionTimer("Feature Extraction", switch=self.timer_switch):
            features_list = self.feat_extractor.forward(imgs.view(b*v, c, h, w)) # (B*V, C, Hn, Wn)
            stage_features = {}
            for i in range(len(self.cfg.unet_output_scales)):
                _, _, hn, wn = features_list[i].shape
                stage_features[f"stage{i+1}"] = features_list[i].view(b, v, self.feature_channels, hn, wn) # (B, V, C, Hn, Wn)
        
        if is_training and self.use_vggt:
            with ExecutionTimer("VGGT Module", switch=self.timer_switch):
                depths, nears, fars = self.vggt_module.forward(imgs, extrinsics)
                context["depth"] = depths
                context["depth_mask"][torch.logical_or(depths < nears.view(b, v, 1, 1), depths > fars.view(b, v, 1, 1))] = 0.0 # remove those too big or small values.
                # Important note: `nears`, `fars` here covered those loaded from DataLoader, 
                # which will decide the candidate depths in the CasMVSNetModule and voxel range in VoxelizedGaussianAdapterModule.
                # and context["depth"], context["depth_mask"], which will be used in the loss function.
                # you can overwrite context["near"], context["far"] to ensure exact camera rendering (though it wonld not happen during training).
            pass
        else:
            context["depth"] = torch.ones((b, v, h, w), device=imgs.device) * fars.view(b, v, 1, 1)  # dummy depth map
            context["depth_mask"] = torch.ones((b, v, h, w), device=imgs.device)  # dummy depth mask
        
        with ExecutionTimer("CAS-MVSNet Module", switch=self.timer_switch):
            cas_module_result: CasMVSNetModuleResult = self.cas_mvsnet_module.forward(
                context, imgs, stage_masks, extrinsics, intrinsics, nears, fars, is_training, outer_features=stage_features if self.cfg.cas_mvsnet_use_out_features else None)
        
        
        ##########################################################
        with ExecutionTimer("Gaussian Adapter Module", switch=self.timer_switch):
            gaussians: EncoderOutput = self.gaussian_adapter_module.forward(
                stage_imgs, 
                stage_features, 
                self.current_idx if is_training else len(self.voxel_size_list), 
                cas_module_result, 
                stage_masks, 
                extrinsics, 
                stage_intrinsics, 
                nears, fars, is_training, 
                self.render_callback)
        gaussians.others["cas_module_result"] = cas_module_result
        gaussians.others["nears"] = nears
        gaussians.others["fars"] = fars
        gaussians.others["stages"] = self.stages
        gaussians.others["scales"] = self.cfg.unet_output_scales
        return gaussians

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
    
    def configure_optimizers(self, cfg):
        return [
            # {'params': self.vggt_module.parameters(), 'lr': cfg.lr}, # we don't train VGGT, so no need to set lr
            {'params': self.feat_extractor.parameters(), 'lr': cfg.lr},
            {'params': self.cas_mvsnet_module.parameters(), 'lr': cfg.lr}, 
            {'params': self.backbone.parameters(), 'lr': cfg.lr}, 
            {'params': self.upsamplerx2.parameters(), 'lr': cfg.lr}, 
            {'params': self.upsamplerx4.parameters(), 'lr': cfg.lr}, 
            {'params': self.feat_enhancer.parameters(), 'lr': cfg.lr}, 
            {'params': self.depth_predictor.parameters(), 'lr': cfg.lr}, 
            {'params': self.transformer.parameters(), 'lr': cfg.lr}, 
            {'params': self.costvolume_sampler.parameters(), 'lr': cfg.lr}, 
        ] + self.gaussian_adapter_module.configure_optimizers(cfg)
