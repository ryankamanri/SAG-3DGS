import torch
from einops import rearrange
from torch import nn

from .unimatch.backbone import UNetEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneMultiviewIncremental(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackboneMultiviewIncremental, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        
        self.num_output_scale = 3
        self.feature_dims = [self.feature_channels, self.feature_channels * 3 // 2, self.feature_channels * 2, self.feature_channels * 3]  # 32, 48, 64, 96
        self.feature_dims_inverse = self.feature_dims[::-1]  # [96, 64, 48, 32]
        # NOTE: '0' here hack to get 1/4 features
        self.unet_encoder = UNetEncoder(
            num_output_scales=self.num_output_scale + 1, # +1 for the original image scale
            feature_dims=self.feature_dims,
        )

        self.transformers = nn.ModuleList([MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=self.feature_dims[i],
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        ) for i in range(self.num_output_scale - 1, -1, -1)])
        
        # I hope every stage features have the same channel(the maximum).
        self.upsampler_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.feature_dims[-1], self.feature_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
            ) for i in range(self.num_output_scale - 1, -1, -1)])
        
        self.unet_decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.feature_dims[-1] + self.feature_dims[i], self.feature_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dims[-1], self.feature_dims[-1], kernel_size=3, stride=1, padding=1),
            ) for i in range(self.num_output_scale - 1, -1, -1)])
        
    @property
    def get_feature_dims(self):
        return [self.feature_dims[-1] for _ in range(self.num_output_scale)] # from course to fine

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.unet_encoder(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def forward(
        self,
        images,
        attn_splits=2,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of view features, and every view has its multiscale features
        
        stage_feature_list = [[x[i] for x in features_list] for i in range(self.num_output_scale + 1)]  # [[[B, C, H, W] * V] for each scale]
        
        out_lists = []
        last_stage_feat =  torch.stack(stage_feature_list[0], dim=1) # [B, V, C, H, W]
        
        for i in range(self.num_output_scale):

            cur_stage_feat = stage_feature_list[i + 1]
        
            # add position to features
            cur_stage_feat = feature_add_position_list(
                cur_stage_feat, attn_splits, self.feature_dims_inverse[i + 1])

            # Transformer, now only for the deepest feature map.
            if i == 0:
                cur_stage_feat = self.transformers[i](
                    cur_stage_feat, attn_num_splits=attn_splits * 2 ** i)

            v = len(cur_stage_feat)
            b, c, h, w = cur_stage_feat[0].shape
            trans_features = torch.stack(cur_stage_feat, dim=1).view(b*v, c, h, w)  # [B*V, C, H, W]
            
            _, _, c_, h_, w_ = last_stage_feat.shape
            last_stage_feat = last_stage_feat.view(b*v, c_, h_, w_)  # [B*V, C', H', W']

            # Upsample
            upsampled_feat = self.upsampler_layers[i](last_stage_feat)  # [B*V, C, H*2, W*2]
            fused_feat = self.unet_decoder_layers[i](
                torch.cat((upsampled_feat, trans_features), dim=1)
            )
            
            # Update the current stage feature
            last_stage_feat = fused_feat.view(b, v, self.feature_dims[-1], h, w)
            out_lists.append(last_stage_feat)  # [B, V, C, H, W]

        return out_lists, [torch.stack(stage_feature_list[i+1], dim=1) for i in range(self.num_output_scale)]  # [B, V, C, H, W] for each scale
