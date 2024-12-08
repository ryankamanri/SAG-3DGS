from dataclasses import dataclass
import torch
from torch import nn
from ..mvsnet import CascadeMVSNet, generate_point_cloud_from_depth_maps
from .multiview_transformer import MultiViewFeatureTransformer
from ..mvsnet.cas_mvsnet_module import CasMVSNetModuleResult
from ..mvsnet.cas_module import Deconv2d



class MultiCostVolumeTransformerModule(nn.Module):
    transformer: MultiViewFeatureTransformer

    def __init__(
        self, 
        num_transformer_layers, 
        input_channels, 
        out_channels, 
        num_head, 
        ffn_dim_expansion, 
        no_cross_attn) -> None:
        super().__init__()
        
        self.feature_channels = out_channels * 4
        self.out_channels = out_channels
        
        self.feature_projection = nn.Linear(input_channels, self.feature_channels)
        
        
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=self.feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )
        
        self.deconv1 = nn.ConvTranspose2d(self.out_channels * 4, self.out_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.out_channels * 2, self.out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv1d(self.out_channels * 4, self.out_channels, 1)
        self.conv2 = nn.Conv1d(self.out_channels * 2, self.out_channels, 1)
        
    def preprocess(self, cas_module_result: CasMVSNetModuleResult, cam_poses: torch.Tensor):
        v = len(cas_module_result.ref_view_result_list)
        b, c, h, w = cas_module_result.ref_view_result_list[0].img.shape
        cat_feature_list = []
        for view_idx in range(v):
            ref_view_result = cas_module_result.ref_view_result_list[view_idx]
            cam_pose_encoding = cam_poses[:, view_idx].view(b, -1, 1, 1).repeat(1, 1, h, w) # (b, pos, h, w)
            
            # element-wise concatnate
            cat_feature = torch.cat((
                ref_view_result.img, 
                ref_view_result.backbone["prob_volume"], 
                cam_pose_encoding
            ), dim=1)
            
            cat_feature = self.feature_projection(cat_feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # C -> 64
            
            cat_feature_list.append(cat_feature)
            
        return cat_feature_list, b
    
        
    def forward(self, cas_module_result: CasMVSNetModuleResult, cam_poses: torch.Tensor, multiview_trans_attn_split: int):
        cat_feature_list, b = self.preprocess(cas_module_result, cam_poses)
        
        transformed_feature_list = self.transformer(cat_feature_list, multiview_trans_attn_split) # (B, 4C, H, W) * V
        
        view_merged_features = torch.stack(transformed_feature_list, dim=1).mean(dim=1) # (B, 4C, H, W)
        
        separated_feat_1 = self.deconv1(view_merged_features) # (B, 2C, 2H, 2W)
        
        separated_feat_2 = self.deconv2(separated_feat_1) # (B, C, 4H, 4W)
        
        course_feat_1 = self.conv1(view_merged_features.view(b, self.out_channels * 4, -1)) # # (B, C, H, W)
        
        course_feat_2 = self.conv2(separated_feat_1.view(b, self.out_channels * 2, -1)) # # (B, C, 2H, 2W)
        
        return course_feat_1, course_feat_2, separated_feat_2.view(b, self.out_channels, -1)