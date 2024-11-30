from dataclasses import dataclass
import torch
from torch import nn
from ..mvsnet import CascadeMVSNet, generate_point_cloud_from_depth_maps



@dataclass
class ReferenceViewResult:
    pretrained: dict[str, object]
    backbone: dict[str, object]

@dataclass
class CasMVSNetModuleResult:
    ref_view_result_list: list[ReferenceViewResult]
    registed_pcd: dict[str, list[torch.Tensor]]

class CasMVSNetModule(nn.Module):
    pretrained_cas_mvsnet: CascadeMVSNet
    backbone_cas_mvsnet: CascadeMVSNet

    def __init__(self, cas_mvsnet_ckpt_path) -> None:
        super().__init__()
        print(f"loading checkpoint from {cas_mvsnet_ckpt_path}...")
        # initialize pretrained mvsnet
        self.pretrained_cas_mvsnet = CascadeMVSNet(refine=False)
        state_dict = torch.load(cas_mvsnet_ckpt_path)
        self.pretrained_cas_mvsnet.load_state_dict(state_dict["model"])
        self.pretrained_cas_mvsnet.eval()
        
        self.backbone_cas_mvsnet = CascadeMVSNet(refine=False, is_used_on_nvs=True)
        
    def preprocess(self, context, ndepths = 192):
        imgs : torch.Tensor = context["origin_image"] # (B, V, C, H, W)
        c2w_extrinsics : torch.Tensor = context["extrinsics"] # (B, V, 4, 4)
        normalized_intrinsics : torch.Tensor = context["origin_intrinsics"] # (B, V, 3, 3)
        nears, fars = context["near"], context["far"] # (B, V)
        b, v, c, h, w = imgs.shape
        # crop image to adapt to mvsnet (h and w can be devided by 16)
        if h % 16 != 0:
            imgs = imgs[..., :-(h % 16), :]
        if w % 16 != 0:
            imgs = imgs[..., :-(w % 16)]
        # convert extrinsics c2w -> w2c
        extrinsics = c2w_extrinsics.inverse()
        # make the intrinsic mat adapt to feature map (w / 4, w / 4)
        intrinsics = normalized_intrinsics.clone()
        intrinsics[..., 0, :] *= w / 4
        intrinsics[..., 1, :] *= h / 4
        
        # multi-stage proj_mats
        # proj_matrices (B, V, 2(intr & extr), 4, 4)
        proj_matrices = torch.zeros(b, v, 2, 4, 4).cuda()
        proj_matrices[..., 0, :, :] = extrinsics
        proj_matrices[..., 1, :3, :3] = intrinsics
        
        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 2
        stage3_pjmats = proj_matrices.clone()
        stage3_pjmats[..., 1, :2, :] = proj_matrices[..., 1, :2, :] * 4

        proj_mat = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }
        
        # the intrinsics adapts depth map (w, h), not (w / 4, h / 4)
        intrinsics[..., :2, :] *= 4
        
        # prepare depth bound (inverse depth) [v*b, d]
        min_depth = (1.0 / fars).view(b*v, 1)
        max_depth = (1.0 / nears).view(b*v, 1)
        depth_candi_curr = (
            min_depth
            + torch.linspace(0.0, 1.0, ndepths).unsqueeze(0).to(min_depth.device)
            * (max_depth - min_depth)
        )
        depth_values = 1 / depth_candi_curr # (B*V, ndepths)
        depth_values = depth_values.reshape(b, v, ndepths) # (B, V, ndepths)
        return imgs, extrinsics, intrinsics, proj_mat, depth_values
        
    def forward(self, context, ndepths=128):
        imgs, extrinsics, intrinsics, proj_mat, depth_values = self.preprocess(context, ndepths)
        b, v, c, h, w = imgs.shape
        
        result = CasMVSNetModuleResult(ref_view_result_list=[], registed_pcd={})
        
        pretrained_depths_est = [] # depth map list
        pretrained_photometric_confidences = []
        # for every reference image, the mvsnet will generate a depth map and a photometric confidence map
        for vi in range(v):
            with torch.no_grad(): # necessary to reduce the memory
                pretrained_outputs = self.pretrained_cas_mvsnet(imgs, proj_mat, depth_values[:, vi, :]) # depth and photometric_confidence
            pretrained_depths_est.append(pretrained_outputs["depth"])
            pretrained_photometric_confidences.append(pretrained_outputs["photometric_confidence"])
            
            backbone_outputs = self.backbone_cas_mvsnet(imgs, proj_mat, depth_values[:, vi, :])
            result.ref_view_result_list.append(ReferenceViewResult(pretrained_outputs, backbone_outputs))
            
            # switch to next image
            imgs = imgs.roll(dims=1, shifts=-1)
            for stage in proj_mat:
                proj_mat[stage] = proj_mat[stage].roll(dims=1, shifts=-1)
                    
        pcd = generate_point_cloud_from_depth_maps(imgs, extrinsics, intrinsics, pretrained_depths_est, pretrained_photometric_confidences)
        result.registed_pcd = pcd
        
        return result