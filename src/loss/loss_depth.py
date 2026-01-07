from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..model.encoder.mvsnet.cas_mvsnet_module import CasMVSNetModuleResult
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss, LossCfg



@dataclass
class LossDepthCfg(LossCfg):
    apply_before_step: int
    stage1_weight: float
    stage2_weight: float
    stage3_weight: float
    stage4_weight: float


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stage_weights = {
            1: self.cfg.stage1_weight, 
            2: self.cfg.stage2_weight, 
            3: self.cfg.stage3_weight, 
            4: self.cfg.stage4_weight, 
        }
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if gaussians.others == {}: return torch.tensor(0., device="cuda")
        
        # After the specified step, don't apply the loss.
        if global_step > self.cfg.apply_before_step:
            return torch.tensor(0, dtype=torch.float32, device="cuda")
        
        depth_gt, depth_gt_mask = torch.tensor(batch["context"]["depth"]), torch.tensor(batch["context"]["depth_mask"])
        b, v, h, w = depth_gt.shape
        stages, scales = gaussians.others["stages"], gaussians.others["scales"]
        depth_gt_stages = [F.interpolate(depth_gt, (h // scale, w // scale)) for scale in scales]
        depth_gt_mask_stages = [F.interpolate(depth_gt_mask, (h // scale, w // scale)) for scale in scales]
        
        cas_module_result: CasMVSNetModuleResult = gaussians.others["cas_module_result"]
        nears: torch.Tensor = gaussians.others["nears"] # (B, V)
        fars: torch.Tensor = gaussians.others["fars"]
        umeyama_relative_scale: torch.Tensor = gaussians.others["s"] # (B)
        b, v = nears.shape
        loss = torch.tensor(0., device="cuda")
        depth_gt_mean, depth_pred_mean = torch.tensor(0., device="cuda"), torch.tensor(0., device="cuda")
        view_idx = 0
        for ref_view_result in cas_module_result.ref_view_result_list:
            for idx, stage in enumerate(stages):
                mask = depth_gt_mask_stages[idx][:, view_idx] == 1.
                if not mask.any(): continue # Skip if no valid pixels in this view for this stage.
                cur_depth_gt_origin, cur_depth_pred = depth_gt_stages[idx][:, view_idx], ref_view_result.backbone[stage]["depth"] # (B, H, W)
                cur_depth_gt = cur_depth_gt_origin * umeyama_relative_scale.view(b, 1, 1)

                # compute confidence of depth from umeyama
                with torch.no_grad():
                    depth_gt_mean += cur_depth_gt.mean() * self.stage_weights[idx+1]
                    depth_pred_mean += cur_depth_pred.mean() * self.stage_weights[idx+1]
                    relative_scale = cur_depth_pred.view(b, -1).mean(dim=-1) / (cur_depth_gt_origin.view(b, -1).mean(dim=-1) + 1e-8)
                    stack_2_scale = torch.stack((relative_scale, umeyama_relative_scale), dim=1)
                    confidence = (stack_2_scale.min(dim=1).values / stack_2_scale.max(dim=1).values) ** 2 # (B)
                    
                near, far = nears[0, 0], fars[0, 0]
                delta_d = torch.Tensor(1. / cur_depth_pred.clamp(min=1e-3) - 1. / cur_depth_gt.clamp(min=1e-3)).abs() / (1. / near - 1. / far) # (N)
                loss += (delta_d * confidence.view(b, 1, 1)).mean() * self.stage_weights[idx+1]
                
            view_idx += 1
        
        depth_gt_mean /= len(cas_module_result.ref_view_result_list)
        depth_pred_mean /= len(cas_module_result.ref_view_result_list)
        gaussians.others["depth_gt_mean"] = depth_gt_mean
        gaussians.others["depth_pred_mean"] = depth_pred_mean
        loss /= len(cas_module_result.ref_view_result_list)
        return loss