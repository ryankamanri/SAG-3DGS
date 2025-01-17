from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..model.encoder.mvsnet.cas_mvsnet_module import CasMVSNetModuleResult
from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss



@dataclass
class LossDepthCfg:
    weight: float
    stage1_weight: float
    stage2_weight: float
    stage3_weight: float


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stage_weights = {
            1: self.cfg.stage1_weight, 
            2: self.cfg.stage2_weight, 
            3: self.cfg.stage3_weight
        }
        
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if gaussians.others == {}: return torch.tensor(0., device="cuda")
        cas_module_result: CasMVSNetModuleResult = gaussians.others["cas_module_result"]
        nears: torch.Tensor = gaussians.others["nears"] # (B, V)
        fars: torch.Tensor = gaussians.others["fars"]
        b, v = nears.shape
        loss = torch.tensor(0., device="cuda")
        view_idx = 0
        for ref_view_result in cas_module_result.ref_view_result_list:
            for stage in range(1, 4):
                delta_d = torch.Tensor(ref_view_result.pretrained[f"stage{stage}"]["depth"] - ref_view_result.backbone[f"stage{stage}"]["depth"]).abs() # (B, H, W)
                delta_d_normalized = delta_d / (fars[:, view_idx] - nears[:, view_idx]).view(b, 1, 1) # (B, H, W)
                confidence = torch.Tensor(ref_view_result.pretrained[f"stage{stage}"]["photometric_confidence"]) # (B, H, W)
                loss += (delta_d_normalized * confidence).mean() * self.stage_weights[stage]
            view_idx += 1
        
        loss /= len(cas_module_result.ref_view_result_list)
        return loss * self.cfg.weight