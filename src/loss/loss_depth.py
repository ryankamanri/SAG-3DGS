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
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if gaussians.others == {}: return torch.tensor(0.).cuda()
        cas_module_result: CasMVSNetModuleResult = gaussians.others["cas_module_result"]
        loss = torch.tensor(0.).cuda()
        for ref_view_result in cas_module_result.ref_view_result_list:
            loss += self.cfg.stage1_weight * torch.Tensor(ref_view_result.pretrained["stage1"]["depth"] - ref_view_result.backbone["stage1"]["depth"]).abs().mean()
            loss += self.cfg.stage2_weight * torch.Tensor(ref_view_result.pretrained["stage2"]["depth"] - ref_view_result.backbone["stage2"]["depth"]).abs().mean()
            loss += self.cfg.stage3_weight * torch.Tensor(ref_view_result.pretrained["stage3"]["depth"] - ref_view_result.backbone["stage3"]["depth"]).abs().mean()
        
        loss /= len(cas_module_result.ref_view_result_list)
        return loss * self.cfg.weight