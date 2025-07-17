from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss, LossCfg


@dataclass
class LossMseCfg(LossCfg):
    pass

@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["image"]
        b, v, c, h, w = gt.shape
        loss = 0.
        for idx, stage in enumerate(gaussians.others["stages"]):
            prop = 1.0 / gaussians.others["scales"][idx]
            render = gaussians.others["stage_renders"][stage].color
            stage_gt = F.interpolate(gt.view(b*v, c, h, w), scale_factor=prop, mode="bilinear", align_corners=False).view(b, v, c, int(h * prop), int(w * prop))
            delta = render - stage_gt
            loss += (delta**2).mean()
        return loss / 3.0 # average over stages
