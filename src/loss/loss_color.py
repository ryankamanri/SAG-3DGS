from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss


@dataclass
class LossColorCfg:
    weight: float

@dataclass
class LossColorCfgWrapper:
    color: LossColorCfg


class LossColor(Loss[LossColorCfg, LossColorCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if gaussians.others == {}: return torch.tensor(0., device="cuda")
        color_loss = Tensor(gaussians.others["color_loss"]).mean()
        return self.cfg.weight * color_loss
