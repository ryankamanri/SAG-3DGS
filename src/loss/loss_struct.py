from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss


@dataclass
class LossStructCfg:
    weight: float
    weight_existence_loss: float
    weight_offset_loss: float

@dataclass
class LossStructCfgWrapper:
    struct: LossStructCfg


class LossStruct(Loss[LossStructCfg, LossStructCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        if gaussians.others == {}: return torch.tensor(0., device="cuda")
        existence_loss = Tensor(gaussians.others["existence_loss"]).mean()
        offset_loss = Tensor(gaussians.others["offset_loss"]).mean()
        return self.cfg.weight * (
            self.cfg.weight_existence_loss * existence_loss +
            self.cfg.weight_offset_loss * offset_loss)
