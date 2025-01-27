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
    weight_gaussian_struct_loss: float

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
        
        # gaussian structure also need to be restricted.
        scales_list: torch.Tensor = gaussians.others["scales_list"] # [(N, 3) * B]
        scales = torch.cat(scales_list, dim=0)
        sorted_scale = scales.sort(dim=2).values
        gaussian_struct_loss = (1 - (sorted_scale[..., 1] / sorted_scale[..., 2]).mean(dim=1)).mean() ** 2
        
        return self.cfg.weight * (
            self.cfg.weight_existence_loss * existence_loss +
            self.cfg.weight_offset_loss * offset_loss +
            self.cfg.weight_gaussian_struct_loss * gaussian_struct_loss)
