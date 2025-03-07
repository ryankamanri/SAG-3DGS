from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput

@dataclass
class LossCfg:
    weight: float

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        assert isinstance(self.cfg, LossCfg), f"{type(self.cfg)} must be a subclass of {LossCfg}!"
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pass
