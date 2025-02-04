from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Literal
import torch
from jaxtyping import Float
from torch import Tensor
from torch.optim import Optimizer

@dataclass
class EncoderOutput:
    """
        means: Float[Tensor, "batch gaussian dim"]
        covariances: Float[Tensor, "batch gaussian dim dim"]
        harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
        opacities: Float[Tensor, "batch gaussian"]
    """
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    others: dict[str, object] = field(default_factory=lambda: {})


def empty_encoder_output(dim=3, d_sh=1, device="cuda") -> EncoderOutput:
    return EncoderOutput(
        means=torch.zeros(1, 0, dim, device=device), 
        covariances=torch.zeros(1, 0, dim, dim, device=device), 
        harmonics=torch.zeros(1, 0, 3, d_sh, device=device), 
        opacities=torch.zeros(1, 0, device=device)
    )
    
@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    max_steps: int
    cosine_lr: bool
    delta_means_lr: float
    quaternion_lr: float
    scale_lr: float
    opacity_lr: float
    shs_d1_lr: float
    shs_d2_lr: float
    shs_d3_lr: float
    shs_d4_lr: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int

DepthRenderingMode = Literal['depth', 'log', 'disparity', 'relative_disparity']

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int

    
class IConfigureOptimizers(ABC):
    @abstractmethod
    def configure_optimizers(self, cfg: OptimizerCfg) -> list[dict]:
        """
        Set optimizer params dict list. A simple example: 
        `[{'params': module.parameters(), 'lr': 1e-5}, ...]`
        """
        pass
    
class IUpdatable(ABC):
    @abstractmethod
    def on_train_batch_start(self, batch, batch_idx: int, optimizer: Optimizer):
        pass