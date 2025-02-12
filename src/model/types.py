from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Literal
import torch
from jaxtyping import Float
from torch import Tensor
import torch.nn as nn
from ..utils import inverse_sigmoid


@dataclass
class EncoderOutput:
    """
        means: Float[Tensor, "batch gaussian dim"]
        scales: Float[Tensor, "batch gaussian dim"]
        rotations: Float[Tensor, "batch gaussian 4"]
        harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
        opacities: Float[Tensor, "batch gaussian"]
    """
    means: Float[Tensor, "batch gaussian dim"]
    scales: Float[Tensor, "batch gaussian dim"]
    rotations: Float[Tensor, "batch gaussian 4"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    others: dict[str, object] = field(default_factory=lambda: {})


def empty_encoder_output(dim=3, d_sh=1, device="cuda") -> EncoderOutput:
    return EncoderOutput(
        means=torch.zeros(1, 0, dim, device=device), 
        scales=torch.zeros(1, 0, dim, device=device), 
        rotations=torch.zeros(1, 0, 4, device=device), 
        harmonics=torch.zeros(1, 0, 3, d_sh, device=device), 
        opacities=torch.zeros(1, 0, device=device)
    )
    
# configs
@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
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
class FineTuneCfg:
    fine_tune_steps: int
    means_lr: float
    quaternion_lr: float
    scale_lr: float
    opacity_lr: float
    shs_d1_lr: float
    shs_d2_lr: float
    shs_d3_lr: float
    shs_d4_lr: float
    lambda_dssim: float

@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    fine_tune: bool
    fine_tune_cfg: FineTuneCfg
    


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
    
    
    
    
CHANNELS = 3
SH_DEGREE = 4
SLICE_SHS_D1 = slice(0, 1 ** 2) # sh degree 1
SLICE_SHS_D2 = slice(1 ** 2, 2 ** 2)
SLICE_SHS_D3 = slice(2 ** 2, 3 ** 2)
SLICE_SHS_D4 = slice(3 ** 2, 4 ** 2)
slice_shs_list = [SLICE_SHS_D1, SLICE_SHS_D2, SLICE_SHS_D3, SLICE_SHS_D4]

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
    

# TODO: change the activation equal to 3DGS
scaling_activation = torch.exp
scaling_deactivation = torch.log
quaternion_activation = torch.nn.functional.normalize
opacity_deactivation = inverse_sigmoid
opacity_activation = torch.sigmoid


class FineTuneGaussianWrapper:
    means: nn.Parameter
    scales: nn.Parameter
    rotations: nn.Parameter
    harmonics: list[nn.Parameter] = []
    opacities: nn.Parameter
    
    def __init__(self, gaussians: EncoderOutput, cfg: FineTuneCfg):
        self.means = nn.Parameter(gaussians.means.clone().requires_grad_(), requires_grad=True)
        self.scales = nn.Parameter(scaling_deactivation(gaussians.scales.clone().requires_grad_()), requires_grad=True)
        self.rotations = nn.Parameter(gaussians.rotations.clone().requires_grad_(), requires_grad=True)
        for i in range(SH_DEGREE):
            self.harmonics.append(nn.Parameter(gaussians.harmonics[..., slice_shs_list[i]].clone().requires_grad_(), requires_grad=True))
        self.opacities = nn.Parameter(opacity_deactivation(gaussians.opacities.clone().requires_grad_()), requires_grad=True)
        
        self.optimizer = torch.optim.Adam(params=[
            {'params': [self.means], 'lr': cfg.means_lr, "name": "means"},
            {'params': [self.scales], 'lr': cfg.scale_lr, "name": "scales"},
            {'params': [self.rotations], 'lr': cfg.quaternion_lr, "name": "rotations"},
            {'params': [self.opacities], 'lr': cfg.opacity_lr, "name": "opacities"}, 
            {'params': [self.harmonics[0]], 'lr': cfg.shs_d1_lr, "name": "shs_d1"}, 
            {'params': [self.harmonics[1]], 'lr': cfg.shs_d2_lr, "name": "shs_d2"}, 
            {'params': [self.harmonics[2]], 'lr': cfg.shs_d3_lr, "name": "shs_d3"}, 
            {'params': [self.harmonics[3]], 'lr': cfg.shs_d4_lr, "name": "shs_d4"}, 
        ], lr=0.0, eps=1e-15)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer, 
            start_factor=1, 
            end_factor=0, 
            total_iters=cfg.fine_tune_steps
            )
        pass
    
    def get_gaussians(self) -> EncoderOutput:
        return EncoderOutput(
            means=self.means, 
            scales=scaling_activation(self.scales), 
            rotations=quaternion_activation(self.rotations), 
            harmonics=torch.cat(self.harmonics, dim=-1), 
            opacities=opacity_activation(self.opacities)
        )
        
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
    pass