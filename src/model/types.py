from dataclasses import dataclass, field
import torch
from jaxtyping import Float
from torch import Tensor


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