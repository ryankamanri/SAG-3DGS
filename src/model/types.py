from dataclasses import dataclass, field

from jaxtyping import Float
from torch import Tensor


@dataclass
class EncoderOutput:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    others: dict[str, object] = field(default_factory=lambda: {})
