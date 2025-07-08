from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):
    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    g, _, d_sh = harmonics.shape
    default_d_sh = 4 ** 2  # Default is 4th order SH, which has 16 coefficients.
    harmonics = torch.cat((harmonics, torch.zeros(g, 3, default_d_sh - d_sh, device=harmonics.device)), dim=-1)
    
    harmonics_view_invariant = harmonics[..., 0]
    harmonics_rest = torch.zeros_like(harmonics[..., 1:].reshape(harmonics.shape[0], -1), device=harmonics.device) # we only export the view invariant part, so rest is zero

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes((harmonics.shape[-1] - 1) * 3)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        harmonics_rest.detach().cpu().contiguous().numpy(),
        inverse_sigmoid(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
