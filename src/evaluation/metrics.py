from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch import Tensor

# follow GeFU & MVSGaussian
import skimage
assert skimage.__version__ == "0.19.0", "skimage version should be 0.19.0, please install it with pip install skimage==0.19.0"


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    psnr = [
        peak_signal_noise_ratio(
            gt.permute(1, 2, 0).detach().cpu().numpy(),
            hat.permute(1, 2, 0).detach().cpu().numpy(),
            data_range=1.0
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(psnr, dtype=predicted.dtype, device=predicted.device)
    


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward((ground_truth - 0.5) * 2, (predicted - 0.5) * 2)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.permute(1, 2, 0).detach().cpu().numpy(),
            hat.permute(1, 2, 0).detach().cpu().numpy(),
            multichannel=True
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
