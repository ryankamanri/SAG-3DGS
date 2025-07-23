from dataclasses import dataclass
from math import exp

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss, LossCfg

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

@dataclass
class LossSSIMCfg(LossCfg):
    pass

@dataclass
class LossSSIMCfgWrapper:
    ssim: LossSSIMCfg


class LossSSIM(Loss[LossSSIMCfg, LossSSIMCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        gt = batch["target"]["image"]
        b, v, c, h, w = gt.shape
        loss = 0.
        for idx, stage in enumerate(gaussians.others["stages"]):
            prop = 1.0 / gaussians.others["scales"][idx]
            render = gaussians.others["stage_renders"][stage].color
            stage_gt = F.interpolate(gt.view(b*v, c, h, w), scale_factor=prop, mode="bilinear", align_corners=False).view(b, v, c, int(h * prop), int(w * prop))
            loss += (1. - ssim(stage_gt.squeeze(1), render.squeeze(1)))
        return loss / len(gaussians.others["stages"]) # average over stages
