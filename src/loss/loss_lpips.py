from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import EncoderOutput
from .loss import Loss, LossCfg


@dataclass
class LossLpipsCfg(LossCfg):
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: EncoderOutput,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        gt = batch["target"]["image"]
        b, v, c, h, w = gt.shape
        loss = 0.
        for stage, idx in zip(("stage1", "stage2", "stage3"), range(3)):
            prop = 1 / 2 ** (2 - idx)
            render = gaussians.others["stage_renders"][stage].color
            stage_gt = F.interpolate(gt.view(b*v, c, h, w), scale_factor=prop, mode="bilinear", align_corners=False).view(b, v, c, int(h * prop), int(w * prop))
            loss += self.lpips.forward(
                rearrange(render, "b v c h w -> (b v) c h w"),
                rearrange(stage_gt, "b v c h w -> (b v) c h w"),
                normalize=True,
            ).mean()
        return loss / 3.0 # average over stages
