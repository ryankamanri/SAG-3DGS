from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_ssim import LossSSIM, LossSSIMCfgWrapper
from .loss_struct import LossStruct, LossStructCfgWrapper
from .loss_color import LossColor, LossColorCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossSSIMCfgWrapper: LossSSIM,
    LossStructCfgWrapper: LossStruct, 
    LossColorCfgWrapper: LossColor
}

LossCfgWrapper = LossColorCfgWrapper | LossDepthCfgWrapper | LossLpipsCfgWrapper | LossSSIMCfgWrapper | LossMseCfgWrapper | LossStructCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
