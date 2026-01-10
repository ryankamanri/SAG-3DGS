from typing import Optional

from .encoder import Encoder
from .encoder_costvolume_incremental import EncoderCostVolumeIncremental, EncoderCostVolumeIncrementalCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "incremental": (EncoderCostVolumeIncremental, None), 
}

EncoderCfg = EncoderCostVolumeIncrementalCfg # Note that this cfg should be changed manually.


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
