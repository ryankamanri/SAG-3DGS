from typing import Optional

from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .encoder_costvolume_incremental import EncoderCostVolumeIncremental, EncoderCostVolumeIncrementalCfg
from .encoder_cascade import EncoderCascade, EncoderCascadeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "incremental": (EncoderCostVolumeIncremental, None), 
    "cascade": (EncoderCascade, None)
}

EncoderCfg = EncoderCascadeCfg | EncoderCostVolumeCfg | EncoderCostVolumeIncrementalCfg # Note that this cfg should be changed manually.


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
