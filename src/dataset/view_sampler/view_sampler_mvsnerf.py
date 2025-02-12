from dataclasses import dataclass
import random
from typing import Literal

import torch
import numpy as np
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerMVSNeRFCfg:
    name: Literal["mvsnerf"]
    num_target_views_train: int
    num_context_views_train: int
    num_target_views_test: int
    num_context_views_test: int
    dtu_train_path: str
    dtu_test_path: str
    dtu_pairs_path: str
    test_pairs_path: str


class ViewSamplerMVSNeRF(ViewSampler[ViewSamplerMVSNeRFCfg]):
    
    def __init__(self, cfg, stage, is_overfitting, cameras_are_circular, step_tracker):
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        self.build_metas()
        self.test_pairs = torch.load(cfg.test_pairs_path)
    
    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i
    
    def build_metas(self):
        self.metas = []
        split_path = self.cfg.dtu_test_path if self.stage == "test" else self.cfg.dtu_train_path
        
        # with open(split_path) as f:
        #     self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if 'train' != self.stage else range(7)

        self.id_list = []

        # for scan in self.scans:
        with open(self.cfg.dtu_pairs_path) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                for light_idx in light_idxs:
                    self.metas += [(light_idx, ref_view, src_views)]
                    self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        if not scene.startswith("dtu"): # 
            # TODO: handle other test dataset and return
            pass
        
        # DTU
        image_count = len(self.metas)
        light_count = 7
        
        if self.stage == 'test':
            light_idx = 3
            return (
                (torch.tensor(self.test_pairs["dtu_train"]) * light_count + light_idx)[:self.num_context_views], 
                (torch.tensor(self.test_pairs["dtu_test"]) * light_count + light_idx)[:self.num_target_views]
            )
        
        light_idx, target_view, src_views = self.metas[random.randint(0, image_count - 1)]

        ids = torch.randperm(5)[:self.num_context_views]
        
        return (
            torch.tensor([src_views[i] * light_count + light_idx for i in ids], device=device), 
            torch.tensor([target_view * light_count + light_idx])
        )
        
    def sample_fine_tune(self, scene, extrinsics, intrinsics, device = ..., **kwargs):
        if self.stage != 'test': return None
        if scene.startswith('dtu'): # DTU
            return torch.tensor(self.test_pairs["dtu_train"])
        # TODO: add other datasets.
        
        
        
    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views_test if self.stage == 'test' else self.cfg.num_context_views_train

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views_test if self.stage == 'test' else self.cfg.num_target_views_train
