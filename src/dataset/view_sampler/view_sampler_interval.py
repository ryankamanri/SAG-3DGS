from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerIntervalCfg:
    name: Literal["interval"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | None
    target_views: list[int] | None


class ViewSamplerInterval(ViewSampler[ViewSamplerIntervalCfg]):
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
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape
        
        step = 10
        dist = self.cfg.num_context_views * step
        
        index_context_begin = torch.randint(
            0,
            num_views - dist if num_views - dist > 0 else 1,
            size=(),
            device=device,
        ).item()
        
        index_context = torch.arange(
            index_context_begin, 
            index_context_begin + dist if num_views > dist else index_context_begin + num_views, 
            step=step
        )

        # Allow the context views to be fixed.
        if self.cfg.context_views is not None:
            assert len(self.cfg.context_views) == self.cfg.num_context_views
            index_context = torch.tensor(
                self.cfg.context_views, dtype=torch.int64, device=device
            )

        index_target = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        # Allow the target views to be fixed.
        if self.cfg.target_views is not None:
            assert len(self.cfg.target_views) == self.cfg.num_target_views
            index_target = torch.tensor(
                self.cfg.target_views, dtype=torch.int64, device=device
            )

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
