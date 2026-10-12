from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import random
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerUniformCfg:
    name: Literal["uniform"]
    num_context_views: int
    num_target_views: int
    start_index: int = 1


class ViewSamplerUniform(ViewSampler[ViewSamplerUniformCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Sample evenly spaced context views and random target views."""
        num_views, _, _ = extrinsics.shape
        if self.cfg.start_index + self.cfg.num_context_views <= num_views:
            index_context = torch.linspace(
                self.cfg.start_index,
                num_views - 1,
                self.cfg.num_context_views,
            ).round().long()
        else:
            index_context = torch.linspace(
                0,
                num_views - 1,
                self.cfg.num_context_views,
            ).round().long()
        available_target_indices = list(range(num_views))
        index_target = torch.tensor(
            random.sample(available_target_indices, self.cfg.num_target_views),
            dtype=torch.int64,
        )
        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
