import random
from dataclasses import dataclass, field
from typing import Literal

from jaxtyping import Float, Int64
from torch import Tensor

from ...utils.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_fps import ViewSamplerFPS, ViewSamplerFPSCfg
from .view_sampler_uniform import ViewSamplerUniform, ViewSamplerUniformCfg


@dataclass
class ViewSamplerMixedCfg:
    name: Literal["mixed"]
    num_context_views: int
    num_target_views: int
    samplers: list[Literal["fps", "uniform"]] = field(
        default_factory=lambda: ["fps", "uniform"]
    )


class ViewSamplerMixed(ViewSampler[ViewSamplerMixedCfg]):
    def __init__(
        self,
        cfg: ViewSamplerMixedCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        self.view_sampler_list: list[ViewSampler] = []

        if "fps" in cfg.samplers:
            self.view_sampler_list.append(
                ViewSamplerFPS(
                    ViewSamplerFPSCfg(
                        name="fps",
                        num_context_views=cfg.num_context_views,
                        num_target_views=cfg.num_target_views,
                    ),
                    stage,
                    is_overfitting,
                    cameras_are_circular,
                    step_tracker,
                )
            )

        if "uniform" in cfg.samplers:
            for start_index in (0, 1, 2):
                self.view_sampler_list.append(
                    ViewSamplerUniform(
                        ViewSamplerUniformCfg(
                            name="uniform",
                            num_context_views=cfg.num_context_views,
                            num_target_views=cfg.num_target_views,
                            start_index=start_index,
                        ),
                        stage,
                        is_overfitting,
                        cameras_are_circular,
                        step_tracker,
                    )
                )

        if not self.view_sampler_list:
            raise ValueError("ViewSamplerMixed requires at least one sampler")

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        num_views = extrinsics.shape[0]
        if not 1 <= self.cfg.num_context_views <= num_views:
            raise ValueError(
                "num_context_views must be in "
                f"[1, {num_views}], got {self.cfg.num_context_views}"
            )
        if not 1 <= self.cfg.num_target_views <= num_views:
            raise ValueError(
                "num_target_views must be in "
                f"[1, {num_views}], got {self.cfg.num_target_views}"
            )

        candidates = [
            sampler
            for sampler in self.view_sampler_list
            if not isinstance(sampler, ViewSamplerUniform)
            or sampler.cfg.start_index <= num_views - self.cfg.num_context_views
        ]
        if not candidates:
            raise ValueError(
                "No mixed sampler can select "
                f"{self.cfg.num_context_views} context views from {num_views} views"
            )
        return random.choice(candidates).sample(scene, extrinsics, intrinsics, **kwargs)

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
