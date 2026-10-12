from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from ..utils.nn_utils import convert_to_buffer
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    chunk_size: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)
        self.lpips = LPIPS(net="vgg", verbose=False)

        # The VGG/LPIPS evaluator is fixed, is not trained, and should not bloat
        # checkpoints.
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]
        if global_step < self.cfg.apply_after_step:
            return image.new_tensor(0.0)

        prediction_color = prediction.color.float()
        target_image = image.float()
        batch_size = prediction_color.shape[0] * prediction_color.shape[1]
        chunk_size = self.cfg.chunk_size
        if chunk_size <= 0 or chunk_size >= batch_size:
            loss = self.lpips.forward(
                rearrange(prediction_color, "b v c h w -> (b v) c h w"),
                rearrange(target_image, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
            return self.cfg.weight * loss.mean()

        losses = []
        flat_prediction = rearrange(prediction_color, "b v c h w -> (b v) c h w")
        flat_target = rearrange(target_image, "b v c h w -> (b v) c h w")
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            losses.append(
                self.lpips.forward(
                    flat_prediction[start:end],
                    flat_target[start:end],
                    normalize=True,
                )
            )
        return self.cfg.weight * torch.cat(losses, dim=0).mean()
