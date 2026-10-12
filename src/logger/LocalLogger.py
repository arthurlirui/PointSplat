import json
import time
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class LocalLogger(Logger):
    """Small dependency-free logger with optional TensorBoard export."""

    def __init__(self, output_dir: Path, use_tensorboard: bool = True) -> None:
        super().__init__()
        self._log_dir = output_dir / "local_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self._log_dir / "metrics.jsonl"
        self.metrics_file.touch(exist_ok=True)
        self.tensorboard_logger = None

        if use_tensorboard:
            try:
                from pytorch_lightning.loggers import TensorBoardLogger

                tensorboard_dir = self._log_dir / "tensorboard"
                self.tensorboard_logger = TensorBoardLogger(
                    save_dir=str(tensorboard_dir),
                    name="",
                    version="",
                )
            except ImportError:
                pass

    @property
    def name(self) -> str:
        return "local"

    @property
    def version(self) -> int:
        return 0

    @rank_zero_only
    def log_hyperparams(self, params) -> None:
        return None

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not metrics:
            return

        cleaned_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                cleaned_metrics[key] = value.item()
            elif isinstance(value, (int, float)):
                cleaned_metrics[key] = value
            else:
                continue

        if self.tensorboard_logger is not None:
            self.tensorboard_logger.log_metrics(cleaned_metrics, step)

        with self.metrics_file.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "timestamp": time.time(),
                        "metrics": cleaned_metrics,
                    }
                )
                + "\n"
            )

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: int | None = None,
        **kwargs,
    ) -> None:
        return None
