import re
from pathlib import Path

import torch

from .typing_utils import *


@dataclass
class CheckpointingCfg:
    load: Optional[str]
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    pretrained_depth: Optional[str]
    pretrained_encoder: Optional[str]
    pretrained_decoder: Optional[str]
    no_strict_load: bool
    resume: bool


def find_latest_ckpt(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

    def checkpoint_step(path: Path) -> int:
        match = re.search(r"step[=_](\d+)", path.stem)
        return int(match.group(1)) if match else -1

    checkpoints = [path for path in ckpt_dir.iterdir() if path.suffix == ".ckpt"]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(checkpoints, key=checkpoint_step)


def _unwrap_state_dict(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "network", "model"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def _format_load_status(name: str, status) -> str:
    missing = list(getattr(status, "missing_keys", []) or [])
    unexpected = list(getattr(status, "unexpected_keys", []) or [])
    return f"[Loaded {name}] missing={missing}, unexpected={unexpected}"


def load_pretrained_model(
    model_wrapper,
    checkpointing_cfg: CheckpointingCfg,
    mode: str = "test",
) -> str:
    results = []
    strict_load = not checkpointing_cfg.no_strict_load

    if checkpointing_cfg.pretrained_model is not None:
        checkpoint = _unwrap_state_dict(
            torch.load(
                checkpointing_cfg.pretrained_model,
                map_location="cpu",
            )
        )
        status = model_wrapper.load_state_dict(checkpoint, strict=strict_load)
        results.append(_format_load_status("pretrained_model", status))

    if checkpointing_cfg.pretrained_encoder is not None:
        checkpoint = _unwrap_state_dict(
            torch.load(
                checkpointing_cfg.pretrained_encoder,
                map_location="cpu",
            )
        )
        status = model_wrapper.encoder.load_state_dict(checkpoint, strict=False)
        results.append(_format_load_status("pretrained_encoder", status))

    if checkpointing_cfg.pretrained_decoder is not None:
        checkpoint = _unwrap_state_dict(
            torch.load(
                checkpointing_cfg.pretrained_decoder,
                map_location="cpu",
            )
        )
        status = model_wrapper.decoder.load_state_dict(checkpoint, strict=False)
        results.append(_format_load_status("pretrained_decoder", status))

    if checkpointing_cfg.pretrained_depth is not None:
        if not hasattr(model_wrapper.encoder, "depth_predictor"):
            raise AttributeError(
                "checkpointing.pretrained_depth was set, but the encoder has no depth_predictor."
            )
        checkpoint = _unwrap_state_dict(
            torch.load(
                checkpointing_cfg.pretrained_depth,
                map_location="cpu",
            )
        )
        status = model_wrapper.encoder.depth_predictor.load_state_dict(
            checkpoint,
            strict=strict_load,
        )
        results.append(_format_load_status("pretrained_depth", status))

    if not results:
        return "[No pretrained checkpoint requested]"
    return "\n".join(results)
