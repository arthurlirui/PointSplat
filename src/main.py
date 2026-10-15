import os
import sys
import warnings
from pathlib import Path

# Set up build environment for gsplat CUDA JIT compilation on Windows.
# This must happen before any import that triggers gsplat's lazy CUDA backend.
if sys.platform == "win32":
    try:
        # Add project root to path so we can import the setup module
        _project_root = str(Path(__file__).resolve().parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        import scripts.setup_build_env  # noqa: F401
    except Exception:
        pass

import hydra
import torch
from jaxtyping import install_import_hook
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.utilities import rank_zero_only

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.logger import LocalLogger
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.utils.ckpt_utils import find_latest_ckpt, load_pretrained_model
    from src.utils.step_tracker import StepTracker
    from src.utils.typing_utils import *


@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:
        output_dir = Path(cfg_dict.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_rank0(f"Saving outputs to {output_dir}.")

    callbacks = []
    logger = False
    if cfg.mode == "train":
        logger = LocalLogger(output_dir=output_dir)
        callbacks.extend(
            [
                LearningRateMonitor(logging_interval="step", log_momentum=True),
                ModelCheckpoint(
                    output_dir / "checkpoints",
                    every_n_train_steps=cfg.checkpointing.every_n_train_steps,
                    save_top_k=cfg.checkpointing.save_top_k,
                    monitor="info/global_step",
                    mode="max",
                    save_last=True,
                ),
            ]
        )
        for callback in callbacks:
            callback.CHECKPOINT_EQUALS_CHAR = "_"

    step_tracker = StepTracker()
    trainer = Trainer(
        precision=cfg.trainer.precision,
        max_epochs=-1,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=cfg.mode == "test",
        num_nodes=cfg.trainer.num_nodes,
        max_steps=cfg.trainer.max_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        plugins=LightningEnvironment() if cfg.trainer.use_plugins else None,
    )
    torch.manual_seed(cfg.trainer.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_wrapper = ModelWrapper(
        cfg.dataset,
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print_rank0("train:", len(data_module.train_dataloader()))
        print_rank0("val:", len(data_module.val_dataloader()))
    else:
        print_rank0("test:", len(data_module.test_dataloader()))

    checkpoint_path = None
    if cfg.checkpointing.resume and cfg.mode == "train":
        checkpoint_dir = output_dir / "checkpoints"
        if checkpoint_dir.is_dir() and any(checkpoint_dir.glob("*.ckpt")):
            checkpoint_path = find_latest_ckpt(checkpoint_dir)
            print_rank0(f"Resuming from {checkpoint_path}")
        else:
            print_rank0("No checkpoint found; starting from the configured weights.")
    elif cfg.checkpointing.load is not None:
        checkpoint_path = Path(cfg.checkpointing.load)

    status = load_pretrained_model(
        model_wrapper,
        cfg.checkpointing,
        cfg.mode,
    )
    print_rank0(status)

    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
