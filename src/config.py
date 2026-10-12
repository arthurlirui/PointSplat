
from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg
from .utils.ckpt_utils import CheckpointingCfg
from .utils.typing_utils import *


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    seed: int
    num_nodes: int
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    log_every_n_steps: int
    precision: Literal["64", "64-true", "32", "32-true", "16", "16-mixed", "bf16", "bf16-mixed"]
    use_plugins: bool



@dataclass
class RootCfg:
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # dacite cannot directly construct a union from a joined Hydra mapping.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {key: value}}), Dummy).dummy
        for key, value in joined.items()
    ]
