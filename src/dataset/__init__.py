from torch.utils.data import Dataset

from ..utils.step_tracker import StepTracker
from .dataset_renbody import DatasetRENBODY, DatasetRENBODYCfg
from .dataset_thuman import DatasetTHuman, DatasetTHumanCfg
from .dataset_example import DatasetExample, DatasetExampleCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "renbody": DatasetRENBODY,
    "thuman": DatasetTHuman,
    "example": DatasetExample,
}


DatasetCfg = DatasetRENBODYCfg | DatasetTHumanCfg | DatasetExampleCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)
