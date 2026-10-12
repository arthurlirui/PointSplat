from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .shims.extr_shim import preprocess_poses, preprocess_sampled_views
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetExampleCfg(DatasetCfgCommon):
    name: Literal["example"]
    roots: list[Path]
    near: float = 1.0e-3
    far: float = 2.0
    shuffle: bool = True
    shuffle_val: bool = False
    train_times_per_scene: int = 1
    num_training_scenes: int = -1
    num_val_scenes: int = -1
    start_scene_training: int = 0
    start_scene_val: int = 0
    frame_per_training_scene: int = -1
    frame_per_val_scene: int = 1
    start_frame_training: int = 0
    start_frame_val: int = 0
    frame_gap_training: int = 1
    frame_gap_val: int = 1
    num_views: int = -1
    norm_extrinsic: bool = True
    norm_sampled_views: bool = True
    scene_scale_factor: float = 1.35


class DatasetExample(Dataset):
    """A documented starting point for a custom multi-view dataset.

    The on-disk format intentionally mirrors the processed DNA/RenBody layout:

    .. code-block:: text

        <root>/
        ├── validation_index.json
        ├── scenes/<scene>/transforms.json
        ├── scenes/<scene>/images/<view>/<frame>.png
        └── masks/<scene>/fmasks/<view>/<frame>.png

    ``transforms.json`` contains ``frame_ids`` and one entry per camera in
    ``frames``. ``file_path`` must contain ``000000``; it is replaced by each
    requested frame id. Camera intrinsics are normalized to [0, 1]: pixel-unit
    values are divided by image width/height in :meth:`_load_view`.
    """

    cfg: DatasetExampleCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetExampleCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler

        root = Path(cfg.roots[0]) if cfg.roots else Path("datasets/example_dataset")
        validation_path = root / "validation_index.json"
        if not validation_path.is_file():
            raise FileNotFoundError(
                f"Example dataset requires {validation_path} with validation scene names."
            )
        with validation_path.open("r") as handle:
            validation_scenes = set(json.load(handle))

        scene_names = sorted(
            path.name
            for path in (root / "scenes").iterdir()
            if path.is_dir() and (path / "transforms.json").is_file()
        )
        if self.stage == "train":
            scene_names = sorted(set(scene_names) - validation_scenes)
            if self.cfg.shuffle:
                random.shuffle(scene_names)
            start = self.cfg.start_scene_training
            count = self.cfg.num_training_scenes
        else:
            scene_names = sorted(set(scene_names) & validation_scenes)
            if self.cfg.shuffle_val:
                random.shuffle(scene_names)
            start = self.cfg.start_scene_val
            count = self.cfg.num_val_scenes
        if count == -1:
            count = len(scene_names)
        scene_names = scene_names[start : min(start + count, len(scene_names))]
        if not scene_names:
            raise RuntimeError(f"No example scenes selected for stage={stage}")

        self.root = root
        self.scene_infos = [self._load_scene(name) for name in scene_names]
        self.items = [
            (scene_index, frame_id)
            for scene_index, scene_info in enumerate(self.scene_infos)
            for frame_id in self._select_frame_ids(scene_info["frame_ids"])
        ]
        if not self.items:
            raise RuntimeError(f"No example frames selected for stage={stage}")

    def _load_scene(self, scene_name: str) -> dict:
        scene_path = self.root / "scenes" / scene_name
        with (scene_path / "transforms.json").open("r") as handle:
            metadata = json.load(handle)

        frames = metadata["frames"]
        if self.cfg.num_views != -1:
            frames = frames[: self.cfg.num_views]
        frame_ids = [int(frame_id) for frame_id in metadata["frame_ids"]]
        if not frames or not frame_ids:
            raise ValueError(f"Scene {scene_name} has no frames or frame_ids")
        return {
            "name": scene_name,
            "path": scene_path,
            "mask_path": self.root / "masks" / scene_name / "fmasks",
            "frames": frames,
            "frame_ids": frame_ids,
            "coordinate_system": metadata.get("coordinate_system", "opengl"),
        }

    def _select_frame_ids(self, frame_ids: list[int]) -> list[int]:
        if self.stage == "train":
            start = self.cfg.start_frame_training
            gap = self.cfg.frame_gap_training
            requested = self.cfg.frame_per_training_scene
        else:
            start = self.cfg.start_frame_val
            gap = self.cfg.frame_gap_val
            requested = self.cfg.frame_per_val_scene
        selected = list(range(start, len(frame_ids), gap))
        if requested != -1:
            selected = selected[:requested]
        return [frame_ids[index] for index in selected]

    def _image_path(self, scene_info: dict, frame: dict, frame_id: int) -> Path:
        # TODO(custom-data): Replace this path convention with your own if needed.
        relative = Path(frame["file_path"])
        relative = relative.with_name(f"{frame_id:0>6}{relative.suffix}")
        return scene_info["path"] / relative

    def _load_view(
        self,
        scene_info: dict,
        frame: dict,
        frame_id: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        image_path = self._image_path(scene_info, frame, frame_id)
        mask_path = (
            scene_info["mask_path"] / frame["camera_label"] / f"{frame_id:0>6}.png"
        )
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing example image/mask: {image_path}, {mask_path}"
            )

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = (torch.from_numpy(mask) > 0.5).float().unsqueeze(0)

        fx = float(frame["fl_x"])
        fy = float(frame["fl_y"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])
        height, width = image.shape[-2:]
        # The crop shim expects normalized intrinsics. Keep this conversion even
        # if the rest of the I/O is customized.
        if max(fx, fy, cx, cy) > 1.0:
            fx, cx = fx / width, cx / width
            fy, cy = fy / height, cy / height
        intrinsic = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        return image, mask, intrinsic

    def __getitem__(self, index: int) -> dict:
        scene_index, frame_id = self.items[index]
        scene_info = self.scene_infos[scene_index]
        scene_name = scene_info["name"]

        images, masks, intrinsics, c2ws = [], [], [], []
        for frame in scene_info["frames"]:
            image, mask, intrinsic = self._load_view(scene_info, frame, frame_id)
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            if scene_info["coordinate_system"] == "opengl":
                c2w[:3, 1:3] *= -1
            images.append(image)
            masks.append(mask)
            intrinsics.append(intrinsic)
            c2ws.append(c2w)

        images_tensor = torch.stack(images, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        intrinsics_tensor = torch.stack(intrinsics, dim=0)
        c2ws_tensor = torch.stack(c2ws, dim=0)

        scene_scale = 1.0 / self.cfg.scene_scale_factor
        if self.cfg.norm_extrinsic:
            c2ws_tensor, scene_scale = preprocess_poses(
                c2ws_tensor,
                scene_scale_factor=self.cfg.scene_scale_factor,
                return_scale=True,
            )

        context_indices, target_indices = self.view_sampler.sample(
            scene_name,
            c2ws_tensor,
            intrinsics_tensor,
        )
        context_c2ws = c2ws_tensor[context_indices]
        target_c2ws = c2ws_tensor[target_indices]
        if self.cfg.norm_sampled_views:
            context_c2ws, target_c2ws = preprocess_sampled_views(
                context_c2ws,
                target_c2ws,
                scene_scale=scene_scale,
            )

        def views(indices: Tensor, c2w: Tensor) -> dict:
            count = len(indices)
            return {
                "extrinsics": c2w,
                "intrinsics": intrinsics_tensor[indices],
                "image": images_tensor[indices],
                "mask": masks_tensor[indices],
                "near": torch.full((count,), self.cfg.near, dtype=torch.float32),
                "far": torch.full((count,), self.cfg.far, dtype=torch.float32),
                "index": indices,
            }

        example = {
            "context": views(context_indices, context_c2ws),
            "target": views(target_indices, target_c2ws),
            "scene": scene_name,
            "timestep": f"{frame_id:0>6}",
        }
        return apply_crop_shim(example, tuple(self.cfg.image_shape))

    def __len__(self) -> int:
        if self.stage == "train":
            return len(self.items) * self.cfg.train_times_per_scene
        return len(self.items)
