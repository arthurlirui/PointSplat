from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.model.encoder import get_encoder
from src.model.encoder.encoder_pointsplat import get_vhull_points


def _finite_or_raise(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def _save_overlay(
    output_dir: Path,
    image: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    from PIL import Image

    overlay = image * 0.7 + mask * 0.3
    overlay = (overlay.detach().clip(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(overlay).save(output_dir / "context_mask_overlay.png")


def _check_initial_opacity(encoder_cfg) -> tuple[float, float]:
    """Verify that a randomly initialized training run does not rank-equal opacities."""
    prune_ratio = encoder_cfg.gaussians_decoder.prune_ratio
    if prune_ratio != 0:
        raise ValueError(
            "Training must disable opacity pruning at initialization "
                f"(prune_ratio={prune_ratio})."
        )

    encoder, _ = get_encoder(encoder_cfg)
    opacity_layer = encoder.output_tokenizer.out_layers["opacity"]
    initial_density = encoder_cfg.gaussians_decoder.init_density
    expected_bias = torch.logit(torch.tensor(initial_density, dtype=torch.float32))
    if not torch.equal(opacity_layer.weight, torch.zeros_like(opacity_layer.weight)):
        raise ValueError("The opacity head weight must be initialized to zero.")
    if not torch.allclose(
        opacity_layer.bias,
        expected_bias.reshape_as(opacity_layer.bias),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "The opacity head bias must initialize every Gaussian to "
            f"init_density={initial_density}."
        )
    return initial_density, prune_ratio


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg_dict: dict) -> None:
    cfg = load_typed_root_config(cfg_dict)
    initial_opacity = None
    opacity_prune_ratio = None
    if cfg.mode == "train":
        initial_opacity, opacity_prune_ratio = _check_initial_opacity(cfg.model.encoder)

    dataset = get_dataset(cfg.dataset, "train", None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    context = batch["context"]
    for name, tensor in (
        ("context.image", context["image"]),
        ("context.mask", context["mask"]),
        ("context.extrinsics", context["extrinsics"]),
        ("context.intrinsics", context["intrinsics"]),
        ("target.image", batch["target"]["image"]),
    ):
        _finite_or_raise(tensor, name)

    if context["image"].shape[0] != 1:
        raise ValueError("Debug the dataset with data_loader.train.batch_size=1")
    mask = context["mask"][:, :, 0]
    foreground_counts = mask.flatten(2).sum(dim=-1).long()
    per_view_foreground_counts = foreground_counts[0]
    if not (per_view_foreground_counts > 0).all():
        raise ValueError(
            f"Every context view needs foreground pixels; got {per_view_foreground_counts.tolist()}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_mask = context["mask"].to(device)
    context_extrinsics = context["extrinsics"].to(device)
    context_intrinsics = context["intrinsics"].to(device)
    encoder_cfg = cfg.model.encoder
    vhull_mask, _, coords_world = get_vhull_points(
        context_mask,
        context_extrinsics,
        context_intrinsics,
        bounds=torch.tensor([encoder_cfg.bounds], device=device),
        resolution=encoder_cfg.vhull_res,
        min_cell_size=encoder_cfg.min_cell_size,
        cube_sample=encoder_cfg.cube_sample,
        threshold=encoder_cfg.threshold,
    )

    point_counts = vhull_mask.long().flatten(1).sum(dim=1)
    if not (point_counts > 0).all():
        raise ValueError(f"Visual hull is empty; point_counts={point_counts.tolist()}")

    points = torch.cat(
        [
            coords_world[batch_index][vhull_mask[batch_index]]
            for batch_index in range(coords_world.shape[0])
        ],
        dim=0,
    )
    _finite_or_raise(points, "visual hull points")
    unique_per_axis = [torch.unique(points[:, axis]).numel() for axis in range(3)]
    if min(unique_per_axis) < 2:
        raise ValueError(
            f"Visual hull is degenerate; point_count={points.shape[0]}, "
            f"unique_per_axis={unique_per_axis}"
        )

    output_dir = Path(cfg_dict.output_dir) / "dataset_debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_overlay(output_dir, context["image"][0, 0], mask[0, 0])

    scene = batch["scene"][0]
    print(f"scene={scene}")
    print(f"context_shape={tuple(context['image'].shape)}")
    print(f"target_shape={tuple(batch['target']['image'].shape)}")
    print(f"foreground_pixel_counts={per_view_foreground_counts.tolist()}")
    print(
        "foreground_ratios="
        + str(
            [
                round(value, 6)
                for value in (
                    per_view_foreground_counts / mask[0, 0].numel()
                ).tolist()
            ]
        )
    )
    print(f"vhull_point_counts={point_counts.tolist()}")
    print(f"vhull_bbox_min={points.min(dim=0).values.detach().cpu().tolist()}")
    print(f"vhull_bbox_max={points.max(dim=0).values.detach().cpu().tolist()}")
    print(f"vhull_unique_per_axis={unique_per_axis}")
    if initial_opacity is not None:
        print(f"initial_opacity={initial_opacity:.6f}")
        print(f"opacity_prune_ratio={opacity_prune_ratio}")
    print(f"overlay={output_dir / 'context_mask_overlay.png'}")


if __name__ == "__main__":
    main()
