"""Generate human segmentation masks for a PointSplat dataset.

This script runs a segmentation model over all images in a dataset and saves
the resulting masks in the expected directory layout, so the standard
PointSplat data pipeline can use them without modification.

Usage examples
--------------
# Generate masks using SAM for a DNA-Rendering dataset
python -m src.modules.segmentation.generate_masks \
    --input datasets/DNA_Rendering_example/scenes \
    --output datasets/DNA_Rendering_example/masks \
    --backend sam --model-type vit_b --layout dna_rendering

# Generate masks using SegFormer
python -m src.modules.segmentation.generate_masks \
    --input datasets/my_data/img \
    --output datasets/my_data/mask \
    --backend segformer --layout thuman

# Use simple threshold (no model needed, works with green-screen data)
python -m src.modules.segmentation.generate_masks \
    --input datasets/DNA_Rendering_example/scenes \
    --output datasets/DNA_Rendering_example/masks \
    --backend threshold --bg-color 0.31 0.39 0.29 --layout dna_rendering
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .segmenter import SegmentationShim


def generate_masks_for_dataset(
    input_dir: Path,
    output_dir: Path,
    backend: str = "threshold",
    model_type: str = "vit_h",
    layout: str = "dna_rendering",
    bg_color: tuple[float, float, float] = (0.31, 0.39, 0.29),
    threshold: float = 0.5,
    device: Optional_str := None,
) -> None:
    """Generate masks for all images in a dataset directory.

    Args:
        input_dir: Directory containing source images.
        output_dir: Directory to save generated masks.
        backend: Segmentation backend ("sam", "segformer", "threshold").
        model_type: Model type/name for the backend.
        layout: Dataset layout ("dna_rendering" or "thuman").
        bg_color: Background color for threshold backend.
        threshold: Binarization threshold.
        device: Inference device.
    """
    # Initialize segmenter
    shim_kwargs = {}
    if backend == "threshold":
        shim_kwargs["bg_color"] = bg_color
    elif backend == "sam":
        shim_kwargs["model_type"] = model_type
    elif backend == "segformer":
        shim_kwargs["model_name"] = model_type

    shim = SegmentationShim(backend=backend, device=device, **shim_kwargs)

    # Find all images based on layout
    if layout == "dna_rendering":
        # scenes/<scene>/images/<cam>/<frame>.{png,webp,jpg}
        image_files = []
        for ext in ("*.png", "*.webp", "*.jpg", "*.jpeg"):
            image_files.extend(input_dir.rglob(ext))

        for img_path in tqdm(image_files, desc="Generating masks"):
            # Compute output path: replace "scenes" with "masks",
            # "images" with "fmasks"
            rel_path = img_path.relative_to(input_dir)
            parts = list(rel_path.parts)
            if "images" in parts:
                idx = parts.index("images")
                parts[idx] = "fmasks"
            else:
                # Direct image, just change extension
                parts[-1] = os.path.splitext(parts[-1])[0] + ".png"

            # If input is scenes/<scene>/images/..., output is masks/<scene>/fmasks/...
            if parts[0] == "scenes":
                parts[0] = "masks"
            elif "scenes" not in parts and "images" not in parts:
                parts = ["masks"] + parts

            out_path = output_dir / Path(*parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                continue

            # Load and segment
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = shim.segment(image, threshold)

            # Save as uint8 PNG
            mask_uint8 = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(str(out_path))

    elif layout == "thuman":
        # img/<scene>_<cam>/2.jpg -> mask/<scene>_<cam>/2.png
        img_dir = input_dir / "img" if (input_dir / "img").is_dir() else input_dir
        out_base = output_dir / "mask" if (input_dir / "img").is_dir() else output_dir

        image_files = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
        for img_path in tqdm(image_files, desc="Generating masks"):
            rel_name = img_path.parent.name  # <scene>_<cam>
            stem = os.path.splitext(img_path.name)[0]  # e.g. "2"
            out_path = out_base / rel_name / f"{stem}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = shim.segment(image, threshold)

            mask_uint8 = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(str(out_path))

    else:
        # Generic: process all images in input_dir
        image_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            image_files.extend(input_dir.rglob(ext))

        for img_path in tqdm(image_files, desc="Generating masks"):
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists():
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            mask = shim.segment(image, threshold)

            mask_uint8 = (mask.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(str(out_path))

    print(f"Done. Masks saved to {output_dir}")


# Fix the type annotation issue
from typing import Optional
Optional_str = Optional[str]

generate_masks_for_dataset.__annotations__["device"] = Optional[str]


def main():
    parser = argparse.ArgumentParser(
        description="Generate human segmentation masks for PointSplat datasets."
    )
    parser.add_argument("--input", type=str, required=True, help="Input image directory.")
    parser.add_argument("--output", type=str, required=True, help="Output mask directory.")
    parser.add_argument(
        "--backend",
        choices=["sam", "segformer", "threshold"],
        default="threshold",
        help="Segmentation backend.",
    )
    parser.add_argument("--model-type", type=str, default="vit_h",
                        help="SAM model type or SegFormer model name.")
    parser.add_argument("--layout", choices=["dna_rendering", "thuman", "generic"],
                        default="generic", help="Dataset layout.")
    parser.add_argument("--bg-color", type=float, nargs=3, default=[0.31, 0.39, 0.29],
                        help="Background color for threshold backend.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold.")
    parser.add_argument("--device", type=str, default=None, help="Inference device.")
    args = parser.parse_args()

    generate_masks_for_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        backend=args.backend,
        model_type=args.model_type,
        layout=args.layout,
        bg_color=tuple(args.bg_color),
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
