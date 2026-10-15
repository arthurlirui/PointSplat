"""Preprocess raw DNA-Rendering / RenBody data into the PointSplat format.

The PointSplat dataset loader (src/dataset/dataset_renbody.py) expects data
in a NeRF-style layout that mirrors the processed Diffuman4D distribution:

    <output_root>/
    ├── validation_index.json          # list of scene names used for val/test
    ├── scenes/<scene_id>/
    │   ├── transforms.json            # camera parameters (NeRF-style)
    │   ├── sparse_pcd.ply             # (optional, read but unused by the loader)
    │   └── images/<camera_label>/<frame:06d>.{png,webp,jpg,jpeg}
    └── masks/<scene_id>/
        └── fmasks/<camera_label>/<frame:06d>.{png,webp,jpg,jpeg}

This script converts two kinds of raw DNA-Rendering inputs:

  1. **Raw DNA-Rendering dataset** (the original release with per-camera
     folders, calibration XML/JSON, and foreground masks).
  2. **COLMAP / NeRF-style transforms** that use a single ``transforms.json``
     but store images flat (no per-camera subfolders) and reference frame
     index 0 — this re-splits them into the per-camera, per-frame layout.

It can also act as a **validator** for the already-downloaded example data:
run ``python -m scripts.preprocess.preprocess_dna_rendering --validate-only
--input datasets/DNA_Rendering_example`` to check that all referenced
images/masks exist and the transforms.json schema is correct.

Usage examples
--------------
# Validate the downloaded example data (no writes)
python -m scripts.preprocess.preprocess_dna_rendering \
    --validate-only --input datasets/DNA_Rendering_example

# Convert a COLMAP-style NeRF export to the PointSplat layout
python -m scripts.preprocess.preprocess_dna_rendering \
    --input /path/to/raw_dna_rendering \
    --output datasets/DNA_Rendering_processed \
    --mode colmap

# Convert the raw DNA-Rendering release (per-camera folders + calibration)
python -m scripts.preprocess.preprocess_dna_rendering \
    --input /path/to/raw_dna_rendering \
    --output datasets/DNA_Rendering_processed \
    --mode raw \
    --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def make_transforms_entry(
    camera_label: str,
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    transform_matrix: np.ndarray,
    file_path: str,
) -> dict:
    """Build one frame entry for transforms.json (NeRF / Instant-NGP style)."""
    return {
        "camera_model": "OPENCV",
        "camera_label": camera_label,
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "transform_matrix": transform_matrix.tolist(),
        "file_path": file_path,
    }


def build_scene_transforms(
    camera_labels: list[str],
    intrinsics: dict[str, dict],
    extrinsics: dict[str, np.ndarray],
    w: int,
    h: int,
    num_frames: int,
) -> dict:
    """Build a transforms.json dict for one scene.

    Parameters
    ----------
    camera_labels : sorted list of camera label strings (e.g. ``["00", ..., "47"]``).
    intrinsics : ``{label: {"fl_x", "fl_y", "cx", "cy"}}``.
    extrinsics : ``{label: 4x4 c2w numpy array}`` (OpenGL/Blender convention;
        the loader converts to OpenCV internally).
    w, h : image width / height.
    num_frames : number of temporal frames per camera.
    """
    frames = []
    for label in camera_labels:
        # file_path uses frame 000000 as placeholder; the loader replaces it
        # with the actual frame index at runtime.
        file_path = f"images/{label}/000000.webp"
        entry = make_transforms_entry(
            camera_label=label,
            fl_x=intrinsics[label]["fl_x"],
            fl_y=intrinsics[label]["fl_y"],
            cx=intrinsics[label]["cx"],
            cy=intrinsics[label]["cy"],
            w=w,
            h=h,
            transform_matrix=extrinsics[label],
            file_path=file_path,
        )
        frames.append(entry)

    return {
        "ply_file_path": "sparse_pcd.ply",
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(root: Path) -> bool:
    """Check that a PointSplat-format DNA-Rendering dataset is well-formed."""
    ok = True
    val_index_path = root / "validation_index.json"
    if not val_index_path.is_file():
        print(f"  [FAIL] Missing validation_index.json at {val_index_path}")
        return False
    with open(val_index_path) as f:
        val_scenes = json.load(f)
    if not isinstance(val_scenes, list):
        print(f"  [FAIL] validation_index.json is not a list")
        return False
    print(f"  [OK] validation_index.json: {len(val_scenes)} scenes")

    scenes_dir = root / "scenes"
    masks_dir = root / "masks"
    if not scenes_dir.is_dir():
        print(f"  [FAIL] Missing scenes/ directory")
        return False

    for scene in tqdm(val_scenes, desc="Validating scenes"):
        scene_path = scenes_dir / scene
        mask_path = masks_dir / scene
        if not scene_path.is_dir():
            print(f"  [FAIL] Missing scene dir: {scene}")
            ok = False
            continue
        if not mask_path.is_dir():
            print(f"  [FAIL] Missing mask dir: {scene}")
            ok = False
            continue

        tj_path = scene_path / "transforms.json"
        if not tj_path.is_file():
            print(f"  [FAIL] Missing transforms.json in {scene}")
            ok = False
            continue
        with open(tj_path) as f:
            tj = json.load(f)
        if "frames" not in tj or "ply_file_path" not in tj:
            print(f"  [FAIL] transforms.json in {scene} missing keys")
            ok = False
            continue

        for frame in tj["frames"]:
            label = frame["camera_label"]
            # Check that at least one frame image exists
            img_dir = scene_path / "images" / label
            mask_frame_dir = mask_path / "fmasks" / label
            if not img_dir.is_dir():
                print(f"  [FAIL] Missing image dir {scene}/images/{label}")
                ok = False
                continue
            if not mask_frame_dir.is_dir():
                print(f"  [FAIL] Missing mask dir {scene}/fmasks/{label}")
                ok = False
                continue
            img_files = list(img_dir.iterdir())
            mask_files = list(mask_frame_dir.iterdir())
            if len(img_files) == 0:
                print(f"  [FAIL] No images in {scene}/images/{label}")
                ok = False
            if len(mask_files) == 0:
                print(f"  [FAIL] No masks in {scene}/fmasks/{label}")
                ok = False

    if ok:
        print(f"\n  [PASS] Dataset at {root} is valid.")
    else:
        print(f"\n  [FAIL] Dataset at {root} has issues (see above).")
    return ok


# ---------------------------------------------------------------------------
# COLMAP / NeRF-style conversion
# ---------------------------------------------------------------------------

def convert_colmap_style(input_dir: Path, output_dir: Path) -> None:
    """Convert a COLMAP/NeRF-style export to the PointSplat layout.

    Expected input structure::

        <input_dir>/
        ├── <scene>/
        │   ├── transforms.json       # standard NeRF transforms (all frames listed)
        │   ├── images/               # flat or per-camera images
        │   └── masks/                # foreground masks
        └── ...

    The input transforms.json may list every frame for every camera in a flat
    ``frames`` array with ``file_path`` entries like ``images/00_000030.webp``
    or ``images/00/000030.webp``.  This function re-organises them into the
    per-camera, per-frame layout expected by the loader.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(
        d for d in input_dir.iterdir() if d.is_dir() and (d / "transforms.json").is_file()
    )
    if not scene_dirs:
        print(f"No scene directories with transforms.json found in {input_dir}")
        return

    val_scenes = []
    for scene_dir in tqdm(scene_dirs, desc="Converting scenes"):
        scene_name = scene_dir.name
        with open(scene_dir / "transforms.json") as f:
            tj = json.load(f)

        frames = tj["frames"]
        out_scene = output_dir / "scenes" / scene_name
        out_mask = output_dir / "masks" / scene_name
        (out_scene / "images").mkdir(parents=True, exist_ok=True)
        (out_mask / "fmasks").mkdir(parents=True, exist_ok=True)

        # Group frames by camera label
        new_frames = []
        cam_labels_seen = set()
        for frame in frames:
            label = frame.get("camera_label", "")
            if not label:
                # Try to extract from file_path: "images/00/000030.webp" -> "00"
                fp = frame["file_path"]
                parts = fp.split("/")
                if len(parts) >= 2:
                    label = parts[-2]
                else:
                    # "images/00_000030.webp" -> "00"
                    fname = parts[-1]
                    label = fname.split("_")[0]

            # Create per-camera directory
            cam_img_dir = out_scene / "images" / label
            cam_mask_dir = out_mask / "fmasks" / label
            cam_img_dir.mkdir(parents=True, exist_ok=True)
            cam_mask_dir.mkdir(parents=True, exist_ok=True)

            # Copy image
            src_img = scene_dir / frame["file_path"]
            if not src_img.exists():
                print(f"  Warning: image not found {src_img}")
                continue
            frame_name = src_img.name  # e.g. 000030.webp
            dst_img = cam_img_dir / frame_name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Copy mask (same name but .png)
            mask_name = os.path.splitext(frame_name)[0] + ".png"
            src_mask = scene_dir / "masks" / label / mask_name
            if not src_mask.exists():
                # Try alternative mask paths
                src_mask = scene_dir / "masks" / f"{label}_{frame_name}"
            if src_mask.exists():
                dst_mask = cam_mask_dir / mask_name
                if not dst_mask.exists():
                    shutil.copy2(src_mask, dst_mask)

            if label not in cam_labels_seen:
                new_frame = frame.copy()
                new_frame["camera_label"] = label
                new_frame["file_path"] = f"images/{label}/{frame_name}"
                new_frames.append(new_frame)
                cam_labels_seen.add(label)

        # Write transforms.json with unique cameras
        new_tj = {
            "ply_file_path": tj.get("ply_file_path", "sparse_pcd.ply"),
            "frames": new_frames,
        }
        with open(out_scene / "transforms.json", "w") as f:
            json.dump(new_tj, f, indent=2)

        val_scenes.append(scene_name)

    # Write validation_index.json (all scenes as val for small datasets)
    with open(output_dir / "validation_index.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    print(f"Done. {len(val_scenes)} scenes written to {output_dir}")


# ---------------------------------------------------------------------------
# Raw DNA-Rendering conversion
# ---------------------------------------------------------------------------

def convert_raw_dna_rendering(
    input_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
) -> None:
    """Convert the raw DNA-Rendering dataset release.

    The raw DNA-Rendering release stores data per-camera with calibration
    files. This function reads the calibration, builds transforms.json, and
    symlinks/copies images and masks into the expected layout.

    Expected raw structure (approximate)::

        <input_dir>/
        ├── <scene>/
        │   ├── cam_00/
        │   │   ├── *.png          # images
        │   │   └── calibration    # intrinsics + extrinsics
        │   ├── cam_01/
        │   └── ...
        └── ...

    Since the raw DNA-Rendering release format may vary, this function
    attempts to auto-detect calibration files (JSON/XML/NPZ) in each camera
    directory. Adjust the ``_read_calibration`` helper for your specific
    release.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    all_scenes = []
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        scene_name = scene_dir.name
        out_scene = output_dir / "scenes" / scene_name
        out_mask = output_dir / "masks" / scene_name
        (out_scene / "images").mkdir(parents=True, exist_ok=True)
        (out_mask / "fmasks").mkdir(parents=True, exist_ok=True)

        # Find camera directories
        cam_dirs = sorted(
            d for d in scene_dir.iterdir()
            if d.is_dir() and d.name.startswith("cam")
        )
        if not cam_dirs:
            print(f"  Skipping {scene_name}: no cam_* directories found")
            continue

        frames = []
        for cam_dir in cam_dirs:
            label = cam_dir.name.replace("cam_", "")
            cam_img_dir = out_scene / "images" / label
            cam_mask_dir = out_mask / "fmasks" / label
            cam_img_dir.mkdir(parents=True, exist_ok=True)
            cam_mask_dir.mkdir(parents=True, exist_ok=True)

            # Read calibration
            intr, extr = _read_calibration(cam_dir)
            if intr is None:
                print(f"  Warning: no calibration in {cam_dir}, skipping")
                continue

            # Find images
            img_files = sorted(
                f for f in cam_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                and "mask" not in f.name.lower()
            )
            if not img_files:
                continue

            # Determine image dimensions
            with Image.open(img_files[0]) as im:
                w, h = im.size

            # Copy/link images and masks
            for img_file in img_files:
                frame_name = img_file.name
                dst_img = cam_img_dir / frame_name
                if not dst_img.exists():
                    _link_or_copy(img_file, dst_img)

                # Look for a mask with the same stem
                stem = os.path.splitext(frame_name)[0]
                mask_candidate = cam_dir / f"{stem}_mask.png"
                if not mask_candidate.exists():
                    mask_candidate = cam_dir / f"{stem}.png"
                if mask_candidate.exists() and "mask" in mask_candidate.name:
                    dst_mask = cam_mask_dir / f"{stem}.png"
                    if not dst_mask.exists():
                        _link_or_copy(mask_candidate, dst_mask)

            # Build transform entry (use first frame as representative)
            file_path = f"images/{label}/{img_files[0].name}"
            entry = make_transforms_entry(
                camera_label=label,
                fl_x=intr["fl_x"],
                fl_y=intr["fl_y"],
                cx=intr["cx"],
                cy=intr["cy"],
                w=w,
                h=h,
                transform_matrix=extr,
                file_path=file_path,
            )
            frames.append(entry)

        if not frames:
            continue

        tj = {"ply_file_path": "sparse_pcd.ply", "frames": frames}
        with open(out_scene / "transforms.json", "w") as f:
            json.dump(tj, f, indent=2)

        all_scenes.append(scene_name)

    # Split into val/train
    num_val = max(1, int(len(all_scenes) * val_ratio))
    val_scenes = all_scenes[:num_val]
    with open(output_dir / "validation_index.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    print(f"Done. {len(all_scenes)} scenes ({len(val_scenes)} val) -> {output_dir}")


def _read_calibration(cam_dir: Path) -> tuple[dict | None, np.ndarray]:
    """Read camera calibration from a camera directory.

    Tries JSON, XML, and NPZ formats. Returns (intrinsics, extrinsics_4x4).
    Override this function for your specific DNA-Rendering release format.
    """
    # Try JSON
    for name in ("calibration.json", "cam.json", "intrinsic.json"):
        p = cam_dir / name
        if p.exists():
            with open(p) as f:
                calib = json.load(f)
            intr = {
                "fl_x": calib.get("fl_x", calib.get("fx", 0)),
                "fl_y": calib.get("fl_y", calib.get("fy", 0)),
                "cx": calib.get("cx", 0),
                "cy": calib.get("cy", 0),
            }
            extr = np.array(calib.get("transform_matrix", calib.get("c2w", np.eye(4))))
            return intr, extr

    # Try NPZ
    for name in ("calibration.npz", "cam.npz"):
        p = cam_dir / name
        if p.exists():
            data = np.load(p, allow_pickle=True)
            intr = {
                "fl_x": float(data.get("fl_x", data.get("fx", 0))),
                "fl_y": float(data.get("fl_y", data.get("fy", 0))),
                "cx": float(data.get("cx", 0)),
                "cy": float(data.get("cy", 0)),
            }
            extr = data.get("transform_matrix", data.get("c2w", np.eye(4)))
            return intr, np.array(extr)

    return None, np.eye(4)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink if possible, otherwise copy."""
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess DNA-Rendering data for PointSplat."
    )
    parser.add_argument("--input", type=str, required=True, help="Input data root.")
    parser.add_argument("--output", type=str, default=None, help="Output data root.")
    parser.add_argument(
        "--mode",
        choices=["colmap", "raw", "validate"],
        default="validate",
        help="Conversion mode: 'colmap' for NeRF-style, 'raw' for DNA-Rendering release, "
             "'validate' to check an existing dataset.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (raw mode).")
    parser.add_argument("--validate-only", action="store_true", help="Shortcut for --mode validate.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.validate_only:
        args.mode = "validate"

    if args.mode == "validate":
        ok = validate_dataset(input_dir)
        sys.exit(0 if ok else 1)
    elif args.mode == "colmap":
        if args.output is None:
            parser.error("--output is required for conversion modes")
        convert_colmap_style(input_dir, Path(args.output))
    elif args.mode == "raw":
        if args.output is None:
            parser.error("--output is required for conversion modes")
        convert_raw_dna_rendering(input_dir, Path(args.output), args.val_ratio)


if __name__ == "__main__":
    main()
