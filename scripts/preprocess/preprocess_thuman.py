"""Preprocess raw THuman2.0 data into the PointSplat format.

The PointSplat THuman dataset loader (src/dataset/dataset_thuman.py) expects
data in the GPS-Gaussian render format:

    <output_root>/
    ├── train/
    │   ├── img/<scene>_<cam>/2.jpg
    │   ├── mask/<scene>_<cam>/2.png
    │   └── parm/<scene>_<cam>/{2_intrinsic.npy, 2_extrinsic.npy}
    └── val/
        └── ... (same layout)

Each ``<scene>_<cam>`` directory represents one camera view of one subject.
The image filename is hard-coded to ``2.jpg`` and the parameter files to
``2_intrinsic.npy`` / ``2_extrinsic.npy`` by the loader.

This script provides three modes:

  1. **validate** — check that an existing dataset has all required files.
  2. **render** — render multi-view images from THuman2.0 mesh data using
     pyrender / pytorch3d, producing the GPS-Gaussian format.
  3. **convert** — convert an existing GPS-Gaussian-style dataset that uses
     different filenames (e.g. ``0.jpg`` instead of ``2.jpg``) into the
     hard-coded ``2.*`` naming convention.

Usage examples
--------------
# Validate the downloaded example data (no writes)
python -m scripts.preprocess.preprocess_thuman \
    --validate-only --input datasets/THuman2_0_example

# Render THuman2.0 meshes to multi-view images
python -m scripts.preprocess.preprocess_thuman \
    --input /path/to/THuman2.0/meshes \
    --output datasets/THuman2_0_render \
    --mode render \
    --num-views 16 \
    --render-width 512 --render-height 512

# Convert a dataset that uses different filenames
python -m scripts.preprocess.preprocess_thuman \
    --input /path/to/existing_thuman \
    --output datasets/THuman2_0_render \
    --mode convert \
    --src-frame-id 0
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
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(root: Path) -> bool:
    """Check that a PointSplat-format THuman2.0 dataset is well-formed."""
    ok = True
    for split in ("val", "train"):
        split_dir = root / split
        if not split_dir.is_dir():
            if split == "train":
                continue  # train is optional for inference
            print(f"  [FAIL] Missing {split}/ directory")
            ok = False
            continue

        img_dir = split_dir / "img"
        mask_dir = split_dir / "mask"
        parm_dir = split_dir / "parm"

        for d in (img_dir, mask_dir, parm_dir):
            if not d.is_dir():
                print(f"  [FAIL] Missing {d} directory")
                ok = False
                continue

        cam_dirs = sorted(d for d in img_dir.iterdir() if d.is_dir())
        if not cam_dirs:
            print(f"  [WARN] No camera directories in {img_dir}")
            continue

        print(f"  [{split}] {len(cam_dirs)} camera directories")

        for cam_dir in tqdm(cam_dirs[:20], desc=f"Validating {split}"):  # sample first 20
            name = cam_dir.name
            # Check image
            img_path = img_dir / name / "2.jpg"
            if not img_path.exists():
                # Try other extensions
                found = False
                for ext in (".png", ".jpeg", ".webp"):
                    if (img_dir / name / f"2{ext}").exists():
                        found = True
                        break
                if not found:
                    print(f"  [FAIL] Missing image: {name}/2.jpg")
                    ok = False

            # Check mask
            mask_path = mask_dir / name / "2.png"
            if not mask_path.exists():
                print(f"  [FAIL] Missing mask: {name}/2.png")
                ok = False

            # Check parameters
            intr_path = parm_dir / name / "2_intrinsic.npy"
            extr_path = parm_dir / name / "2_extrinsic.npy"
            if not intr_path.exists():
                print(f"  [FAIL] Missing intrinsic: {name}/2_intrinsic.npy")
                ok = False
            if not extr_path.exists():
                print(f"  [FAIL] Missing extrinsic: {name}/2_extrinsic.npy")
                ok = False

            # Validate intrinsic shape
            if intr_path.exists():
                intr = np.load(intr_path)
                if intr.shape != (3, 3) and intr.size < 4:
                    print(f"  [FAIL] Bad intrinsic shape {intr.shape} for {name}")
                    ok = False

            # Validate extrinsic shape
            if extr_path.exists():
                extr = np.load(extr_path)
                if extr.shape not in [(4, 4), (3, 4)]:
                    print(f"  [FAIL] Bad extrinsic shape {extr.shape} for {name}")
                    ok = False

    if ok:
        print(f"\n  [PASS] Dataset at {root} is valid.")
    else:
        print(f"\n  [FAIL] Dataset at {root} has issues (see above).")
    return ok


# ---------------------------------------------------------------------------
# Render mode
# ---------------------------------------------------------------------------

def render_thuman_meshes(
    input_dir: Path,
    output_dir: Path,
    num_views: int = 16,
    render_width: int = 512,
    render_height: int = 512,
    elevation: float = 0.0,
    radius: float = 2.5,
    val_ratio: float = 0.2,
) -> None:
    """Render THuman2.0 meshes into multi-view images.

    This mode requires ``pyrender`` (or ``pytorch3d``) and ``trimesh``.

    Expected input structure::

        <input_dir>/
        ├── <subject>/          # e.g. 0000/
        │   ├── <subject>.obj   # mesh file
        │   ├── <subject>.mtl
        │   └── texture files
        └── ...

    For each subject, the script renders ``num_views`` evenly-spaced azimuthal
    views on a circle of radius ``radius`` at elevation ``elevation`` degrees,
    saves the rendered RGB image, foreground mask, and camera parameters.
    """
    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh is required for render mode. Install with:")
        print("  pip install trimesh pyrender")
        sys.exit(1)

    try:
        import pyrender
    except ImportError:
        print("ERROR: pyrender is required for render mode. Install with:")
        print("  pip install pyrender")
        sys.exit(1)

    # Find subject directories with mesh files
    subject_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and any(d.glob("*.obj")) or any(d.glob("*.ply"))
    )

    if not subject_dirs:
        print(f"No subject directories with meshes found in {input_dir}")
        return

    # Create output directories
    out_train = output_dir / "train"
    out_val = output_dir / "val"
    num_val = max(1, int(len(subject_dirs) * val_ratio))
    val_subjects = set(s.name for s in subject_dirs[:num_val])

    for subject_dir in tqdm(subject_dirs, desc="Rendering subjects"):
        subject_name = subject_dir.name
        split = "val" if subject_name in val_subjects else "train"
        split_dir = output_dir / split

        # Load mesh
        mesh_file = next(subject_dir.glob("*.obj"), None) or next(subject_dir.glob("*.ply"), None)
        if mesh_file is None:
            print(f"  Skipping {subject_name}: no mesh found")
            continue

        mesh = trimesh.load(str(mesh_file), process=False)

        # Center and scale the mesh
        mesh.vertices = mesh.vertices - mesh.centroid
        max_extent = np.abs(mesh.vertices).max()
        if max_extent > 0:
            mesh.vertices = mesh.vertices / max_extent

        # Create pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        if mesh.visual and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        else:
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        scene.add(pr_mesh)

        # Camera setup
        camera = pyrender.PerspectiveCamera(
            yfov=np.pi / 3.0,
            aspectRatio=render_width / render_height,
        )

        # Light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

        # Renderer
        try:
            renderer = pyrender.OffscreenRenderer(
                viewport_width=render_width,
                viewport_height=render_height,
            )
        except Exception as e:
            print(f"  Cannot create renderer (try OS Mesa / EGL): {e}")
            return

        # Render views
        for view_idx in range(num_views):
            azimuth = 2.0 * np.pi * view_idx / num_views
            elev_rad = np.deg2rad(elevation)

            # Camera position on a circle
            cam_x = radius * np.cos(azimuth) * np.cos(elev_rad)
            cam_y = radius * np.sin(azimuth) * np.cos(elev_rad)
            cam_z = radius * np.sin(elev_rad)

            # Look-at: camera to origin
            eye = np.array([cam_x, cam_y, cam_z])
            target = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])

            forward = target - eye
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up_vec = np.cross(right, forward)

            # Build c2w (OpenGL convention: -Z forward, Y up)
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up_vec
            c2w[:3, 2] = -forward
            c2w[:3, 3] = eye

            # Compute intrinsic matrix
            yfov = np.pi / 3.0
            fl_y = (render_height / 2.0) / np.tan(yfov / 2.0)
            fl_x = fl_y * (render_width / render_height)
            cx = render_width / 2.0
            cy = render_height / 2.0
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float64)

            # W2C extrinsic (world-to-camera)
            w2c = np.linalg.inv(c2w)

            # Add camera and light to scene
            cam_node = scene.add(camera, pose=c2w)
            light_node = scene.add(light, pose=c2w)

            # Render
            color, depth = renderer.render(scene)

            # Remove nodes for next iteration
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

            # Generate mask from alpha / depth
            if color.shape[2] == 4:
                mask = (color[:, :, 3] > 0).astype(np.uint8) * 255
                color = color[:, :, :3]
            else:
                mask = (depth > 0).astype(np.uint8) * 255

            # Save outputs
            frame_name = f"{subject_name}_{view_idx:03d}"
            img_out = split_dir / "img" / frame_name
            mask_out = split_dir / "mask" / frame_name
            parm_out = split_dir / "parm" / frame_name
            img_out.mkdir(parents=True, exist_ok=True)
            mask_out.mkdir(parents=True, exist_ok=True)
            parm_out.mkdir(parents=True, exist_ok=True)

            Image.fromarray(color).save(str(img_out / "2.jpg"), quality=95)
            Image.fromarray(mask).save(str(mask_out / "2.png"))

            np.save(str(parm_out / "2_intrinsic.npy"), K)
            np.save(str(parm_out / "2_extrinsic.npy"), w2c)

        renderer.delete()

    print(f"Done. {len(subject_dirs)} subjects rendered to {output_dir}")


# ---------------------------------------------------------------------------
# Convert mode
# ---------------------------------------------------------------------------

def convert_existing_dataset(
    input_dir: Path,
    output_dir: Path,
    src_frame_id: int = 0,
) -> None:
    """Convert a GPS-Gaussian-style dataset to the hard-coded ``2.*`` naming.

    If the source dataset uses ``0.jpg`` / ``0_intrinsic.npy`` etc., this
    copies/links them to ``2.jpg`` / ``2_intrinsic.npy``.
    """
    for split in ("train", "val"):
        in_split = input_dir / split
        out_split = output_dir / split
        if not in_split.is_dir():
            continue

        for subdir in ("img", "mask", "parm"):
            in_sub = in_split / subdir
            out_sub = out_split / subdir
            if not in_sub.is_dir():
                continue

            cam_dirs = sorted(d for d in in_sub.iterdir() if d.is_dir())
            for cam_dir in tqdm(cam_dirs, desc=f"Converting {split}/{subdir}"):
                out_cam = out_sub / cam_dir.name
                out_cam.mkdir(parents=True, exist_ok=True)

                if subdir == "img":
                    src = cam_dir / f"{src_frame_id}.jpg"
                    dst = out_cam / "2.jpg"
                    if src.exists() and not dst.exists():
                        _link_or_copy(src, dst)
                elif subdir == "mask":
                    src = cam_dir / f"{src_frame_id}.png"
                    dst = out_cam / "2.png"
                    if src.exists() and not dst.exists():
                        _link_or_copy(src, dst)
                elif subdir == "parm":
                    for suffix in ("intrinsic", "extrinsic"):
                        src = cam_dir / f"{src_frame_id}_{suffix}.npy"
                        dst = out_cam / f"2_{suffix}.npy"
                        if src.exists() and not dst.exists():
                            _link_or_copy(src, dst)

    print(f"Done. Converted dataset -> {output_dir}")


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
        description="Preprocess THuman2.0 data for PointSplat."
    )
    parser.add_argument("--input", type=str, required=True, help="Input data root.")
    parser.add_argument("--output", type=str, default=None, help="Output data root.")
    parser.add_argument(
        "--mode",
        choices=["render", "convert", "validate"],
        default="validate",
        help="Processing mode.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Shortcut for --mode validate.")
    parser.add_argument("--num-views", type=int, default=16, help="Number of views to render.")
    parser.add_argument("--render-width", type=int, default=512, help="Render image width.")
    parser.add_argument("--render-height", type=int, default=512, help="Render image height.")
    parser.add_argument("--elevation", type=float, default=0.0, help="Camera elevation in degrees.")
    parser.add_argument("--radius", type=float, default=2.5, help="Camera distance from origin.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--src-frame-id", type=int, default=0, help="Source frame ID for convert mode.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.validate_only:
        args.mode = "validate"

    if args.mode == "validate":
        ok = validate_dataset(input_dir)
        sys.exit(0 if ok else 1)
    elif args.mode == "render":
        if args.output is None:
            parser.error("--output is required for conversion modes")
        render_thuman_meshes(
            input_dir, Path(args.output),
            num_views=args.num_views,
            render_width=args.render_width,
            render_height=args.render_height,
            elevation=args.elevation,
            radius=args.radius,
            val_ratio=args.val_ratio,
        )
    elif args.mode == "convert":
        if args.output is None:
            parser.error("--output is required for conversion modes")
        convert_existing_dataset(input_dir, Path(args.output), args.src_frame_id)


if __name__ == "__main__":
    main()
