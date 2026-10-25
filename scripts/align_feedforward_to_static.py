"""Align a high-quality static Gaussian-splatting scene with the feedforward
Gaussians produced by PointSplat, and render visualisations of the alignment.

Two-stage pipeline
------------------
1. **Geometric registration** (:mod:`src.alignment.register`):
   estimate a similarity transform (R, t, s) that maps the feedforward
   Gaussians onto the static scene via PCA pre-alignment + ICP.
2. **Image-space alignment visualisation** (:mod:`src.alignment.viz_align`):
   render both clouds from shared camera poses and produce side-by-side,
   checkerboard, residual, and overlay PNGs, plus PSNR / SSIM / mask-IoU
   metrics.

Inputs
------
The script accepts two PLY files:

* ``--static-ply``: the high-quality static scene.  This is typically a
  3DGS PLY produced by training the official ``gaussian-splatting`` repo
  on the scene's images, OR the COLMAP sparse-point PLY referenced by
  ``transforms.json`` (``ply_file_path``).  Both are auto-detected by
  :func:`src.utils.gaussiansplatting_utils.load_ply`.
* ``--feedforward-ply``: a PointSplat Gaussian export.  Set
  ``test.save_gaussian=true`` in the PointSplat eval config to dump this
  PLY alongside the rendered images.

If either PLY is not provided, the script *synthesises* a demo cloud
from the scene's ``transforms.json`` so the pipeline can still be
exercised end-to-end (see ``--demo``).  This is useful for verifying the
alignment code without running a full 3DGS optimisation.

Shared camera poses are read from the scene's ``transforms.json`` (the
NeRF-style camera file already used by the PointSplat dataset loaders).

Usage
-----
# Full pipeline with real PLYs:
python scripts/align_feedforward_to_static.py \
    --scene datasets/DNA_Rendering_example/scenes/0012_09 \
    --static-ply /path/to/static_gs.ply \
    --feedforward-ply /path/to/pointsplat_export.ply \
    --output experiments/alignment_demo

# Demo mode (no PLYs needed):
python scripts/align_feedforward_to_static.py \
    --scene datasets/DNA_Rendering_example/scenes/0012_09 \
    --demo \
    --output experiments/alignment_demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running this script from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alignment import (
    GaussianCloud,
    SimilarityTransform,
    register_similar,
    apply_similarity,
    alignment_report,
    save_alignment_visualization,
)
from src.alignment.viz_align import Camera


# ---------------------------------------------------------------------------
# Scene / camera loading
# ---------------------------------------------------------------------------
def load_cameras_from_transforms(
    transforms_path: Path,
    image_shape: tuple[int, int] | None = None,
    max_cameras: int = 8,
    stride: int = 1,
) -> list[Camera]:
    """Load a list of :class:`Camera` from a NeRF-style ``transforms.json``.

    The format matches the DNA-Rendering / THuman2.0 datasets shipped with
    PointSplat: each frame has ``w``, ``h``, ``fl_x``, ``fl_y``, ``cx``,
    ``cy``, and a 4x4 ``transform_matrix`` (c2w, OPENCV convention).
    """
    with open(transforms_path) as f:
        data = json.load(f)
    frames = data["frames"][::stride]
    if len(frames) > max_cameras:
        # Evenly subsample.
        idx = np.linspace(0, len(frames) - 1, max_cameras).astype(int)
        frames = [frames[i] for i in idx]

    cameras = []
    for fr in frames:
        H = int(fr.get("h", image_shape[0] if image_shape else 1024))
        W = int(fr.get("w", image_shape[1] if image_shape else 1024))
        fx, fy = float(fr["fl_x"]), float(fr["fl_y"])
        cx, cy = float(fr["cx"]), float(fr["cy"])
        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        c2w = torch.tensor(fr["transform_matrix"], dtype=torch.float32)
        # The transforms.json poses are stored in the OpenGL/Blender convention
        # (Y up, Z back); PointSplat's dataset loader flips them to COLMAP /
        # OpenCV (Y down, Z forward) via ``c2w[:3, 1:3] *= -1`` so that the
        # downstream gsplat renderer (which expects Z-forward) works.  We
        # apply the same flip here so our visualisation renderer, which also
        # assumes Z-forward, sees a consistent camera frame.
        c2w = c2w.clone()
        c2w[:3, 1:3] *= -1
        cameras.append(Camera(extrinsics=c2w, intrinsics=K, image_shape=(H, W)))
    return cameras


# ---------------------------------------------------------------------------
# Demo cloud synthesis
# ---------------------------------------------------------------------------
def _synthesise_demo_clouds(
    cameras: list[Camera], n_static: int = 4000, n_ff: int = 12000, seed: int = 0
) -> tuple[GaussianCloud, GaussianCloud]:
    """Build a static + a (mis-aligned, noisier) feedforward cloud.

    The static cloud is a random point cloud roughly in front of the
    cameras.  The feedforward cloud is the static cloud *plus* a known
    random similarity transform (R, t, s) and additive Gaussian noise --
    this lets us verify that :func:`register_similar` recovers the
    inverse transform.
    """
    rng = np.random.default_rng(seed)
    # Static cloud: a small ball centred at the world origin, where the
    # scene's cameras converge (DNA-Rendering cameras form a ring around
    # the origin).  This guarantees the points are visible.
    pts = rng.normal(size=(n_static, 3)).astype(np.float32)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-6
    pts *= rng.uniform(0.2, 0.6, size=(n_static, 1)).astype(np.float32)
    # Centre on origin; the cameras look inward from a ~3.7m ring.
    pts += rng.normal(0.0, 0.05, size=pts.shape).astype(np.float32)

    colors = rng.uniform(0.2, 0.9, size=(n_static, 3)).astype(np.float32)
    static = GaussianCloud(
        means=torch.from_numpy(pts),
        scales=torch.full((n_static, 3), 0.01, dtype=torch.float32),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0] * n_static, dtype=torch.float32).view(n_static, 4),
        colors=torch.from_numpy(colors),
        opacities=torch.ones(n_static, dtype=torch.float32),
        source="static_gs_demo",
    )

    # Pick a known similarity transform.
    from scipy.spatial.transform import Rotation as R
    R_mat = R.from_euler("zyx", [25.0, 10.0, -5.0], degrees=True).as_matrix().astype(np.float32)
    t_vec = np.array([0.4, -0.3, 0.2], dtype=np.float32)
    s_val = 1.15

    ff_pts = s_val * (pts @ R_mat.T) + t_vec
    # Add noise to feedforward positions (PointSplat's feedforward Gaussians
    # are noisy relative to a clean 3DGS optimisation).
    ff_pts += rng.normal(0.0, 0.02, size=ff_pts.shape).astype(np.float32)

    ff_colors = rng.uniform(0.2, 0.9, size=(n_ff, 3)).astype(np.float32)
    # Resample ff points to n_ff to simulate PointSplat's denser output.
    ff_pts_resampled = np.repeat(ff_pts, n_ff // n_static + 1, axis=0)[:n_ff]
    feedforward = GaussianCloud(
        means=torch.from_numpy(ff_pts_resampled),
        scales=torch.full((n_ff, 3), 0.012, dtype=torch.float32),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0] * n_ff, dtype=torch.float32).view(n_ff, 4),
        colors=torch.from_numpy(ff_colors),
        opacities=torch.ones(n_ff, dtype=torch.float32),
        source="pointsplat_ff_demo",
    )

    return static, feedforward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True,
                    help="Path to a scene directory containing transforms.json")
    ap.add_argument("--static-ply", type=Path, default=None,
                    help="Path to a static high-quality GS .ply (3DGS export or COLMAP sparse pcd).")
    ap.add_argument("--feedforward-ply", type=Path, default=None,
                    help="Path to a PointSplat feedforward Gaussian export .ply.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output directory for the alignment report + visualisations.")
    ap.add_argument("--demo", action="store_true",
                    help="Synthesise demo clouds instead of loading PLYs (verifies the pipeline).")
    ap.add_argument("--max-cameras", type=int, default=8,
                    help="Maximum number of cameras to render from.")
    ap.add_argument("--max-icp-iters", type=int, default=50)
    ap.add_argument("--point-radius-px", type=float, default=2.0)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[align] scene: {args.scene}")
    print(f"[align] output: {args.output}")

    transforms_path = args.scene / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found at {transforms_path}")
    cameras = load_cameras_from_transforms(transforms_path, max_cameras=args.max_cameras)
    print(f"[align] loaded {len(cameras)} cameras from {transforms_path}")

    # Load clouds.
    if args.demo:
        print("[align] demo mode: synthesising static + feedforward clouds")
        static, feedforward = _synthesise_demo_clouds(cameras)
    else:
        if args.static_ply is None or args.feedforward_ply is None:
            raise ValueError(
                "Either --demo or both --static-ply and --feedforward-ply must be given."
            )
        print(f"[align] loading static PLY: {args.static_ply}")
        static = GaussianCloud.from_ply(args.static_ply, source="static_gs")
        print(f"[align]   -> {len(static)} Gaussians, centroid={static.means.mean(0).tolist()}")
        print(f"[align] loading feedforward PLY: {args.feedforward_ply}")
        feedforward = GaussianCloud.from_ply(args.feedforward_ply, source="pointsplat_ff")
        print(f"[align]   -> {len(feedforward)} Gaussians, centroid={feedforward.means.mean(0).tolist()}")

    # ---- Stage 1: register feedforward -> static --------------------------
    print("[align] stage 1: similarity registration (PCA + ICP)")
    transform, reg_info = register_similar(
        source=feedforward, target=static, max_icp_iters=args.max_icp_iters
    )
    print(f"[align]   transform: {transform}")
    print(f"[align]   pre-align  chamfer: {reg_info['pre_align_chamfer']:.6f}")
    print(f"[align]   post-align chamfer: {reg_info['post_align_chamfer']:.6f}")
    print(f"[align]   post-align median : {reg_info['post_align_median']:.6f}")
    print(f"[align]   post-align p90    : {reg_info['post_align_p90']:.6f}")
    print(f"[align]   ICP iters: {reg_info['icp_iters']}")

    # ---- Stage 2: image-space alignment visualisation ---------------------
    print("[align] stage 2: image-space alignment visualisation")
    ff_aligned = apply_similarity(feedforward, transform)

    report = alignment_report(
        static=static,
        feedforward=ff_aligned,
        cameras=cameras,
        point_radius_px=args.point_radius_px,
    )
    print(f"[align]   mean PSNR    : {report.mean_psnr:.3f} dB")
    print(f"[align]   mean SSIM    : {report.mean_ssim:.4f}")
    print(f"[align]   mean L1      : {report.mean_l1:.4f}")
    print(f"[align]   mean mask IoU: {report.mean_mask_iou:.4f}")

    paths = save_alignment_visualization(
        static=static,
        feedforward=ff_aligned,
        cameras=cameras,
        output_dir=args.output / "visualisations",
        point_radius_px=args.point_radius_px,
        max_cameras=args.max_cameras,
    )

    # ---- Persist report ---------------------------------------------------
    report_path = args.output / "alignment_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "transform": {
                "R": transform.R.tolist(),
                "t": transform.t.tolist(),
                "s": float(transform.s),
            },
            "registration": reg_info,
            "image_space": report.as_dict(),
            "visualisations": {k: v for k, v in paths.items()},
            "static_summary": static.summary(),
            "feedforward_summary": feedforward.summary(),
            "feedforward_aligned_summary": ff_aligned.summary(),
        }, f, indent=2)
    print(f"[align] wrote report: {report_path}")
    print(f"[align] wrote {sum(len(v) for v in paths.values())} visualisation PNGs to {args.output / 'visualisations'}")
    print("[align] done.")


if __name__ == "__main__":
    main()
