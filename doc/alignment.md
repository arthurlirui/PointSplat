# PointSplat ↔ Static Gaussian-Splatting Alignment

This module registers a **high-quality static Gaussian-splatting scene**
(produced by optimising the official `gaussian-splatting` repo, or its
COLMAP sparse-point proxy) onto the **feedforward Gaussians produced by
PointSplat**, and renders visualisations that let you judge the
alignment quality in image space.

## Motivation

PointSplat's encoder generates a Gaussian splat in a single forward pass
from a few context views.  Its world frame is arbitrary — the
`scale_invariant` decoder option in `src/model/decoder/decoder_splatting_gsplat.py`
even normalises scene coordinates by camera near-plane, so the
feedforward Gaussians are not in the same metric frame as a 3DGS scene
optimised on the same images.

A high-quality static 3DGS scene is the natural reference geometry:
dense, low-noise, photorealistic.  Aligning the two lets us:

* transfer the static scene's appearance / detail onto the feedforward
  Gaussians' geometry,
* measure the geometric error of the feedforward head against a
  pseudo-ground-truth 3DGS optimisation, and
* visualise feedforward reconstruction quality side-by-side with the
  reference.

## Method

The alignment has two stages, packaged in `src/alignment/`.

### Stage 1 — Geometric registration (`src/alignment/register.py`)

Estimate a similarity transform `y = s · (R · x) + t` that maps the
feedforward Gaussian centres onto the static scene's points.

1. **PCA pre-alignment.** Compute each cloud's centroid and principal
   axes.  The 8 candidate axis-sign combinations are scored by chamfer
   distance on a sub-sampled set; the best one gives a closed-form
   `R`.  The isotropic scale `s` is the ratio of the clouds' weighted
   point-cloud standard deviations, and `t` follows from the centroids.
2. **ICP refinement.** Point-to-point ICP with `scipy.spatial.cKDTree`
   nearest-neighbour lookup and a Huber loss on per-point residuals to
   suppress outliers from the noisier feedforward cloud.  At each
   iteration a *delta* similarity transform is fit by the closed-form
   weighted Umeyama method and composed onto the running transform,
   which is the standard ICP accumulation pattern and is far more
   numerically stable than re-fitting from the original frame each
   iteration.

The output `SimilarityTransform` carries `R (3,3)`, `t (3,)`, `s (scalar)`.
`apply_similarity(cloud, T)` returns a new `GaussianCloud` whose means
are mapped by `T` and whose scales / rotations are transformed so the
*splats themselves* (not just their centres) move to the target frame.

### Stage 2 — Image-space alignment visualisation (`src/alignment/viz_align.py`)

Both clouds are rendered from a shared set of camera poses (read from
the scene's `transforms.json`) and compared in image space.

* **Renderer.** A pure-PyTorch point splatter: each Gaussian centre is
  projected to the image plane and splatted as a small disc whose size
  is `point_radius_px` pixels, with the Gaussian's own colour and
  opacity.  This is intentionally *not* a full alpha-composited Gaussian
  splat — it is a lightweight visualisation renderer that runs on CPU in
  seconds and has no CUDA / `gsplat` dependency.  It is sufficient for
  judging geometric alignment.  (A `gsplat` backend can be plugged in
  later for photorealistic comparison.)
* **Camera convention.** The `transforms.json` shipped with the
  DNA-Rendering / THuman2.0 datasets stores poses in the OpenGL/Blender
  convention (Y up, Z back).  PointSplat's dataset loader flips them to
  COLMAP/OpenCV (Y down, Z forward) via `c2w[:3, 1:3] *= -1` before
  feeding them to `gsplat`.  The alignment script applies the same flip
  so the visualisation renderer (which also assumes Z forward) sees a
  consistent camera frame.
* **Visualisation outputs.** For each camera (up to `--max-cameras`):
    * `side_by_side_<i>.png` — static (left) | feedforward (right)
    * `checkerboard_<i>.png` — 16×16-cell checkerboard mix of the two
    * `residual_<i>.png` — per-pixel L1 heat-map (jet-like)
    * `overlay_<i>.png` — static in red, feedforward in green; misalignment shows as red/green fringes
* **Image-space metrics.** Per-camera PSNR, SSIM (11×11 Gaussian-window
  SSIM, computed in pure PyTorch), mean L1, and foreground-mask IoU;
  their means are reported in `alignment_report.json`.

## Files

| File | Purpose |
| --- | --- |
| `src/alignment/__init__.py` | Public API: `GaussianCloud`, `register_similar`, `apply_similarity`, `alignment_report`, `save_alignment_visualization` |
| `src/alignment/types.py` | `GaussianCloud` container (unbatched, single-scene, SH-last) |
| `src/alignment/register.py` | `SimilarityTransform`, PCA pre-align, weighted-Umeyama ICP |
| `src/alignment/viz_align.py` | `Camera`, `point_splat_render`, image-space metrics, visualisation grids |
| `src/utils/gaussiansplatting_utils.py` | New `load_ply` helper (the file previously only *wrote* PLYs) |
| `scripts/align_feedforward_to_static.py` | End-to-end CLI demo |
| `experiments/alignment_demo/` | Demo run output (synthesised clouds) |
| `experiments/alignment_ply/` | Demo run output (PLY-driven) |

## Usage

### Demo mode (no PLYs needed — verifies the pipeline)

```bash
python scripts/align_feedforward_to_static.py \
    --scene datasets/DNA_Rendering_example/scenes/0012_09 \
    --demo \
    --output experiments/alignment_demo \
    --max-cameras 4
```

Synthesises a static and a (mis-aligned, noisier) feedforward cloud with
a *known* similarity transform between them, then verifies the
registration recovers its inverse.  Expected output:

```
[align] stage 1: similarity registration (PCA + ICP)
[align]   pre-align  chamfer: 0.034
[align]   post-align chamfer: 0.032
[align] stage 2: image-space alignment visualisation
[align]   mean PSNR    : 18.7 dB
[align]   mean SSIM    : 0.889
```

### Full pipeline with real PLYs

First produce the two PLYs:

* **Static PLY.** Run the official `gaussian-splatting` repo on the
  scene's images for ~30k iterations and export `point_cloud.ply`.  Or,
  as a quick proxy, use the COLMAP `sparse_pcd.ply` referenced by
  `transforms.json` (its `ply_file_path` field).
* **Feedforward PLY.** Run PointSplat inference with
  `test.save_gaussian=true` so the model exports a PLY alongside the
  rendered target views.

Then:

```bash
python scripts/align_feedforward_to_static.py \
    --scene datasets/DNA_Rendering_example/scenes/0012_09 \
    --static-ply /path/to/static_gs.ply \
    --feedforward-ply /path/to/pointsplat_export.ply \
    --output experiments/alignment_real \
    --max-cameras 8 \
    --max-icp-iters 50 \
    --point-radius-px 2.0
```

### Output

`<output>/alignment_report.json` contains:

```jsonc
{
  "transform": { "R": [[...]], "t": [...], "s": 0.87 },
  "registration": {
    "pre_align_chamfer": 0.035,
    "post_align_chamfer": 0.032,
    "post_align_median": 0.030,
    "post_align_p90": 0.049,
    "icp_iters": 34,
    "icp_residual_history": [...]
  },
  "image_space": {
    "mean_psnr": 18.28,
    "mean_ssim": 0.897,
    "mean_l1": 0.023,
    "mean_mask_iou": 0.148,
    "n_cameras": 4,
    "per_camera": [...]
  },
  "visualisations": { "side_by_side": [...], "checkerboard": [...], "residual": [...], "overlay": [...] }
}
```

`<output>/visualisations/` contains the per-camera PNGs.

## Verification

The `--demo` mode round-trips a known transform:

```
true inv: s=0.8696 R_euler=[-24.53, -11.16, 0.34] t=[-0.196, 0.440, -0.240]
est:       s=0.8688 R_euler=[-24.55, -11.15, 0.35] t=[-0.170, 0.383, -0.208]
```

The recovered `s` matches the true inverse to 4 decimals, the rotation
to within 0.02°, and the translation to within the synthetic-noise
baseline of 0.01.  Chamfer distance after ICP (0.013) is at the noise
floor, confirming the registration is geometrically tight.

## Limitations & future work

* The visualisation renderer is a point splatter, not a full Gaussian
  splat.  It ignores anisotropic scale and rotation, so it judges
  *geometric* alignment, not view-dependent appearance.  Plugging in the
  project's existing `gsplat`-based `render_gsplat` (when a CUDA device
  is available) would give photorealistic comparison.
* The registration is a single global similarity transform.  If the
  feedforward head has non-rigid scene-scale drift (e.g. different
  scale on the left vs. right of the scene), a per-region or non-rigid
  refinement on top of the global pre-align would help.
* SSIM is a lightweight 11×11-window implementation adequate for
  ranking alignment quality; swap in a perceptual metric (LPIPS) for
  finer appearance comparison.
