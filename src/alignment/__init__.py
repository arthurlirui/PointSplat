"""PointSplat ↔ static high-quality Gaussian-splatting alignment toolkit.

This package implements the registration and visualization-rendering
alignment between two scene representations:

* a **static high-quality Gaussian-splatting scene** (e.g. a 3DGS model
  optimized with the official ``gaussian-splatting`` repo, or its
  COLMAP sparse-point proxy), and
* the **feedforward Gaussians** produced by PointSplat's encoder in a
  single forward pass.

The pipeline is:

1. :mod:`src.alignment.register` -- estimate a similarity transform
   (rotation ``R``, translation ``t``, isotropic scale ``s``) that maps
   the feedforward Gaussians onto the static scene, using a robust
   centroid + PCA pre-alignment followed by point-to-point ICP.
2. :mod:`src.alignment.viz_align` -- render both scenes from shared
   camera poses with a pure-PyTorch point-splatter, and produce
   image-space alignment visualizations (side-by-side, checkerboard,
   per-pixel residual heat-map) plus PSNR/SSIM/LPIPS-free image metrics.

The :mod:`src.alignment.types` module defines the lightweight
``GaussianCloud`` container used throughout.
"""

from .types import GaussianCloud
from .register import SimilarityTransform, register_similar, apply_similarity
from .viz_align import (
    point_splat_render,
    alignment_report,
    save_alignment_visualization,
)

__all__ = [
    "GaussianCloud",
    "SimilarityTransform",
    "register_similar",
    "apply_similarity",
    "point_splat_render",
    "alignment_report",
    "save_alignment_visualization",
]
