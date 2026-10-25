"""Lightweight container for a Gaussian cloud used by the alignment toolkit.

We deliberately *do not* reuse :class:`src.model.types.Gaussians` here.
That dataclass is tied to the training-time batched tensor layout
(``batch gaussian ...``) and to gsplat's SH-first harmonics convention.
Alignment operates on a single scene at a time, on CPU, and needs an
SH-last layout that is friendlier for visualization.  Keeping a separate
container avoids forcing the alignment code to know about batch
dimensions or training-only fields such as ``gaussians_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor


@dataclass
class GaussianCloud:
    """A single scene's Gaussians, unbatched, on any device.

    Attributes
    ----------
    means : (N, 3) float
        Gaussian centres in world space.
    scales : (N, 3) float
        Per-axis scales (already activated, i.e. ``exp`` of the PLY value).
    rotations : (N, 4) float
        Unit quaternions ``(w, x, y, z)``.
    colors : (N, 3) float
        View-independent RGB colour in ``[0, 1]``.  When the source PLY
        only stores SH DC coefficients we convert them to RGB on load.
    opacities : (N,) float
        Per-Gaussian opacity in ``[0, 1]`` (already sigmoid-activated).
    source : str
        Free-form label, e.g. ``"static_gs"`` or ``"pointsplat_ff"``.
    """

    means: Float[Tensor, "gaussian 3"]
    scales: Float[Tensor, "gaussian 3"]
    rotations: Float[Tensor, "gaussian 4"]
    colors: Float[Tensor, "gaussian 3"]
    opacities: Float[Tensor, " gaussian"]
    source: str = "unknown"

    # ---- construction helpers -------------------------------------------------
    @classmethod
    def from_ply(cls, ply_path: str | Path, source: str = "unknown") -> "GaussianCloud":
        """Load a Gaussian cloud from a 3DGS / COLMAP PLY file.

        Accepts both full 3DGS exports (with scales, rotations, SH) and
        plain sparse-point clouds (only ``x/y/z``).  See
        :func:`src.utils.gaussiansplatting_utils.load_ply` for details.
        """
        from ..utils.gaussiansplatting_utils import load_ply

        d = load_ply(Path(ply_path))
        # PLY harmonics are (N, 3, d_sh); take DC band -> (N, 3).
        colors = d["harmonics"][..., 0].clamp(0.0, 1.0)
        return cls(
            means=d["xyz"],
            scales=d["scales"],
            rotations=d["rotations"],
            colors=colors,
            opacities=d["opacities"],
            source=source,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Tensor], source: str = "unknown") -> "GaussianCloud":
        """Build a cloud from a :func:`load_ply`-style dict."""
        colors = d["harmonics"][..., 0].clamp(0.0, 1.0)
        return cls(
            means=d["xyz"],
            scales=d["scales"],
            rotations=d["rotations"],
            colors=colors,
            opacities=d["opacities"],
            source=source,
        )

    # ---- transforms -----------------------------------------------------------
    def to(self, device: torch.device | str) -> "GaussianCloud":
        return GaussianCloud(
            means=self.means.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            colors=self.colors.to(device),
            opacities=self.opacities.to(device),
            source=self.source,
        )

    def __len__(self) -> int:
        return int(self.means.shape[0])

    def select(self, mask: Tensor) -> "GaussianCloud":
        return GaussianCloud(
            means=self.means[mask],
            scales=self.scales[mask],
            rotations=self.rotations[mask],
            colors=self.colors[mask],
            opacities=self.opacities[mask],
            source=self.source,
        )

    def summary(self) -> dict[str, float | int | str]:
        return {
            "source": self.source,
            "n_gaussians": len(self),
            "means_min": self.means.min(0).values.tolist(),
            "means_max": self.means.max(0).values.tolist(),
            "means_centroid": self.means.mean(0).tolist(),
            "opacity_mean": float(self.opacities.mean().item()),
        }
