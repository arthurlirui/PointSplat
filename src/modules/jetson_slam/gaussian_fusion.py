"""Multi-frame Gaussian fusion for temporal consistency.

Maintains a persistent 3D Gaussian model that is updated as new frames
arrive. Gaussians from consecutive frames are fused using:
- Spatial proximity matching (nearest neighbor in 3D)
- Attribute blending (weighted average of overlapping Gaussians)
- Opacity accumulation (confidence-weighted)
- Redundancy elimination (merge nearby Gaussians)

This produces a temporally consistent 3D representation that avoids the
flickering artifacts of independent per-frame prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ...model.types import Gaussians


@dataclass
class FusionConfig:
    """Configuration for multi-frame Gaussian fusion.

    Attributes:
        merge_distance: Maximum distance for merging Gaussians (in scene units).
        opacity_threshold: Gaussians below this opacity are pruned.
        max_gaussians: Maximum Gaussians in the fused model.
        temporal_weight: Weight for temporal blending (0=replace, 1=keep old).
        confidence_decay: Decay factor for frame confidence over time.
    """
    merge_distance: float = 0.02
    opacity_threshold: float = 0.01
    max_gaussians: int = 200000
    temporal_weight: float = 0.7
    confidence_decay: float = 0.95


class GaussianFusion:
    """Multi-frame Gaussian fusion for temporal consistency.

    Maintains a global set of Gaussians that is incrementally updated
    as new per-frame predictions arrive. Supports:

    - **Accumulation**: New Gaussians are added to the global model
    - **Merging**: Nearby Gaussians are merged with weighted averaging
    - **Pruning**: Low-opacity Gaussians are removed
    - **Confidence tracking**: Each Gaussian has a confidence score
      that decays over time

    Args:
        config: Fusion configuration.
        device: Device for tensor operations.
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or FusionConfig()
        self.device = device

        # Global Gaussian model state
        self.global_means: Optional[Tensor] = None
        self.global_scales: Optional[Tensor] = None
        self.global_rotations: Optional[Tensor] = None
        self.global_opacities: Optional[Tensor] = None
        self.global_colors: Optional[Tensor] = None
        self.global_confidence: Optional[Tensor] = None  # [N, 1]
        self.frame_count: int = 0

    def fuse(
        self,
        new_gaussians: Gaussians,
        world_transform: Optional[Tensor] = None,
    ) -> Gaussians:
        """Fuse new per-frame Gaussians into the global model.

        Args:
            new_gaussians: Gaussians from the current frame's prediction.
            world_transform: Optional 4x4 transform to map frame-local
                            coordinates to world coordinates (for SLAM).

        Returns:
            Fused global Gaussians.
        """
        self.frame_count += 1
        cfg = self.config

        # Transform new Gaussians to world space if needed
        means = new_gaussians.means.clone()
        if world_transform is not None:
            # Apply rigid transform to means
            R = world_transform[:3, :3]
            t = world_transform[:3, 3]
            means = means @ R.T + t

        scales = new_gaussians.scales.clone()
        rotations = new_gaussians.rotations.clone()
        opacities = new_gaussians.opacities.clone()
        colors = new_gaussians.harmonics.clone()

        B = means.shape[0]
        assert B == 1, "Fusion operates on single-batch"

        # Squeeze batch dimension
        means = means[0]          # [N, 3]
        scales = scales[0]        # [N, 3]
        rotations = rotations[0]  # [N, 4]
        opacities = opacities[0]  # [N] or [N, 1]
        if opacities.ndim == 1:
            opacities = opacities.unsqueeze(-1)
        colors = colors[0]        # [N, 3] or [N, C, ...]

        new_confidence = torch.ones_like(opacities)

        if self.global_means is None:
            # First frame: initialize
            self.global_means = means
            self.global_scales = scales
            self.global_rotations = rotations
            self.global_opacities = opacities
            self.global_colors = colors
            self.global_confidence = new_confidence
        else:
            # Subsequent frames: fuse
            # 1. Decay confidence of existing Gaussians
            self.global_confidence *= cfg.confidence_decay

            # 2. Find matches between new and existing Gaussians
            # Use approximate nearest neighbor (chunk-based for memory)
            merge_dist_sq = cfg.merge_distance ** 2

            # Compute pairwise distances (chunked for large N)
            chunk_size = 4096
            matched_new = torch.zeros(means.shape[0], dtype=torch.bool, device=self.device)
            match_indices = torch.full(
                (means.shape[0],), -1, dtype=torch.long, device=self.device
            )

            for start in range(0, means.shape[0], chunk_size):
                end = min(start + chunk_size, means.shape[0])
                chunk = means[start:end]  # [chunk, 3]

                # Distance to all existing Gaussians
                dist_sq = torch.cdist(chunk, self.global_means)  # [chunk, N_existing]

                # Find nearest existing Gaussian within merge distance
                min_dist, min_idx = dist_sq.min(dim=1)
                close_enough = min_dist < merge_dist_sq

                matched_new[start:end] = close_enough
                match_indices[start:end] = torch.where(
                    close_enough, min_idx, torch.full_like(min_idx, -1)
                )

            # 3. Merge matched Gaussians
            matched_idx = match_indices[matched_new]  # indices into global
            new_matched_idx = torch.where(matched_new)[0]

            if len(matched_idx) > 0:
                # Temporal blend
                w_new = 1.0 - cfg.temporal_weight
                w_old = cfg.temporal_weight

                # Blend means
                self.global_means[matched_idx] = (
                    w_old * self.global_means[matched_idx] +
                    w_new * means[new_matched_idx]
                )

                # Blend opacities (take max)
                self.global_opacities[matched_idx] = torch.maximum(
                    self.global_opacities[matched_idx],
                    opacities[new_matched_idx],
                )

                # Blend colors
                self.global_colors[matched_idx] = (
                    w_old * self.global_colors[matched_idx] +
                    w_new * colors[new_matched_idx]
                )

                # Update confidence
                self.global_confidence[matched_idx] += new_confidence[new_matched_idx]

            # 4. Add unmatched new Gaussians
            unmatched_new_idx = torch.where(~matched_new)[0]
            if len(unmatched_new_idx) > 0:
                self.global_means = torch.cat([
                    self.global_means,
                    means[unmatched_new_idx]
                ], dim=0)
                self.global_scales = torch.cat([
                    self.global_scales,
                    scales[unmatched_new_idx]
                ], dim=0)
                self.global_rotations = torch.cat([
                    self.global_rotations,
                    rotations[unmatched_new_idx]
                ], dim=0)
                self.global_opacities = torch.cat([
                    self.global_opacities,
                    opacities[unmatched_new_idx]
                ], dim=0)
                self.global_colors = torch.cat([
                    self.global_colors,
                    colors[unmatched_new_idx]
                ], dim=0)
                self.global_confidence = torch.cat([
                    self.global_confidence,
                    new_confidence[unmatched_new_idx]
                ], dim=0)

            # 5. Prune low-confidence/opacity Gaussians
            keep = (
                (self.global_opacities.squeeze(-1) > cfg.opacity_threshold) &
                (self.global_confidence.squeeze(-1) > 0.01)
            )
            self.global_means = self.global_means[keep]
            self.global_scales = self.global_scales[keep]
            self.global_rotations = self.global_rotations[keep]
            self.global_opacities = self.global_opacities[keep]
            self.global_colors = self.global_colors[keep]
            self.global_confidence = self.global_confidence[keep]

            # 6. Limit total Gaussians
            if self.global_means.shape[0] > cfg.max_gaussians:
                # Keep top-k by confidence × opacity
                scores = (
                    self.global_confidence.squeeze(-1) *
                    self.global_opacities.squeeze(-1)
                )
                _, top_idx = torch.topk(scores, cfg.max_gaussians)
                self.global_means = self.global_means[top_idx]
                self.global_scales = self.global_scales[top_idx]
                self.global_rotations = self.global_rotations[top_idx]
                self.global_opacities = self.global_opacities[top_idx]
                self.global_colors = self.global_colors[top_idx]
                self.global_confidence = self.global_confidence[top_idx]

        # Return as Gaussians dataclass
        return Gaussians(
            means=self.global_means.unsqueeze(0),
            covariances=None,
            harmonics=self.global_colors.unsqueeze(0),
            opacities=self.global_opacities.unsqueeze(0),
            scales=self.global_scales.unsqueeze(0),
            rotations=self.global_rotations.unsqueeze(0),
        )

    def get_stats(self) -> dict:
        """Get current fusion statistics."""
        if self.global_means is None:
            return {"frame_count": 0, "num_gaussians": 0}

        return {
            "frame_count": self.frame_count,
            "num_gaussians": self.global_means.shape[0],
            "avg_confidence": self.global_confidence.mean().item(),
            "avg_opacity": self.global_opacities.mean().item(),
        }

    def reset(self) -> None:
        """Reset the global Gaussian model."""
        self.global_means = None
        self.global_scales = None
        self.global_rotations = None
        self.global_opacities = None
        self.global_colors = None
        self.global_confidence = None
        self.frame_count = 0
