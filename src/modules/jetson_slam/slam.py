"""SLAM-style incremental 3D human reconstruction.

Integrates PointSplat with camera tracking and multi-frame Gaussian fusion
to build a persistent 3D human model from a video stream. The pipeline:

1. **Camera tracking** (external): ORB-SLAM3/VINS provides per-frame camera poses
2. **Segmentation**: Automatic human mask generation (from the segmentation module)
3. **Feed-forward prediction**: PointSplat encoder predicts per-frame Gaussians
4. **Multi-frame fusion**: Gaussians are fused into a global consistent model
5. **Rendering**: The global model is rendered from any viewpoint

The module supports both offline (batch) and online (streaming) modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn

from ...model.types import Gaussians
from ..segmentation import SegmentationShim
from .edge_deploy import EdgeInferenceEngine, EdgeInferenceConfig
from .gaussian_fusion import GaussianFusion, FusionConfig


@dataclass
class SLAMConfig:
    """Configuration for Gaussian SLAM.

    Attributes:
        edge_config: Edge inference configuration.
        fusion_config: Gaussian fusion configuration.
        segmentation_backend: Segmentation backend for auto-masking.
        enable_segmentation: If True, auto-generate masks; if False, use provided masks.
        warm_start: If True, use previous frame's latent as initialization.
        max_frames: Maximum number of frames to process (0 = unlimited).
        save_trajectory: If True, save camera trajectory.
    """
    edge_config: EdgeInferenceConfig = field(default_factory=EdgeInferenceConfig)
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    segmentation_backend: Literal["sam", "segformer", "threshold", "none"] = "threshold"
    enable_segmentation: bool = True
    warm_start: bool = True
    max_frames: int = 0
    save_trajectory: bool = True


class GaussianSLAM:
    """Gaussian SLAM for incremental 3D human reconstruction.

    Combines PointSplat feed-forward prediction with multi-frame Gaussian
    fusion and camera tracking for real-time human capture.

    Pipeline per frame:
        1. Receive multi-view images + camera poses
        2. (Optional) Segment human from background
        3. Run PointSplat encoder to predict Gaussians
        4. Transform Gaussians to world coordinates using camera poses
        5. Fuse with global Gaussian model
        6. Render global model from desired viewpoints

    Args:
        encoder: PointSplat encoder.
        decoder: PointSplat decoder.
        config: SLAM configuration.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: Optional[SLAMConfig] = None,
    ):
        self.config = config or SLAMConfig()

        # Initialize edge inference engine
        self.engine = EdgeInferenceEngine(
            encoder, decoder, self.config.edge_config
        )

        # Initialize Gaussian fusion
        self.fusion = GaussianFusion(
            self.config.fusion_config,
            device=self.config.edge_config.device,
        )

        # Initialize segmentation (optional)
        self.segmenter = None
        if self.config.enable_segmentation and self.config.segmentation_backend != "none":
            self.segmenter = SegmentationShim(
                backend=self.config.segmentation_backend,
                device=self.config.edge_config.device,
            )

        # State
        self.trajectory = []  # List of camera poses
        self.frame_count = 0
        self.total_timing = {
            "segmentation_ms": 0,
            "encoder_ms": 0,
            "decoder_ms": 0,
            "fusion_ms": 0,
            "total_ms": 0,
        }

    def process_frame(
        self,
        context: dict,
        target: Optional[dict] = None,
        camera_poses: Optional[torch.Tensor] = None,
    ) -> dict:
        """Process a single frame.

        Args:
            context: Context views dict with image, extrinsics, intrinsics.
                     If masks are not provided and segmentation is enabled,
                     masks will be auto-generated.
            target: Optional target views for rendering evaluation.
            camera_poses: Optional camera poses [V, 4, 4] for world transform.
                         If None, uses context extrinsics.

        Returns:
            Dict with:
                - "gaussians": Fused global Gaussians
                - "rendering": Rendered target views (if target provided)
                - "timing": Per-stage timing
                - "stats": Fusion statistics
        """
        t_start = time.perf_counter()
        timing = {}

        # Step 1: Segmentation (if enabled and masks not provided)
        if self.segmenter is not None and "mask" not in context:
            t0 = time.perf_counter()
            context = self.segmenter.apply_to_example(context)
            t1 = time.perf_counter()
            timing["segmentation_ms"] = (t1 - t0) * 1000
        else:
            timing["segmentation_ms"] = 0

        # Step 2: Feed-forward prediction
        t0 = time.perf_counter()
        inference_result = self.engine.infer(context, target or context)
        t1 = time.perf_counter()
        timing["encoder_ms"] = inference_result["timing"]["encoder_ms"]
        timing["decoder_ms"] = inference_result["timing"]["decoder_ms"]

        # Step 3: Multi-frame fusion
        t2 = time.perf_counter()

        # Determine world transform
        world_transform = None
        if camera_poses is not None:
            # Use the first camera pose as the world transform
            world_transform = camera_poses[0] if camera_poses.ndim == 3 else camera_poses

        fused_gaussians = self.fusion.fuse(
            inference_result["gaussians"],
            world_transform=world_transform,
        )

        t3 = time.perf_counter()
        timing["fusion_ms"] = (t3 - t2) * 1000

        # Save trajectory
        if self.config.save_trajectory and camera_poses is not None:
            self.trajectory.append(camera_poses.cpu().clone())

        timing["total_ms"] = (t3 - t_start) * 1000

        # Update running totals
        for k, v in timing.items():
            self.total_timing[k] += v

        self.frame_count += 1

        # Prepare result
        result = {
            "gaussians": fused_gaussians,
            "timing": timing,
            "stats": self.fusion.get_stats(),
        }

        if target is not None:
            result["rendering"] = inference_result["color"]

        return result

    def render_view(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Render the global Gaussian model from a novel viewpoint.

        Args:
            extrinsics: Camera extrinsics [4, 4] or [B, 4, 4].
            intrinsics: Camera intrinsics [3, 3] or [B, 3, 3].
            near: Near plane.
            far: Far plane.
            image_shape: (H, W) render resolution.

        Returns:
            Rendered image [B, 3, H, W].
        """
        if self.fusion.global_means is None:
            raise RuntimeError("No Gaussians in the global model. Process frames first.")

        from ...model.encoder import EncoderOutput

        # Get current fused Gaussians
        gaussians = self.fusion.fuse(Gaussians(
            means=torch.zeros(1, 0, 3, device=self.config.edge_config.device),
            covariances=None,
            harmonics=torch.zeros(1, 0, 3, device=self.config.edge_config.device),
            opacities=torch.zeros(1, 0, device=self.config.edge_config.device),
            scales=torch.zeros(1, 0, 3, device=self.config.edge_config.device),
            rotations=torch.zeros(1, 0, 4, device=self.config.edge_config.device),
        ))

        # Ensure batch dimension for extrinsics/intrinsics
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if near.ndim == 0:
            near = near.unsqueeze(0)
        if far.ndim == 0:
            far = far.unsqueeze(0)

        encoder_output = EncoderOutput(
            gaussians=gaussians,
            gaussians_mask=None,
            depth=None,
        )

        with torch.no_grad():
            decoder_output = self.engine.decoder(
                encoder_output,
                extrinsics,
                intrinsics,
                near,
                far,
                image_shape,
            )

        return decoder_output.color

    def get_stats(self) -> dict:
        """Get comprehensive SLAM statistics."""
        return {
            "frame_count": self.frame_count,
            "fusion_stats": self.fusion.get_stats(),
            "timing_stats": {
                k: {
                    "total_ms": v,
                    "avg_ms": v / max(self.frame_count, 1),
                }
                for k, v in self.total_timing.items()
            },
            "trajectory_length": len(self.trajectory),
            "memory_stats": self.engine.get_memory_stats(),
        }

    def reset(self) -> None:
        """Reset the SLAM state."""
        self.fusion.reset()
        self.trajectory = []
        self.frame_count = 0
        self.total_timing = {k: 0 for k in self.total_timing}
        self.engine.reset_memory_stats()
