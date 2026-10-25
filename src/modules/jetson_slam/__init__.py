"""Jetson edge deployment and SLAM-style multi-frame fusion module.

This module provides:

1. **Edge deployment utilities**: Configures PointSplat for resource-constrained
   platforms (Jetson Orin/Xavier), including INT8 quantization, memory
   optimization, and TensorRT export helpers.

2. **Multi-frame Gaussian fusion**: Maintains a persistent 3D Gaussian model
   that is updated incrementally as new frames arrive. Fuses Gaussians from
   consecutive frames using spatial proximity and temporal coherence.

3. **SLAM integration**: Interfaces with camera tracking (ORB-SLAM/VINS) to
   provide camera poses, and maintains a consistent global 3D model.

Usage
-----
::

    from src.modules.jetson_slam import EdgeInferenceConfig, GaussianSLAM

    # Edge deployment
    config = EdgeInferenceConfig(
        quantization="int8_dynamic",
        enable_flash_attention=True,
        max_gaussians=100000,
    )
    engine = EdgeInferenceEngine(encoder, decoder, config)

    # SLAM mode
    slam = GaussianSLAM(encoder, decoder, config)
    for frame in video_stream:
        result = slam.process_frame(frame, camera_poses)
"""

from .edge_deploy import EdgeInferenceEngine, EdgeInferenceConfig
from .gaussian_fusion import GaussianFusion, FusionConfig
from .slam import GaussianSLAM, SLAMConfig

__all__ = [
    "EdgeInferenceEngine",
    "EdgeInferenceConfig",
    "GaussianFusion",
    "FusionConfig",
    "GaussianSLAM",
    "SLAMConfig",
]
