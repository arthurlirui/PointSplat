"""PointSplat extension modules.

This package contains four research direction modules that extend the base
PointSplat algorithm:

1. ``segmentation`` — Automatic human segmentation (SAM/SegFormer) to replace
   pre-computed masks, enabling in-the-wild inference.
2. ``iterative_optim`` — Per-scene iterative optimization that refines
   feed-forward predicted Gaussians for higher quality.
3. ``quantization`` — INT8 post-training quantization of the encoder for
   accelerated inference on resource-constrained devices.
4. ``jetson_slam`` — Edge deployment and incremental SLAM-style multi-frame
   Gaussian fusion for real-time human capture.
"""

__version__ = "1.0.0"
