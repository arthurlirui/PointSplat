"""Automatic human segmentation module for PointSplat.

This module provides segmentation models that generate human foreground masks
from input images, replacing the need for pre-computed mask files. It supports
two backends:

- **SAM** (Segment Anything Model):通用分割模型，通过点提示或自动模式生成 mask
- **SegFormer**: 语义分割模型，直接输出 person 类别的 mask

The module integrates with PointSplat's data pipeline as a drop-in replacement
for on-disk masks, working both as an offline preprocessing tool and as an
online data shim.

Usage
-----
Offline (preprocess)::

    python -m src.modules.segmentation.generate_masks \
        --input datasets/my_images \
        --output datasets/my_masks \
        --backend sam \
        --model-type vit_h

Online (data shim)::

    from src.modules.segmentation import SegmentationShim
    shim = SegmentationShim(backend="sam", model_type="vit_h")
    mask = shim.segment(image_tensor)  # [3, H, W] -> [1, H, W]
"""

from .segmenter import SegmentationShim, SegmenterBase, SAMSegmenter, SegFormerSegmenter
from .generate_masks import generate_masks_for_dataset

__all__ = [
    "SegmentationShim",
    "SegmenterBase",
    "SAMSegmenter",
    "SegFormerSegmenter",
    "generate_masks_for_dataset",
]
