"""Segmentation backends for human mask generation.

Provides a unified interface for different segmentation models. Each backend
implements :meth:`segment` which takes an RGB image tensor and returns a binary
foreground mask.

Backends:
    - SAMSegmenter: Uses Meta's Segment Anything Model (SAM)
    - SegFormerSegmenter: Uses SegFormer semantic segmentation

The :class:`SegmentationShim` wraps any backend and provides a convenient
interface for integration with PointSplat's data pipeline.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SegmenterBase(abc.ABC):
    """Abstract base class for segmentation backends."""

    @abc.abstractmethod
    def segment(
        self,
        image: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment a single image and return a binary foreground mask.

        Args:
            image: RGB image, either [3, H, W] torch.Tensor in [0,1] or
                   [H, W, 3] uint8 numpy array.
            threshold: Binarization threshold for soft masks.

        Returns:
            Binary mask tensor [1, H, W] float in {0.0, 1.0}.
        """
        ...

    def segment_batch(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment a batch of images.

        Args:
            images: [B, 3, H, W] torch.Tensor in [0, 1].
            threshold: Binarization threshold.

        Returns:
            [B, 1, H, W] binary mask tensor.
        """
        masks = []
        for i in range(images.shape[0]):
            masks.append(self.segment(images[i], threshold))
        return torch.stack(masks, dim=0)

    def _to_numpy(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        """Convert input to [H, W, 3] uint8 numpy array."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = (image.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
        return image

    def _select_person_mask(
        self,
        masks: list[np.ndarray],
        image: np.ndarray,
        strategy: str = "largest",
    ) -> np.ndarray:
        """Select the person mask from multiple candidate masks.

        Args:
            masks: List of binary masks [H, W].
            image: Original image [H, W, 3].
            strategy: "largest" (largest area), "center" (contains image center).

        Returns:
            Selected binary mask [H, W].
        """
        if len(masks) == 0:
            return np.zeros_like(image[..., 0], dtype=np.uint8)
        if len(masks) == 1:
            return masks[0]

        if strategy == "center":
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            best_dist = float("inf")
            best_mask = masks[0]
            for m in masks:
                ys, xs = np.where(m > 0)
                if len(ys) == 0:
                    continue
                cx, cy = xs.mean(), ys.mean()
                dist = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_mask = m
            return best_mask
        else:  # largest
            best_area = 0
            best_mask = masks[0]
            for m in masks:
                area = m.sum()
                if area > best_area:
                    best_area = area
                    best_mask = m
            return best_mask


class SAMSegmenter(SegmenterBase):
    """Segment Anything Model (SAM) backend.

    Uses SAM to generate human foreground masks. Supports automatic mode
    (generates all masks, selects the person) or point-prompt mode (prompts
    the image center).

    Args:
        model_type: SAM model type ("vit_h", "vit_l", "vit_b", "vit_tiny").
        checkpoint_path: Path to SAM checkpoint. If None, downloads from hub.
        device: Device to run inference on.
        automatic: If True, use automatic mask generation; if False, use
                   center-point prompt.
        points_per_side: Grid density for automatic mode.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        automatic: bool = False,
        points_per_side: int = 32,
    ):
        self.model_type = model_type
        self.automatic = automatic
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment_anything is required for SAM backend. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint(model_type)

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(self.device)
        self.sam = sam
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side
        )

    def _get_default_checkpoint(self, model_type: str) -> str:
        """Get default checkpoint path, downloading if necessary."""
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        if model_type not in urls:
            raise ValueError(f"Unknown model type: {model_type}")

        cache_dir = Path.home() / ".cache" / "sam"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = urls[model_type].split("/")[-1]
        ckpt_path = cache_dir / fname

        if not ckpt_path.exists():
            import urllib.request
            print(f"Downloading SAM {model_type} checkpoint to {ckpt_path}...")
            urllib.request.urlretrieve(urls[model_type], str(ckpt_path))

        return str(ckpt_path)

    def segment(
        self,
        image: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment using SAM."""
        img_np = self._to_numpy(image)
        h, w = img_np.shape[:2]

        if self.automatic:
            # Automatic mode: generate all masks, select person
            masks_data = self.mask_generator.generate(img_np)
            masks = [m["segmentation"].astype(np.uint8) * 255 for m in masks_data]
            mask = self._select_person_mask(masks, img_np, strategy="largest")
        else:
            # Point-prompt mode: prompt at image center
            self.predictor.set_image(img_np)
            center_point = np.array([[w // 2, h // 2]])
            masks, scores, _ = self.predictor.predict(
                point_coords=center_point,
                point_labels=np.array([1]),
                multimask_output=True,
            )
            # Select the mask with highest score
            best_idx = scores.argmax()
            mask = (masks[best_idx].astype(np.uint8)) * 255

        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        return mask_tensor


class SegFormerSegmenter(SegmenterBase):
    """SegFormer semantic segmentation backend.

    Uses SegFormer (via HuggingFace transformers) to generate person-class
    masks from the ADE20K or Cityscapes pretrained model.

    Args:
        model_name: HuggingFace model name (e.g. "nvidia/segformer-b0-finetuned-ade-512-512").
        person_class_id: Class ID for "person" in the model's label set.
        device: Device to run inference on.
    """

    # ADE20K person class ID
    ADE20K_PERSON_ID = 12

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        person_class_id: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.person_class_id = person_class_id or self.ADE20K_PERSON_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for SegFormer backend. "
                "Install with: pip install transformers"
            )

        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def segment(
        self,
        image: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment using SegFormer."""
        img_np = self._to_numpy(image)
        h, w = img_np.shape[:2]

        inputs = self.processor(images=img_np, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_classes, H', W']

        # Upsample to original resolution
        logits = F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )

        # Extract person class probability
        person_prob = torch.softmax(logits, dim=1)[0, self.person_class_id]
        person_prob = person_prob.cpu()

        # Binarize
        mask = (person_prob > threshold).float().unsqueeze(0)
        return mask


class SimpleThresholdSegmenter(SegmenterBase):
    """Simple background subtraction segmenter (fallback, no dependencies).

    Uses HSV color space thresholding to separate the foreground subject
    from a known background color. Works best with green-screen or solid
    background setups (like DNA-Rendering).

    Args:
        bg_color: Background RGB color [r, g, b] in [0, 1].
        tolerance: Color distance tolerance.
    """

    def __init__(
        self,
        bg_color: tuple[float, float, float] = (0.31, 0.39, 0.29),
        tolerance: float = 0.15,
    ):
        self.bg_color = torch.tensor(bg_color).reshape(3, 1, 1)
        self.tolerance = tolerance

    def segment(
        self,
        image: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment using color distance from background."""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        if image.ndim == 2:
            image = image.unsqueeze(0).expand(3, -1, -1)

        # Compute color distance
        dist = (image - self.bg_color.to(image.device)).abs().mean(dim=0, keepdim=True)
        mask = (dist > self.tolerance).float()
        return mask


class SegmentationShim:
    """High-level segmentation shim for PointSplat data pipeline.

    Wraps a segmentation backend and provides a convenient interface for
    generating masks on-the-fly or as a preprocessing step.

    Args:
        backend: "sam", "segformer", or "threshold".
        model_type: Model-specific configuration.
        device: Inference device.
        cache_dir: Directory for caching generated masks. If None, no caching.
        **kwargs: Backend-specific arguments.

    Example::

        shim = SegmentationShim(backend="threshold")
        mask = shim.segment(image)  # [3, H, W] -> [1, H, W]

        # Batch mode
        masks = shim.segment_batch(images)  # [B, 3, H, W] -> [B, 1, H, W]
    """

    def __init__(
        self,
        backend: Literal["sam", "segformer", "threshold"] = "threshold",
        model_type: str = "vit_h",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.backend_name = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if backend == "sam":
            self.segmenter = SAMSegmenter(
                model_type=model_type,
                device=device,
                **kwargs,
            )
        elif backend == "segformer":
            self.segmenter = SegFormerSegmenter(
                model_name=model_type,
                device=device,
                **kwargs,
            )
        elif backend == "threshold":
            self.segmenter = SimpleThresholdSegmenter(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def segment(
        self,
        image: torch.Tensor | np.ndarray,
        threshold: float = 0.5,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Segment a single image.

        Args:
            image: RGB image [3, H, W] or [H, W, 3].
            threshold: Binarization threshold.
            cache_key: If provided and cache_dir is set, cache/load the mask.

        Returns:
            Binary mask [1, H, W] float in {0, 1}.
        """
        if cache_key and self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                return torch.load(cache_path)

        mask = self.segmenter.segment(image, threshold)

        if cache_key and self.cache_dir:
            torch.save(mask, cache_path)

        return mask

    def segment_batch(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Segment a batch of images.

        Args:
            images: [B, 3, H, W] in [0, 1].
            threshold: Binarization threshold.

        Returns:
            [B, 1, H, W] binary masks.
        """
        return self.segmenter.segment_batch(images, threshold)

    def apply_to_example(
        self,
        example: dict,
        threshold: float = 0.5,
    ) -> dict:
        """Apply segmentation to a PointSplat data example dict.

        Replaces context and target masks with automatically generated ones.

        Args:
            example: PointSplat data example with "context" and "target" keys.
            threshold: Binarization threshold.

        Returns:
            Modified example with updated masks.
        """
        for stage in ("context", "target"):
            if stage not in example:
                continue
            images = example[stage]["image"]  # [V, 3, H, W]
            masks = self.segment_batch(images, threshold)
            example[stage]["mask"] = masks

        return example
