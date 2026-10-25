"""Edge deployment utilities for Jetson platforms.

Configures PointSplat for resource-constrained devices by:
- Applying INT8 quantization to the encoder
- Limiting the number of Gaussians
- Reducing visual hull resolution
- Enabling gradient checkpointing
- Optionally exporting to TensorRT

Supported platforms:
    - Jetson AGX Orin (64GB/32GB)
    - Jetson Orin NX (16GB/8GB)
    - Jetson Xavier AGX (32GB/16GB)
    - Jetson Xavier NX (8GB)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn

from ..quantization import quantize_encoder, QuantizationConfig


@dataclass
class EdgeInferenceConfig:
    """Configuration for edge deployment.

    Attributes:
        quantization: Quantization method for the encoder.
        max_gaussians: Maximum number of Gaussians to keep (pruning).
        vhull_resolution: Visual hull grid resolution (lower = faster).
        enable_flash_attention: Use xformers flash attention.
        enable_gradient_checkpoint: Use gradient checkpointing (memory savings).
        downsample_ratio: Input image downsample ratio (1.0 = no downsample).
        render_resolution: Output render resolution (lower = faster).
        device: Target device.
        jetson_profile: Pre-configured profile for specific Jetson models.
    """
    quantization: Literal[
        "none", "bf16_weight_only", "int8_dynamic", "int4_bitsandbytes"
    ] = "bf16_weight_only"
    max_gaussians: int = 100000
    vhull_resolution: int = 64
    enable_flash_attention: bool = True
    enable_gradient_checkpoint: bool = True
    downsample_ratio: float = 1.0
    render_resolution: tuple[int, int] = (512, 512)
    device: str = "cuda"
    jetson_profile: Optional[Literal[
        "agx_orin_64", "agx_orin_32", "orin_nx_16", "orin_nx_8",
        "xavier_agx_32", "xavier_nx_8"
    ]] = None

    def apply_profile(self) -> None:
        """Apply a pre-configured Jetson profile."""
        if self.jetson_profile is None:
            return

        profiles = {
            "agx_orin_64": EdgeInferenceConfig(
                quantization="int8_dynamic",
                max_gaussians=150000,
                vhull_resolution=80,
                render_resolution=(512, 512),
            ),
            "agx_orin_32": EdgeInferenceConfig(
                quantization="int8_dynamic",
                max_gaussians=120000,
                vhull_resolution=72,
                render_resolution=(512, 512),
            ),
            "orin_nx_16": EdgeInferenceConfig(
                quantization="int8_dynamic",
                max_gaussians=80000,
                vhull_resolution=64,
                render_resolution=(384, 384),
            ),
            "orin_nx_8": EdgeInferenceConfig(
                quantization="int4_bitsandbytes",
                max_gaussians=50000,
                vhull_resolution=48,
                render_resolution=(256, 256),
            ),
            "xavier_agx_32": EdgeInferenceConfig(
                quantization="int8_dynamic",
                max_gaussians=100000,
                vhull_resolution=64,
                render_resolution=(384, 384),
            ),
            "xavier_nx_8": EdgeInferenceConfig(
                quantization="int4_bitsandbytes",
                max_gaussians=40000,
                vhull_resolution=48,
                render_resolution=(256, 256),
            ),
        }

        profile = profiles[self.jetson_profile]
        for k, v in profile.__dict__.items():
            if k != "jetson_profile":
                setattr(self, k, v)


class EdgeInferenceEngine:
    """Edge inference engine for PointSplat on Jetson platforms.

    Wraps the encoder and decoder with edge-specific optimizations:
    quantization, Gaussian pruning, resolution reduction, and memory management.

    Args:
        encoder: PointSplat encoder.
        decoder: PointSplat decoder.
        config: Edge inference configuration.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: Optional[EdgeInferenceConfig] = None,
    ):
        self.config = config or EdgeInferenceConfig()
        self.config.apply_profile()

        print(f"[Edge] Initializing edge inference engine")
        print(f"  Profile: {self.config.jetson_profile or 'custom'}")
        print(f"  Quantization: {self.config.quantization}")
        print(f"  Max Gaussians: {self.config.max_gaussians}")
        print(f"  VHull resolution: {self.config.vhull_resolution}")
        print(f"  Render resolution: {self.config.render_resolution}")

        # Apply quantization to encoder
        if self.config.quantization != "none":
            quant_config = QuantizationConfig(method=self.config.quantization)
            self.encoder = quantize_encoder(encoder, quant_config)
        else:
            self.encoder = encoder

        self.decoder = decoder

        # Move to device
        self.encoder = self.encoder.to(self.config.device)
        self.decoder = self.decoder.to(self.config.device)
        self.encoder.eval()
        self.decoder.eval()

        # Override vhull resolution if encoder supports it
        if hasattr(self.encoder, "vhull_res"):
            self.encoder.vhull_res = self.config.vhull_resolution
            print(f"  VHull resolution set to {self.encoder.vhull_res}")

        # Override max Gaussians
        if hasattr(self.encoder, "cfg") and hasattr(self.encoder.cfg, "gaussians_decoder"):
            self.encoder.cfg.gaussians_decoder.num_target = self.config.max_gaussians
            print(f"  Max Gaussians set to {self.config.max_gaussians}")

    @torch.no_grad()
    def infer(
        self,
        context: dict,
        target: dict,
    ) -> dict:
        """Run edge-optimized inference.

        Args:
            context: Context views dict with image, extrinsics, intrinsics, etc.
            target: Target views dict.

        Returns:
            Dict with "color" (rendered images), "gaussians", and "timing".
        """
        timing = {}

        # Downsample context if needed
        if self.config.downsample_ratio < 1.0:
            for key in ("image", "mask"):
                if key in context:
                    context[key] = torch.nn.functional.interpolate(
                        context[key].float(),
                        scale_factor=self.config.downsample_ratio,
                        mode="bilinear",
                        align_corners=False,
                    )

        # Encode
        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        encoder_output = self.encoder(context, global_step=0)

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timing["encoder_ms"] = (t1 - t0) * 1000

        # Prune Gaussians by opacity
        gaussians = encoder_output.gaussians
        if gaussians.opacities.shape[1] > self.config.max_gaussians:
            # Top-k by opacity
            k = self.config.max_gaussians
            _, indices = torch.topk(gaussians.opacities.squeeze(-1), k, dim=1)
            for attr in ("means", "scales", "rotations", "harmonics", "opacities"):
                tensor = getattr(gaussians, attr)
                setattr(gaussians, attr, torch.gather(
                    tensor, 1, indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
                ))

        # Decode
        t2 = time.perf_counter()

        from ...model.encoder import EncoderOutput
        pruned_output = EncoderOutput(
            gaussians=gaussians,
            gaussians_mask=encoder_output.gaussians_mask,
            depth=encoder_output.depth,
        )

        decoder_output = self.decoder(
            pruned_output,
            target["extrinsics"],
            target["intrinsics"],
            target["near"],
            target["far"],
            self.config.render_resolution,
        )

        if self.config.device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        timing["decoder_ms"] = (t3 - t2) * 1000
        timing["total_ms"] = (t3 - t0) * 1000

        return {
            "color": decoder_output.color,
            "gaussians": gaussians,
            "timing": timing,
        }

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        if self.config.device == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        return {}

    def reset_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        if self.config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def export_tensorrt(self, output_path: str) -> None:
        """Export the encoder to TensorRT for maximum inference speed.

        This is a placeholder for TensorRT export. On Jetson, this would
        use torch2trt or the TensorRT Python API to convert the encoder
        to a TensorRT engine.

        Args:
            output_path: Path to save the TensorRT engine.
        """
        try:
            import torch2trt
            print("[Edge] TensorRT export not fully implemented.")
            print("  Use torch2trt to convert the encoder:")
            print(f"    from torch2trt import torch2trt")
            print(f"    engine = torch2trt(encoder, [example_input])")
            print(f"    torch.save(engine.state_dict(), '{output_path}')")
        except ImportError:
            print("[Edge] torch2trt not installed. Install with:")
            print("  pip install torch2trt")
