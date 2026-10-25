"""Quantization module for PointSplat encoder acceleration.

This module provides post-training quantization (PTQ) for the PointSplat
encoder, reducing memory footprint and accelerating inference on
resource-constrained devices (e.g., Jetson).

The decoder (gsplat CUDA kernel) is NOT quantized — it always runs in fp32.
Only the encoder's nn.Linear and nn.LayerNorm layers are quantized.

Supported quantization methods:
    - INT8 dynamic quantization (PyTorch built-in)
    - INT8 static quantization with calibration
    - BF16 weight-only quantization (simplest, 2× memory reduction)
    - INT4 weight-only quantization (via bitsandbytes, 8× memory reduction)

Usage
-----
::

    from src.modules.quantization import quantize_encoder

    quantized_encoder = quantize_encoder(
        encoder,
        method="int8_dynamic",
    )
    # Use quantized_encoder normally for inference
"""

from .quantizer import (
    quantize_encoder,
    QuantizationConfig,
    calibrate_encoder,
    benchmark_encoder,
)

__all__ = [
    "quantize_encoder",
    "QuantizationConfig",
    "calibrate_encoder",
    "benchmark_encoder",
]
