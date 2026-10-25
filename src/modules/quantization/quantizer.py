"""Encoder quantization implementations.

Provides multiple quantization strategies for the PointSplat encoder:

1. **INT8 dynamic quantization**: PyTorch's built-in dynamic quantization.
   Weights quantized to INT8, activations quantized dynamically at runtime.
   No calibration data needed. Works on CPU and GPU.

2. **INT8 static quantization**: Weights and activations both quantized to
   INT8. Requires calibration with representative data to determine
   activation ranges. Better performance than dynamic but needs calibration.

3. **BF16 weight-only**: Simply casts all Linear weights to bfloat16.
   Minimal accuracy loss, 2× memory reduction, no special libraries needed.

4. **INT4 weight-only** (via bitsandbytes): NF4 or FP4 quantization of
   Linear weights. 8× memory reduction. Requires bitsandbytes package.

The gsplat decoder and geometric preprocessing ops (visual hull, ray query)
are NOT quantized — they remain in fp32.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class QuantizationConfig:
    """Configuration for encoder quantization.

    Attributes:
        method: Quantization method.
        exclude_heads: If True, keep GSLayer output heads in fp16/fp32.
        calibration_batches: Number of batches for static quantization calibration.
        bitsandbytes_4bit_type: NF4 or FP4 for bitsandbytes.
    """
    method: Literal[
        "int8_dynamic",
        "int8_static",
        "bf16_weight_only",
        "int4_bitsandbytes",
    ] = "bf16_weight_only"
    exclude_heads: bool = True
    calibration_batches: int = 10
    bitsandbytes_4bit_type: str = "nf4"


def quantize_encoder(
    encoder: nn.Module,
    config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """Apply quantization to the PointSplat encoder.

    Args:
        encoder: The EncoderPointsplat module to quantize.
        config: Quantization configuration. If None, uses bf16_weight_only.

    Returns:
        Quantized encoder (modified in-place for some methods).

    Raises:
        ImportError: If required packages are not installed.
    """
    if config is None:
        config = QuantizationConfig()

    if config.method == "bf16_weight_only":
        return _quantize_bf16_weights(encoder, config)
    elif config.method == "int8_dynamic":
        return _quantize_int8_dynamic(encoder, config)
    elif config.method == "int8_static":
        return _quantize_int8_static(encoder, config)
    elif config.method == "int4_bitsandbytes":
        return _quantize_int4_bitsandbytes(encoder, config)
    else:
        raise ValueError(f"Unknown quantization method: {config.method}")


def _get_quantizable_modules(
    encoder: nn.Module,
    exclude_heads: bool = True,
) -> list[tuple[str, nn.Module]]:
    """Identify modules that should be quantized.

    Args:
        encoder: The encoder module.
        exclude_heads: If True, exclude GSLayer output heads.

    Returns:
        List of (name, module) pairs for quantizable Linear layers.
    """
    quantizable = []
    for name, module in encoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if exclude_heads:
            # Skip GSLayer output heads (xyz, scale, rotation, opacity, color)
            # These are in output_tokenizer.out_layers
            if "output_tokenizer" in name and "out_layers" in name:
                continue
            # Skip output_tokenizer.mlp_net (small, sensitive)
            if "output_tokenizer" in name and "mlp_net" in name:
                continue

        quantizable.append((name, module))

    return quantizable


def _quantize_bf16_weights(
    encoder: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """BF16 weight-only quantization.

    Simply casts all Linear weights to bfloat16. This is the simplest
    quantization method with minimal accuracy loss and 2× memory reduction.
    """
    print("[Quantization] Applying BF16 weight-only quantization...")

    quantizable = _get_quantizable_modules(encoder, config.exclude_heads)
    total_params = 0
    for name, module in quantizable:
        module.weight.data = module.weight.data.to(torch.bfloat16)
        total_params += module.weight.numel()

    print(f"  Quantized {len(quantizable)} Linear layers ({total_params / 1e6:.1f}M params)")
    print(f"  Memory reduction: ~2× (fp32 -> bf16)")

    encoder._quantization_method = "bf16_weight_only"
    return encoder


def _quantize_int8_dynamic(
    encoder: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """INT8 dynamic quantization using PyTorch built-in.

    Quantizes Linear layer weights to INT8. Activations are quantized
    dynamically at runtime. No calibration data needed.
    """
    print("[Quantization] Applying INT8 dynamic quantization...")

    # Get module names to quantize (exclude heads if configured)
    module_names = [
        name for name, _ in _get_quantizable_modules(encoder, config.exclude_heads)
    ]

    # PyTorch dynamic quantization works on module types, not individual modules.
    # We need to manually quantize each target Linear.
    total_params = 0
    for name, module in encoder.named_modules():
        if name not in module_names:
            continue

        # Create a dynamically quantized version
        quantized = torch.ao.quantization.quantize_dynamic(
            module, {nn.Linear}, dtype=torch.qint8
        )
        # Replace the module
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        if parent_name:
            parent = encoder.get_submodule(parent_name)
        else:
            parent = encoder
        setattr(parent, child_name, quantized)
        total_params += module.weight.numel()

    print(f"  Quantized {len(module_names)} Linear layers ({total_params / 1e6:.1f}M params)")
    print(f"  Memory reduction: ~4× (fp32 -> int8)")
    print(f"  Note: Dynamic quantization may not accelerate GPU inference.")
    print(f"        Best suited for CPU inference.")

    encoder._quantization_method = "int8_dynamic"
    return encoder


def _quantize_int8_static(
    encoder: nn.Module,
    config: QuantizationConfig,
    calibration_loader=None,
) -> nn.Module:
    """INT8 static quantization with calibration.

    Quantizes both weights and activations to INT8. Requires calibration
    with representative data to determine activation quantization ranges.

    Args:
        encoder: Encoder to quantize.
        config: Quantization configuration.
        calibration_loader: DataLoader for calibration. If None, returns
                           encoder with only weight quantization.
    """
    print("[Quantization] Applying INT8 static quantization...")

    # Prepare for quantization
    encoder.eval()

    # Add quantization stubs
    quant_config = torch.ao.quantization.get_default_qconfig("fbgemm")
    torch.ao.quantization.prepare(encoder, inplace=True)

    if calibration_loader is not None:
        print(f"  Calibrating with {config.calibration_batches} batches...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= config.calibration_batches:
                    break
                encoder(batch["context"])

        # Convert to quantized
        torch.ao.quantization.convert(encoder, inplace=True)
        print("  Static quantization complete.")
    else:
        print("  WARNING: No calibration data provided. Only weight quantization applied.")
        # Fall back to dynamic
        return _quantize_int8_dynamic(encoder, config)

    encoder._quantization_method = "int8_static"
    return encoder


def _quantize_int4_bitsandbytes(
    encoder: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """INT4 weight-only quantization using bitsandbytes.

    Quantizes Linear layer weights to 4-bit (NF4 or FP4 format).
    Requires the bitsandbytes package. 8× memory reduction.

    Args:
        encoder: Encoder to quantize.
        config: Quantization configuration.

    Raises:
        ImportError: If bitsandbytes is not installed.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for INT4 quantization. "
            "Install with: pip install bitsandbytes"
        )

    print(f"[Quantization] Applying INT4 ({config.bitsandbytes_4bit_type}) quantization...")

    quantizable = _get_quantizable_modules(encoder, config.exclude_heads)
    total_params = 0

    for name, module in quantizable:
        # Create 4-bit quantized linear replacement
        quantized_linear = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type=config.bitsandbytes_4bit_type,
            compute_dtype=torch.float16,
        )

        # Copy and quantize weights
        quantized_linear.weight = bnb.nn.Params4bit(
            module.weight.data.to("cuda"),
            requires_grad=False,
            quant_type=config.bitsandbytes_4bit_type,
        )
        if module.bias is not None:
            quantized_linear.bias = nn.Parameter(module.bias.data.clone())

        # Replace the module
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        if parent_name:
            parent = encoder.get_submodule(parent_name)
        else:
            parent = encoder
        setattr(parent, child_name, quantized_linear)
        total_params += module.weight.numel()

    print(f"  Quantized {len(quantizable)} Linear layers ({total_params / 1e6:.1f}M params)")
    print(f"  Memory reduction: ~8× (fp32 -> int4)")

    encoder._quantization_method = "int4_bitsandbytes"
    return encoder


def calibrate_encoder(
    encoder: nn.Module,
    calibration_loader,
    num_batches: int = 10,
) -> nn.Module:
    """Calibrate the encoder for static quantization.

    Runs the encoder on calibration data to collect activation statistics.

    Args:
        encoder: Encoder prepared for static quantization.
        calibration_loader: DataLoader yielding PointSplat batches.
        num_batches: Number of calibration batches.

    Returns:
        Calibrated encoder (ready for convert()).
    """
    print(f"[Calibration] Running {num_batches} calibration batches...")
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(calibration_loader, total=num_batches)):
            if i >= num_batches:
                break
            encoder(batch["context"])
    print("[Calibration] Done.")
    return encoder


def benchmark_encoder(
    encoder: nn.Module,
    batch: dict,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark encoder inference time.

    Args:
        encoder: Encoder to benchmark.
        batch: Sample batch for benchmarking.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed runs.

    Returns:
        Dict with timing statistics.
    """
    encoder.eval()
    device = next(encoder.parameters()).device

    # Memory before
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(batch["context"], 0)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = encoder(batch["context"], 0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    # Memory after
    peak_memory = 0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    # Model size
    param_size = sum(p.nelement() * p.element_size() for p in encoder.parameters()) / 1024**2

    results = {
        "method": getattr(encoder, "_quantization_method", "fp32"),
        "avg_time_ms": (sum(times) / len(times)) * 1000,
        "min_time_ms": min(times) * 1000,
        "max_time_ms": max(times) * 1000,
        "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        "param_size_mb": param_size,
        "peak_gpu_memory_mb": peak_memory,
        "num_runs": num_runs,
    }

    print(f"\n[Benchmark] Method: {results['method']}")
    print(f"  Avg time: {results['avg_time_ms']:.1f} ms")
    print(f"  Min time: {results['min_time_ms']:.1f} ms")
    print(f"  Max time: {results['max_time_ms']:.1f} ms")
    print(f"  Param size: {results['param_size_mb']:.1f} MB")
    print(f"  Peak GPU memory: {results['peak_gpu_memory_mb']:.1f} MB")

    return results
