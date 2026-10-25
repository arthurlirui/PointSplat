# PointSplat Extension Modules

This document describes the four extension modules added to PointSplat for
enhanced functionality: automatic segmentation, iterative optimization,
quantization, and edge SLAM deployment.

## Table of Contents
- [1. Segmentation Module](#1-segmentation-module)
- [2. Iterative Optimization Module](#2-iterative-optimization-module)
- [3. Quantization Module](#3-quantization-module)
- [4. Jetson SLAM Module](#4-jetson-slam-module)
- [Unified Demo Entry Point](#unified-demo-entry-point)

---

## 1. Segmentation Module

**Location**: `src/modules/segmentation/`

Replaces pre-computed masks with automatic human segmentation, enabling
PointSplat to run on in-the-wild images without mask files.

### Backends

| Backend | Dependencies | Accuracy | Speed | Use Case |
|---|---|---|---|---|
| `threshold` | None | Low (green-screen only) | Fast | DNA-Rendering style data |
| `sam` | segment-anything | High | Medium | General images |
| `segformer` | transformers | Medium | Fast | Person detection |

### Usage

**Offline mask generation:**
```bash
python -m src.modules.segmentation.generate_masks \
    --input datasets/my_data/scenes \
    --output datasets/my_data/masks \
    --backend threshold --layout dna_rendering
```

**Online (data shim):**
```python
from src.modules.segmentation import SegmentationShim

shim = SegmentationShim(backend="threshold")
mask = shim.segment(image_tensor)  # [3, H, W] -> [1, H, W]
example = shim.apply_to_example(example)  # Replace masks in batch
```

---

## 2. Iterative Optimization Module

**Location**: `src/modules/iterative_optim/`

Refines the feed-forward predicted Gaussians through per-scene optimization.
Expected PSNR improvement: +1-3 dB over baseline.

### Modes

| Mode | Description | Steps | Convergence |
|---|---|---|---|
| `latent` | Optimize per-scene latent (frozen encoder) | 100-500 | Fast |
| `gaussian` | Direct Gaussian param optimization + densify | 500-2000 | Slower |

### Usage

```python
from src.modules.iterative_optim import IterativeOptimizer, OptimConfig

config = OptimConfig(mode="latent", num_steps=200, lr=1e-3,
                     mae_weight=1.0, lpips_weight=0.1)
optimizer = IterativeOptimizer(encoder, decoder, mode="latent", config=config)
refined_gaussians = optimizer.refine(batch, output_dir="experiments/refined")
```

---

## 3. Quantization Module

**Location**: `src/modules/quantization/`

Reduces encoder memory footprint and accelerates inference.

### Methods

| Method | Memory | Speedup | Accuracy | Calibration |
|---|---|---|---|---|
| `bf16_weight_only` | 2× reduction | ~1.2× | Negligible loss | None |
| `int8_dynamic` | 4× reduction | ~1.5-2× (CPU) | Small loss | None |
| `int8_static` | 4× reduction | ~2× (CPU) | Small loss | Required |
| `int4_bitsandbytes` | 8× reduction | ~2.5× | Moderate loss | None |

### Usage

```python
from src.modules.quantization import quantize_encoder, QuantizationConfig, benchmark_encoder

config = QuantizationConfig(method="int8_dynamic")
quantized_encoder = quantize_encoder(encoder, config)
results = benchmark_encoder(quantized_encoder, sample_batch)
```

**Note**: The gsplat decoder is NOT quantized (always fp32). Only the encoder's
Linear layers are quantized. GSLayer output heads are kept in fp16 for stability.

---

## 4. Jetson SLAM Module

**Location**: `src/modules/jetson_slam/`

Edge deployment and incremental 3D reconstruction for real-time human capture.

### Components

- **EdgeInferenceEngine**: Quantized inference with pre-configured Jetson profiles
- **GaussianFusion**: Temporal multi-frame Gaussian fusion
- **GaussianSLAM**: Full SLAM pipeline (tracking + segmentation + prediction + fusion)

### Jetson Profiles

| Profile | Platform | Quantization | Max Gaussians | Resolution |
|---|---|---|---|---|
| `agx_orin_64` | Jetson AGX Orin 64GB | INT8 | 150K | 512×512 |
| `agx_orin_32` | Jetson AGX Orin 32GB | INT8 | 120K | 512×512 |
| `orin_nx_16` | Jetson Orin NX 16GB | INT8 | 80K | 384×384 |
| `orin_nx_8` | Jetson Orin NX 8GB | INT4 | 50K | 256×256 |
| `xavier_agx_32` | Xavier AGX 32GB | INT8 | 100K | 384×384 |
| `xavier_nx_8` | Xavier NX 8GB | INT4 | 40K | 256×256 |

### Usage

```python
from src.modules.jetson_slam import GaussianSLAM, SLAMConfig, EdgeInferenceConfig

config = SLAMConfig(
    edge_config=EdgeInferenceConfig(jetson_profile="agx_orin_32"),
    enable_segmentation=True,
    segmentation_backend="threshold",
)
slam = GaussianSLAM(encoder, decoder, config)

for frame in video_stream:
    result = slam.process_frame(context, target, camera_poses)
    print(f"Frame {frame_idx}: {result['stats']['num_gaussians']} Gaussians")
```

---

## Unified Demo Entry Point

**Location**: `scripts/demo.py`

Single entry point for all demo modes.

```bash
# List all modes
python scripts/demo.py --list-modes

# Baseline inference
python scripts/demo.py --mode baseline --dataset dna

# With auto-segmentation
python scripts/demo.py --mode segmentation --dataset dna --segmenter threshold

# With iterative optimization
python scripts/demo.py --mode iterative --dataset dna --num-steps 200

# Quantized inference
python scripts/demo.py --mode quantized --dataset dna --quant-method int8_dynamic

# SLAM mode
python scripts/demo.py --mode slam --dataset dna --jetson-profile agx_orin_32
```

### Datasets
- `--dataset dna`: DNA-Rendering example data
- `--dataset thuman`: THuman2.0 example data
