# Changelog

All notable changes to the PointSplat project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-26

### Added — Extension Modules

#### Segmentation Integration (`src/modules/segmentation/`)
- **SegmentationShim**: Unified interface for automatic human mask generation
- **SAMSegmenter**: Meta SAM backend with automatic and point-prompt modes
- **SegFormerSegmenter**: HuggingFace SegFormer backend for person-class segmentation
- **SimpleThresholdSegmenter**: Color-distance-based fallback (no dependencies)
- **generate_masks.py**: Offline mask generation script with support for DNA-Rendering and THuman2.0 layouts
- Integration as data shim: `SegmentationShim.apply_to_example()` replaces masks in pipeline

#### Iterative Optimization (`src/modules/iterative_optim/`)
- **LatentOptimizer**: Per-scene latent optimization with frozen encoder (recommended)
- **GaussianOptimizer**: Direct Gaussian parameter optimization with optional densification
- **IterativeOptimizer**: Unified wrapper for both modes
- **OptimConfig**: Configurable loss weights (MAE, LPIPS, SSIM), densification thresholds
- Cosine annealing LR schedule, gradient-based pruning, visualization logging

#### Quantization (`src/modules/quantization/`)
- **INT8 dynamic quantization**: PyTorch built-in, no calibration needed
- **INT8 static quantization**: With calibration data for better performance
- **BF16 weight-only**: Simplest method, 2× memory reduction, minimal accuracy loss
- **INT4 weight-only**: Via bitsandbytes (NF4/FP4), 8× memory reduction
- **benchmark_encoder()**: Latency and memory benchmarking utility
- **calibrate_encoder()**: Calibration data collection for static quantization
- GSLayer output heads excluded from quantization for numerical stability

#### Jetson SLAM (`src/modules/jetson_slam/`)
- **EdgeInferenceEngine**: Edge-optimized inference with quantization and pruning
- **EdgeInferenceConfig**: Pre-configured profiles for Jetson AGX Orin, Orin NX, Xavier AGX/NX
- **GaussianFusion**: Multi-frame temporal fusion with spatial proximity matching
  - Confidence tracking with temporal decay
  - Merge/accumulate/prune pipeline
  - Configurable max Gaussians limit
- **GaussianSLAM**: Full SLAM pipeline integrating:
  - Camera tracking interface (ORB-SLAM3/VINS compatible)
  - Automatic segmentation
  - Feed-forward prediction
  - Multi-frame Gaussian fusion
  - Novel view rendering from global model
  - Trajectory saving

### Added — Unified Demo Entry Point
- **scripts/demo.py**: Single entry point for all demo modes
  - `--mode baseline`: Standard inference
  - `--mode segmentation`: Inference with auto-segmentation
  - `--mode iterative`: Feed-forward + iterative optimization
  - `--mode quantized`: Quantized inference
  - `--mode slam`: SLAM-style multi-frame reconstruction
  - `--list-modes`: Show available modes
- Supports both DNA-Rendering and THuman2.0 datasets
- Automatic video generation after inference

### Added — Preprocessing Scripts
- **scripts/preprocess/preprocess_dna_rendering.py**: DNA-Rendering data preprocessing
  - `--mode validate`: Validate dataset structure
  - `--mode colmap`: Convert COLMAP/NeRF-style export
  - `--mode raw`: Convert raw DNA-Rendering release
- **scripts/preprocess/preprocess_thuman.py**: THuman2.0 data preprocessing
  - `--mode validate`: Validate dataset structure
  - `--mode render`: Render multi-view images from mesh data
  - `--mode convert`: Convert existing dataset naming conventions

### Added — Build Environment
- **scripts/setup_build_env.py**: Windows build environment for gsplat JIT compilation
  - Configures MSVC 2022 paths (cl.exe, INCLUDE, LIB)
  - Sets CUDA_HOME and nvcc paths
  - Patches distutils._msvccompiler for setuptools 80+
  - Fixes gsplat -Wno-attributes incompatibility on Windows
- **scripts/setup_env.sh**: Shell script for environment setup
- **scripts/make_video_demo.py**: OpenCV-based video generation from inference images
- **scripts/run_demo.sh**: One-command demo runner

### Added — Documentation
- **docs/extensions.md**: Extension modules documentation
- **docs/deployment.md**: Edge deployment guide
- **CHANGELOG.md**: This file

### Changed
- **src/main.py**: Added automatic build environment setup for Windows (gsplat JIT)
- **requirements.txt**: No changes (all new dependencies are optional)

### Fixed
- gsplat CUDA JIT compilation on Windows (MSVC + distutils._msvccompiler patch)
- DataLoader worker crash on Windows (`num_workers=0` recommendation)
- numpy version compatibility (pinned to 1.24.4, compatible opencv-python/plyfile)

### Verified
- DNA-Rendering inference: PSNR 26.08, SSIM 0.899, LPIPS 0.075
- THuman2.0 inference: 10 scenes, 39s total
- 26 video demos generated (16 DNA-Rendering + 10 THuman2.0)
- Conda environment: Python 3.10, PyTorch 2.4.0 + CUDA 12.1, RTX A4500

---

## [0.1.0] - 2026-10-15 (Initial Release)

### Added
- PointSplat inference code (ECCV 2026)
- Pretrained checkpoint: `pointsplat_mixed.pt`
- Example datasets: DNA-Rendering (16 scenes) and THuman2.0 (10 scenes)
- Training and fine-tuning support
- Configurable dataset pipeline (Hydra-based)
- LPIPS + MAE loss functions
- gsplat-based differentiable Gaussian rasterization decoder
