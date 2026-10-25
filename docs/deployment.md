# Edge Deployment Guide

This guide covers deploying PointSplat on Jetson edge devices and configuring
the SLAM pipeline for real-time human capture.

## Prerequisites

### Jetson Setup
1. Install JetPack 5.x or 6.x (includes CUDA, cuDNN, TensorRT)
2. Install PyTorch for Jetson (follow NVIDIA's Jetson PyTorch guide)
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### gsplat Compilation
gsplat requires JIT compilation on first run. On Jetson:

```bash
export TORCH_CUDA_ARCH_LIST="8.7"  # Orin
# Or: export TORCH_CUDA_ARCH_LIST="7.2"  # Xavier
python -c "from gsplat.cuda._backend import _C; print('gsplat compiled!')"
```

This takes 5-10 minutes on first run.

## Deployment Profiles

### Jetson AGX Orin (64GB/32GB)
- **Quantization**: INT8 dynamic
- **Max Gaussians**: 120K-150K
- **Resolution**: 512×512
- **Expected latency**: 3-5s/frame

### Jetson Orin NX (16GB/8GB)
- **Quantization**: INT8 (16GB) / INT4 (8GB)
- **Max Gaussians**: 50K-80K
- **Resolution**: 256-384
- **Expected latency**: 5-10s/frame

### Jetson Xavier NX (8GB)
- **Quantization**: INT4 (bitsandbytes)
- **Max Gaussians**: 40K
- **Resolution**: 256×256
- **Expected latency**: 8-15s/frame

## SLAM Integration

### Camera Tracking
PointSplat SLAM expects camera poses from an external tracker. Compatible trackers:

- **ORB-SLAM3**: Visual SLAM, outputs camera poses
- **VINS-Mono/Fusion**: Visual-inertial SLAM
- **ARKit/ARCore**: Mobile AR frameworks

Interface:
```python
# camera_poses: [V, 4, 4] c2w matrices
result = slam.process_frame(context, target, camera_poses)
```

### Segmentation
For in-the-wild deployment, use the `threshold` backend (no model needed)
or `segformer` (lighter than SAM).

### Multi-frame Fusion
The GaussianFusion module maintains a global 3D model:
- **Merge distance**: 0.02 (adjust based on scene scale)
- **Temporal weight**: 0.7 (higher = more stable, lower = more responsive)
- **Max Gaussians**: Limited by device memory

## TensorRT Export (Optional)

For maximum inference speed on Jetson:

```bash
pip install torch2trt
```

```python
from torch2trt import torch2trt

# Create example input
example_input = torch.randn(1, 8, 9, 512, 512).cuda()

# Convert encoder to TensorRT
encoder_trt = torch2trt(encoder, [example_input], fp16_mode=True)

# Save
torch.save(encoder_trt.state_dict(), "encoder_trt.pth")
```

Expected 2-3× speedup over PyTorch INT8.

## Troubleshooting

### Out of Memory (OOM)
- Reduce `max_gaussians`
- Lower `vhull_resolution` (64 or 48)
- Use INT4 quantization
- Reduce render resolution

### Slow Inference
- Check CUDA is being used: `torch.cuda.is_available()`
- Enable TF32: `torch.set_float32_matmul_precision("high")`
- Use TensorRT export
- Reduce input resolution with `downsample_ratio`

### gsplat Compilation Failure
- Ensure CUDA toolkit is installed (`nvcc --version`)
- Set `TORCH_CUDA_ARCH_LIST` correctly for your Jetson model
- Check MSVC/compiler availability
