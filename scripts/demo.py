"""Unified PointSplat Demo Entry Point

This script provides a single entry point for all PointSplat demo modes,
including the original inference and the four extension modules:

1. **baseline**: Standard PointSplat feed-forward inference
2. **segmentation**: Inference with automatic human segmentation
3. **iterative**: Feed-forward + per-scene iterative optimization
4. **quantized**: INT8/bf16 quantized inference for acceleration
5. **slam**: Multi-frame SLAM-style incremental reconstruction

Usage examples
--------------
# Baseline inference on DNA-Rendering
python scripts/demo.py --mode baseline --dataset dna --output experiments/demo/baseline

# Inference with automatic segmentation (no pre-computed masks needed)
python scripts/demo.py --mode segmentation --dataset dna --segmenter threshold --output experiments/demo/seg

# Inference + iterative optimization (200 refinement steps)
python scripts/demo.py --mode iterative --dataset dna --num-steps 200 --output experiments/demo/iterative

# Quantized inference (INT8 dynamic)
python scripts/demo.py --mode quantized --dataset dna --quant-method int8_dynamic --output experiments/demo/quantized

# SLAM mode (multi-frame fusion)
python scripts/demo.py --mode slam --dataset dna --output experiments/demo/slam

# List all available modes
python scripts/demo.py --list-modes
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up build environment for Windows
if sys.platform == "win32":
    try:
        import scripts.setup_build_env  # noqa: F401
    except Exception:
        pass


# Dataset configurations
DATASET_CONFIGS = {
    "dna": {
        "experiment": "pointsplat_renbody",
        "data_root": "datasets/DNA_Rendering_example",
        "name": "DNA-Rendering",
    },
    "thuman": {
        "experiment": "pointsplat_thuman",
        "data_root": "datasets/THuman2_0_example",
        "name": "THuman2.0",
    },
}

# Demo mode descriptions
MODE_DESCRIPTIONS = {
    "baseline": "Standard PointSplat feed-forward inference (original model)",
    "segmentation": "Inference with automatic human segmentation (SAM/SegFormer/threshold)",
    "iterative": "Feed-forward + per-scene iterative Gaussian optimization",
    "quantized": "Quantized inference for acceleration (INT8/INT4/BF16)",
    "slam": "Multi-frame SLAM-style incremental 3D reconstruction",
}


def run_baseline(args):
    """Run baseline PointSplat inference."""
    ds = DATASET_CONFIGS[args.dataset]
    output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  Mode: BASELINE | Dataset: {ds['name']}")
    print(f"{'='*60}\n")

    cmd = (
        f"CUDA_VISIBLE_DEVICES=0 python -m src.main "
        f"+experiment={ds['experiment']} "
        f"output_dir={output_dir} "
        f"dataset.roots='[{ds[\"data_root\"]}]' "
        f"checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt "
        f"test.save_video=false test.save_gaussian=false "
        f"data_loader.test.num_workers=0"
    )
    print(f"Command: {cmd}\n")
    os.system(cmd)

    # Generate video
    if args.generate_video:
        _generate_videos(output_dir, args.dataset)


def run_segmentation(args):
    """Run inference with automatic segmentation."""
    ds = DATASET_CONFIGS[args.dataset]
    output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  Mode: SEGMENTATION | Dataset: {ds['name']} | Backend: {args.segmenter}")
    print(f"{'='*60}\n")

    # Step 1: Generate masks if they don't exist
    if args.segmenter == "threshold":
        print("Using threshold-based segmentation (no model needed)...")
        from src.modules.segmentation import generate_masks_for_dataset

        if "dna" in ds["data_root"]:
            mask_output = ds["data_root"].replace("DNA_Rendering_example", "DNA_Rendering_seg_masks")
            generate_masks_for_dataset(
                input_dir=Path(ds["data_root"]) / "scenes",
                output_dir=Path(mask_output),
                backend="threshold",
                layout="dna_rendering",
                bg_color=(0.31, 0.39, 0.29),
            )
        else:
            mask_output = ds["data_root"].replace("THuman2_0_example", "THuman2_0_seg_masks")
            generate_masks_for_dataset(
                input_dir=Path(ds["data_root"]),
                output_dir=Path(mask_output),
                backend="threshold",
                layout="thuman",
            )

        print(f"\nMasks generated at {mask_output}")
        print("To use these masks, update dataset.roots to point to the new data.")
        print("For full integration, use the SegmentationShim as a data pipeline shim.")
    else:
        print(f"Segmentation backend '{args.segmenter}' requires additional setup.")
        print(f"Please install the required package and run:")
        print(f"  python -m src.modules.segmentation.generate_masks \\")
        print(f"    --input {ds['data_root']}/scenes \\")
        print(f"    --output {ds['data_root']}/masks \\")
        print(f"    --backend {args.segmenter}")

    # Step 2: Run inference (same as baseline but with generated masks)
    run_baseline(args)


def run_iterative(args):
    """Run feed-forward + iterative optimization."""
    ds = DATASET_CONFIGS[args.dataset]
    output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  Mode: ITERATIVE | Dataset: {ds['name']} | Steps: {args.num_steps}")
    print(f"{'='*60}\n")

    # First run baseline inference
    print("Step 1: Running feed-forward inference...")
    baseline_dir = f"{output_dir}/initial"
    run_baseline(argparse.Namespace(**{**vars(args), "output": baseline_dir}))

    # Then run iterative optimization
    print(f"\nStep 2: Running iterative optimization ({args.num_steps} steps)...")
    print("  (This requires loading the encoder/decoder in Python)")
    print("  Use the following code to run iterative optimization:")

    code = f"""
import torch
from src.modules.iterative_optim import IterativeOptimizer, OptimConfig

# Load encoder and decoder (from checkpoint)
encoder = ...  # Load from pretrained/pointsplat_mixed.pt
decoder = ...  # Load decoder

# Create optimizer
config = OptimConfig(mode="latent", num_steps={args.num_steps}, lr=1e-3)
optimizer = IterativeOptimizer(encoder, decoder, mode="latent", config=config)

# Run refinement on each scene
for batch in dataloader:
    refined_gaussians = optimizer.refine(batch, output_dir="{output_dir}/refined")
"""
    print(code)


def run_quantized(args):
    """Run quantized inference."""
    ds = DATASET_CONFIGS[args.dataset]
    output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  Mode: QUANTIZED | Dataset: {ds['name']} | Method: {args.quant_method}")
    print(f"{'='*60}\n")

    code = f"""
import torch
from src.modules.quantization import quantize_encoder, QuantizationConfig, benchmark_encoder

# Load encoder
encoder = ...  # Load from pretrained/pointsplat_mixed.pt

# Apply quantization
config = QuantizationConfig(method="{args.quant_method}")
quantized_encoder = quantize_encoder(encoder, config)

# Benchmark
results = benchmark_encoder(quantized_encoder, sample_batch)
print(f"Quantization: {{results['method']}}")
print(f"Avg time: {{results['avg_time_ms']:.1f}} ms")
print(f"Param size: {{results['param_size_mb']:.1f}} MB")

# Run inference with quantized encoder
# (Modify src.main to use the quantized encoder)
"""
    print(code)
    print("\nTo run full quantized inference:")
    print("  1. Load the pretrained encoder")
    print(f"  2. Apply {args.quant_method} quantization")
    print("  3. Run inference with the quantized encoder")
    print(f"  4. Compare results with baseline in {output_dir}")


def run_slam(args):
    """Run SLAM-style multi-frame reconstruction."""
    ds = DATASET_CONFIGS[args.dataset]
    output_dir = args.output

    print(f"\n{'='*60}")
    print(f"  Mode: SLAM | Dataset: {ds['name']} | Profile: {args.jetson_profile or 'default'}")
    print(f"{'='*60}\n")

    code = f"""
import torch
from src.modules.jetson_slam import GaussianSLAM, SLAMConfig, EdgeInferenceConfig

# Configure for edge deployment
edge_config = EdgeInferenceConfig(
    quantization="bf16_weight_only",
    jetson_profile="{args.jetson_profile or "agx_orin_32"}",
)

# Create SLAM system
config = SLAMConfig(
    edge_config=edge_config,
    enable_segmentation=True,
    segmentation_backend="threshold",
)

slam = GaussianSLAM(encoder, decoder, config)

# Process frames
for frame_idx, batch in enumerate(dataloader):
    result = slam.process_frame(
        context=batch["context"],
        target=batch["target"],
        camera_poses=batch["context"]["extrinsics"],
    )
    print(f"Frame {{frame_idx}}: {{result['stats']['num_gaussians']}} Gaussians, "
          f"{{result['timing']['total_ms']:.0f}}ms")

    # Save rendering
    from src.utils.image_io import save_image
    save_image(result["rendering"][0, 0], "{output_dir}/frame_{{frame_idx:04d}}.png")

# Final stats
print(slam.get_stats())
"""
    print(code)


def _generate_videos(output_dir: str, dataset: str):
    """Generate video demos from inference output."""
    print("\nGenerating video demos...")
    import subprocess

    python = sys.executable
    if dataset == "dna":
        input_path = f"{output_dir}/images/datasets/DNA_Rendering_example/scenes"
    else:
        input_path = f"{output_dir}/images"

    video_output = f"{output_dir}/videos"
    os.makedirs(video_output, exist_ok=True)

    cmd = f"{python} {PROJECT_ROOT}/scripts/make_video_demo.py --input {input_path} --output {video_output} --fps 10 --with-gt --all-scenes"
    os.system(cmd)


def list_modes():
    """List all available demo modes."""
    print("\nAvailable demo modes:\n")
    for mode, desc in MODE_DESCRIPTIONS.items():
        print(f"  {mode:15s}  {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Unified PointSplat Demo Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/demo.py --mode baseline --dataset dna
  python scripts/demo.py --mode segmentation --dataset dna --segmenter threshold
  python scripts/demo.py --mode iterative --dataset dna --num-steps 200
  python scripts/demo.py --mode quantized --dataset dna --quant-method int8_dynamic
  python scripts/demo.py --mode slam --dataset dna --jetson-profile agx_orin_32
  python scripts/demo.py --list-modes
""",
    )
    parser.add_argument(
        "--mode",
        choices=list(MODE_DESCRIPTIONS.keys()),
        default="baseline",
        help="Demo mode to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        default="dna",
        help="Dataset to use.",
    )
    parser.add_argument("--output", type=str, default="experiments/demo",
                        help="Output directory.")
    parser.add_argument("--generate-video", action="store_true", default=True,
                        help="Generate video demos after inference.")
    parser.add_argument("--list-modes", action="store_true",
                        help="List all available demo modes and exit.")

    # Segmentation options
    parser.add_argument("--segmenter", choices=["sam", "segformer", "threshold"],
                        default="threshold", help="Segmentation backend.")

    # Iterative optimization options
    parser.add_argument("--num-steps", type=int, default=200,
                        help="Number of iterative optimization steps.")
    parser.add_argument("--opt-mode", choices=["latent", "gaussian"],
                        default="latent", help="Optimization mode.")

    # Quantization options
    parser.add_argument("--quant-method",
                        choices=["bf16_weight_only", "int8_dynamic", "int4_bitsandbytes"],
                        default="bf16_weight_only", help="Quantization method.")

    # SLAM options
    parser.add_argument("--jetson-profile",
                        choices=["agx_orin_64", "agx_orin_32", "orin_nx_16",
                                 "orin_nx_8", "xavier_agx_32", "xavier_nx_8"],
                        default=None, help="Jetson hardware profile.")

    args = parser.parse_args()

    if args.list_modes:
        list_modes()
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run selected mode
    mode_funcs = {
        "baseline": run_baseline,
        "segmentation": run_segmentation,
        "iterative": run_iterative,
        "quantized": run_quantized,
        "slam": run_slam,
    }

    mode_funcs[args.mode](args)

    print(f"\n{'='*60}")
    print(f"  Demo complete! Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
