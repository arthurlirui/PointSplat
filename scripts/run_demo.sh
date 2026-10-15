#!/bin/bash
# PointSplat Demo Runner - complete pipeline from setup to video output
#
# Usage:
#   bash scripts/run_demo.sh           # Run both demos
#   bash scripts/run_demo.sh dna       # Run DNA-Rendering demo only
#   bash scripts/run_demo.sh thuman    # Run THuman2.0 demo only
#
# Prerequisites:
#   - Conda environment "pointsplat" created (see setup below)
#   - Datasets downloaded to datasets/
#   - Checkpoint downloaded to pretrained/

set -e

PYTHON="D:/conda-envs/pointsplat/python.exe"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-all}"

run_dna_rendering() {
    echo "=========================================="
    echo "  DNA-Rendering Inference Demo"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.main \
        +experiment=pointsplat_renbody \
        output_dir=experiments/eval_pointsplat_dna_rendering \
        dataset.roots='[datasets/DNA_Rendering_example]' \
        checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt \
        test.save_video=false \
        test.save_gaussian=false \
        data_loader.test.num_workers=0

    echo "Generating DNA-Rendering video demos..."
    $PYTHON scripts/make_video_demo.py \
        --input "experiments/eval_pointsplat_dna_rendering/images/datasets/DNA_Rendering_example/scenes" \
        --output "experiments/video_demo/dna_rendering" \
        --fps 10 --with-gt --all-scenes
}

run_thuman() {
    echo "=========================================="
    echo "  THuman2.0 Inference Demo"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 $PYTHON -m src.main \
        +experiment=pointsplat_thuman \
        output_dir=experiments/eval_pointsplat_thuman \
        dataset.roots='[datasets/THuman2_0_example]' \
        checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt \
        test.save_video=false \
        test.save_gaussian=false \
        data_loader.test.num_workers=0

    echo "Generating THuman2.0 video demos..."
    $PYTHON scripts/make_video_demo.py \
        --input "experiments/eval_pointsplat_thuman/images" \
        --output "experiments/video_demo/thuman2" \
        --fps 10 --with-gt --all-scenes
}

case "$MODE" in
    dna)    run_dna_rendering ;;
    thuman) run_thuman ;;
    all)    run_dna_rendering; run_thuman ;;
    *)      echo "Usage: $0 [dna|thuman|all]"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo "  Done! Video demos saved to:"
echo "    experiments/video_demo/dna_rendering/"
echo "    experiments/video_demo/thuman2/"
echo "=========================================="
