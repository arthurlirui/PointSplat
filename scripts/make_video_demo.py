"""Generate a video demo from PointSplat inference output images.

PointSplat saves rendered target views as individual PNG files under
``<output_dir>/images/<scene>/color/000000.png``, ``000001.png``, etc.
This script collects those images and assembles them into an MP4 video,
sorted by view index, so you can visualise the novel-view synthesis result
as a fly-around.

This is a standalone alternative to ``test.save_video=true`` (which uses
skvideo/FFmpeg at inference time).  It uses OpenCV's VideoWriter, which is
already a dependency (``opencv_python``) and ships its own FFmpeg.

Usage
-----
# Basic: build a video from one scene's rendered images
python scripts/make_video_demo.py \
    --input experiments/eval_pointsplat_dna_rendering/images \
    --output experiments/video_demo \
    --fps 10

# Include GT comparison side-by-side
python scripts/make_video_demo.py \
    --input experiments/eval_pointsplat_dna_rendering/images \
    --output experiments/video_demo \
    --fps 10 --with-gt

# Include input context views as a prefix
python scripts/make_video_demo.py \
    --input experiments/eval_pointsplat_dna_rendering/images \
    --output experiments/video_demo \
    --fps 10 --with-gt --with-input

# Process all scene directories at once
python scripts/make_video_demo.py \
    --input experiments/eval_pointsplat_dna_rendering/images \
    --output experiments/video_demo \
    --fps 10 --all-scenes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def find_scenes(input_dir: Path) -> list[str]:
    """Find all scene directories under the input images/ directory."""
    scenes = []
    for d in sorted(input_dir.iterdir()):
        if d.is_dir() and (d / "color").is_dir():
            scenes.append(d.name)
    return scenes


def load_scene_images(
    scene_dir: Path,
    include_gt: bool = False,
    include_input: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, list[np.ndarray] | None]:
    """Load rendered images for a scene, sorted by view index.

    Returns (predictions, ground_truths, input_views).
    """
    color_dir = scene_dir / "color"
    if not color_dir.is_dir():
        return [], None, None

    pred_files = {}
    gt_files = {}
    input_files = {}

    for f in color_dir.iterdir():
        if not f.is_file():
            continue
        # Match "000023.png" (not "000023_gt.png", not "input_000023.png")
        name = f.name
        if name.startswith("input_"):
            m = re.match(r"input_(\d+)\.png", name)
            if m:
                input_files[int(m.group(1))] = f
        elif name.endswith("_gt.png"):
            m = re.match(r"(\d+)_gt\.png", name)
            if m:
                gt_files[int(m.group(1))] = f
        elif name.endswith(".png"):
            m = re.match(r"(\d+)\.png", name)
            if m:
                pred_files[int(m.group(1))] = f

    # Sort by index
    pred_sorted = [pred_files[k] for k in sorted(pred_files.keys())]
    gt_sorted = [gt_files[k] for k in sorted(gt_files.keys())] if gt_files else []
    input_sorted = [input_files[k] for k in sorted(input_files.keys())] if input_files else []

    predictions = [cv2.imread(str(f)) for f in pred_sorted]
    predictions = [p for p in predictions if p is not None]

    ground_truths = None
    if include_gt and gt_sorted:
        ground_truths = [cv2.imread(str(f)) for f in gt_sorted]
        ground_truths = [g for g in ground_truths if g is not None]

    input_views = None
    if include_input and input_sorted:
        input_views = [cv2.imread(str(f)) for f in input_sorted]
        input_views = [i for i in input_views if i is not None]

    return predictions, ground_truths, input_views


def make_side_by_side(
    predictions: list[np.ndarray],
    ground_truths: list[np.ndarray] | None,
) -> list[np.ndarray]:
    """Create side-by-side comparison frames (pred | gt)."""
    frames = []
    n = len(predictions)
    gt_n = len(ground_truths) if ground_truths else 0
    for i in range(n):
        pred = predictions[i]
        if ground_truths and i < gt_n:
            gt = ground_truths[i]
            # Ensure same height
            h = max(pred.shape[0], gt.shape[0])
            if pred.shape[0] != h:
                pred = cv2.resize(pred, (int(pred.shape[1] * h / pred.shape[0]), h))
            if gt.shape[0] != h:
                gt = cv2.resize(gt, (int(gt.shape[1] * h / gt.shape[0]), h))
            # Add a gap
            gap = np.ones((h, 10, 3), dtype=np.uint8) * 128
            frame = np.hstack([pred, gap, gt])
        else:
            frame = pred
        frames.append(frame)
    return frames


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Write frames to an MP4 file using OpenCV's VideoWriter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"  ERROR: VideoWriter failed to open for {path}")
        return
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  Saved {len(frames)} frames -> {path}")


def process_scene(
    scene_dir: Path,
    output_dir: Path,
    fps: int,
    with_gt: bool,
    with_input: bool,
) -> None:
    """Process one scene directory into a video."""
    scene_name = scene_dir.name
    print(f"\nProcessing scene: {scene_name}")

    predictions, ground_truths, input_views = load_scene_images(
        scene_dir, include_gt=with_gt, include_input=with_input
    )

    if not predictions:
        print(f"  WARNING: No prediction images found for {scene_name}")
        return

    print(f"  Found {len(predictions)} prediction frames")

    # Build the combined frames
    if with_gt and ground_truths:
        print(f"  Found {len(ground_truths)} GT frames")
        frames = make_side_by_side(predictions, ground_truths)
    else:
        frames = predictions

    # Optionally prepend input views
    if with_input and input_views:
        print(f"  Found {len(input_views)} input context frames")
        # Add input views as prefix, with a label gap
        gap = np.ones((input_views[0].shape[0], 10, 3), dtype=np.uint8) * 128
        all_frames = list(input_views) + [gap] + frames
        # Ensure all same width
        max_w = max(f.shape[1] for f in all_frames)
        normalized = []
        for f in all_frames:
            if f.shape[1] != max_w:
                f = cv2.copyMakeBorder(
                    f, 0, 0, 0, max_w - f.shape[1], cv2.BORDER_CONSTANT, value=0
                )
            normalized.append(f)
        frames = normalized

    # Write the video
    suffix = "_compare" if with_gt else ""
    video_path = output_dir / f"{scene_name}{suffix}.mp4"
    write_video(frames, video_path, fps)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video demos from PointSplat inference output."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the images/ directory (e.g. experiments/eval/images).",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/video_demo",
        help="Output directory for videos.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate.")
    parser.add_argument("--with-gt", action="store_true", help="Include GT side-by-side.")
    parser.add_argument("--with-input", action="store_true", help="Include input context views as prefix.")
    parser.add_argument("--all-scenes", action="store_true", help="Process all scene directories.")
    parser.add_argument("--scene", type=str, default=None, help="Process a specific scene.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.all_scenes:
        scenes = find_scenes(input_dir)
        if not scenes:
            print(f"ERROR: No scene directories found in {input_dir}")
            sys.exit(1)
        print(f"Found {len(scenes)} scenes: {scenes}")
        for scene in scenes:
            process_scene(input_dir / scene, output_dir, args.fps, args.with_gt, args.with_input)
    elif args.scene:
        scene_dir = input_dir / args.scene
        if not scene_dir.is_dir():
            print(f"ERROR: Scene directory not found: {scene_dir}")
            sys.exit(1)
        process_scene(scene_dir, output_dir, args.fps, args.with_gt, args.with_input)
    else:
        # If input_dir itself is a scene directory (has color/ subdir)
        if (input_dir / "color").is_dir():
            process_scene(input_dir, output_dir, args.fps, args.with_gt, args.with_input)
        else:
            scenes = find_scenes(input_dir)
            if not scenes:
                print(f"ERROR: No scene directories found in {input_dir}")
                print("Use --all-scenes or --scene <name> to specify which to process.")
                sys.exit(1)
            print(f"Found {len(scenes)} scenes. Processing the first one.")
            print(f"Use --all-scenes to process all.")
            process_scene(input_dir / scenes[0], output_dir, args.fps, args.with_gt, args.with_input)

    print(f"\nDone! Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
