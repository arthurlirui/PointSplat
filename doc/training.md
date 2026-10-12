# PointSplat Training

Both training modes use `+experiment=pointsplat_training`. The preset defaults to
train-from-scratch on processed DNA-Rendering/RenBody data and disables Gaussian
opacity pruning during training.

The default loader expects the processed DNA-Rendering distribution described by
[Diffuman4D](https://github.com/zju3dv/Diffuman4D/blob/main/README.md#processed-dna-rendering-dataset).
Its root should contain `validation_index.json`, `scenes/<scene>/transforms.json`,
`scenes/<scene>/images/<view>/<frame>`, and
`masks/<scene>/fmasks/<view>/<frame>`. Adjust `dataset.roots` if your data lives
elsewhere.

Each run writes Lightning checkpoints, JSONL metrics, TensorBoard events, and
training visualizations under `output_dir`. To resume, reuse the same
`output_dir`, set `checkpointing.resume=true`, and increase `trainer.max_steps`.

## Fine-tune from the official checkpoint

Fine-tuning initializes the encoder with `pretrained/pointsplat_mixed.pt`, uses
a lower learning rate, and restores the checkpoint's Gaussian scale schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29610 \
  -m src.main \
  +experiment=pointsplat_training \
  checkpointing.pretrained_encoder=pretrained/pointsplat_mixed.pt \
  optimizer.lr=4.0e-5 \
  'model.encoder.gaussians_decoder.clip_scaling=[-1.0,0.0,3.0,0.0]' \
  output_dir=experiments/pointsplat/dna_finetune
```

## Train from scratch

The shared preset already contains the from-scratch settings: no checkpoint,
`optimizer.lr=2e-4`, `trainer.max_steps=300000`, and
`model.encoder.gaussians_decoder.clip_scaling=[100.0,1e-10,3.0,3000.0]`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nnodes=1 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=29611 \
  -m src.main \
  +experiment=pointsplat_training \
  output_dir=experiments/pointsplat/dna_scratch
```

Change `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, `--master_port`, and the
usual Hydra overrides for your machine and experiment.

## Custom dataset (example-dataset)

The registered `example` dataset is a template rather than support for a specific
external dataset. Copy and rename both files, then register the new loader in
`src/dataset/__init__.py`:

```bash
cp src/dataset/dataset_example.py src/dataset/dataset_mydataset.py
cp config/dataset/example.yaml config/dataset/mydataset.yaml
```

The template uses the processed DNA/RenBody layout:

```text
datasets/example_dataset/
├── validation_index.json
├── scenes/<scene>/
│   ├── transforms.json
│   └── images/<view>/<frame>.png
└── masks/<scene>/fmasks/<view>/<frame>.png
```

`transforms.json` must describe every camera with its coordinate system,
camera-to-world pose, and normalized intrinsics. Replace the
`TODO(custom-data)` readers with your source-specific parsing while preserving
the template's output contract.

In particular, keep these operations:

1. Convert OpenGL/Blender poses to OpenCV camera axes, or explicitly declare the
   source convention as `opencv`.
2. Normalize the complete camera rig with
   `preprocess_poses(..., return_scale=True)`, then normalize sampled
   context/target views with `preprocess_sampled_views(..., scene_scale=...)`.
   Do not use raw metric poses or normalize views independently.
3. Use normalized intrinsics, and update them after any crop or resize.
4. Return aligned binary foreground masks and positive `near`/`far` bounds in the
   same normalized camera space.

## Debug training issues

Before training, inspect one batch and its visual hull:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/debug_pointsplat_dataset.py \
  +experiment=pointsplat_training \
  dataset=example \
  dataset.roots='[datasets/example_dataset]' \
  output_dir=experiments/debug/example_dataset \
  dataset.num_training_scenes=1 \
  dataset.frame_per_training_scene=1
```

The check fails on non-finite tensors, empty or misaligned foreground masks, an
empty/degenerate visual hull, nonzero training-time opacity pruning, or an
opacity head that does not initialize every Gaussian to `init_density`. Inspect
the printed point counts and `dataset_debug/context_mask_overlay.png`; a mask
that truncates the subject or covers the whole image usually indicates bad crop,
pose, or intrinsics metadata.

For a smoke test, keep the same overrides and run a short training job (for
example, 10 steps on one scene) with `train.vis_every_n_steps=1`. Confirm that
`train/gs_num` is positive, losses are finite, and the subject is not truncated
in the prediction images under `training_vis/` before launching a full run.
