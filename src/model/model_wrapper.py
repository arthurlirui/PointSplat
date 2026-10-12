import torch
import time
from einops import rearrange
from pytorch_lightning import LightningModule
import numpy as np
import json
import os
import torch.nn.functional as F
import math
from PIL import Image
from torch import nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset import DatasetCfg
from ..global_cfg import get_cfg
from ..loss import Loss

from .encoder import Encoder, EncoderOutput
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .decoder.decoder import Decoder, DecoderOutput, DepthRenderingMode
from .types import Gaussians

from ..utils.benchmarker import Benchmarker
from ..utils.image_io import save_image, save_video
from ..utils.step_tracker import StepTracker
from ..utils.metrics_utils import compute_lpips, compute_psnr, compute_ssim
from ..utils.typing_utils import *
from ..utils.visualization.vis_depth import viz_depth_tensor
from ..utils.stablize_camera import render_stabilization_path
from ..utils.gaussiansplatting_utils import save_gaussians


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    downsample_context_ratio: float
    mask_output: bool
    move_to_cpu: bool


@dataclass
class OptimizerCfg:
    lr: float
    weight_decay: float


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    print_log_every_n_steps: int
    vis_every_n_steps: int
    train_ignore_large_loss: float
    mask_target: bool
    mask_output: bool


class ModelWrapper(LightningModule):
    dataset_cfg: DatasetCfg
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self._train_step_start: float | None = None
        self._backward_start: float | None = None
        self._last_backward_time = 0.0

        # This is used for testing.
        self.benchmarker = Benchmarker()

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def _apply_supervision_masks(
        self,
        batch: dict,
        output: DecoderOutput,
    ) -> None:
        target = batch["target"]
        mask = target.get("mask", None)
        if mask is None or not (
            self.train_cfg.mask_target or self.train_cfg.mask_output
        ):
            return

        mask_expanded = mask.expand_as(target["image"])
        background = (
            output.background_color
            if output.background_color is not None
            else torch.zeros_like(target["image"])
        )
        if self.train_cfg.mask_target:
            target["image"] = (
                target["image"] * mask_expanded
                + (1.0 - mask_expanded) * background
            )
        if self.train_cfg.mask_output:
            output.color = (
                output.color * mask_expanded
                + (1.0 - mask_expanded) * background
            )

    def _forward_training_batch(
        self,
        batch: dict,
        log_metrics: bool = True,
    ) -> tuple[EncoderOutput, DecoderOutput, Gaussians | None, torch.Tensor]:
        _, _, _, height, width = batch["target"]["image"].shape
        with self.benchmarker.time("encoder"):
            encoder_output: EncoderOutput = self.encoder(
                batch["context"],
                self.global_step,
                scene_names=batch["scene"],
            )
        encoder_output.context = batch["context"]
        gaussians = encoder_output.gaussians
        if gaussians is not None and log_metrics:
            self.log("train/gs_num", gaussians.means.shape[1], sync_dist=True)

        with self.benchmarker.time("decoder"):
            output: DecoderOutput = self.decoder.forward(
                encoder_output,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (height, width),
                depth_mode=self.train_cfg.depth_mode,
            )

        self._apply_supervision_masks(batch, output)
        total_loss = batch["target"]["image"].new_tensor(0.0)
        for loss_fn in self.losses:
            loss = loss_fn.forward(
                output,
                batch,
                gaussians,
                self.global_step,
            )
            if log_metrics:
                self.log(f"loss/{loss_fn.name}", loss, sync_dist=True)
            total_loss = total_loss + loss
        if log_metrics:
            self.log("loss/total", total_loss, sync_dist=True)
        return encoder_output, output, gaussians, total_loss

    def _save_training_vis(
        self,
        batch: dict,
        output: DecoderOutput,
        prefix: str,
    ) -> None:
        if self.global_rank != 0:
            return

        output_dir = Path(get_cfg()["output_dir"]) / "training_vis"
        step = self.global_step + 1
        context = batch["context"]["image"][0].detach().cpu()
        target = batch["target"]["image"][0].detach().cpu()
        prediction = output.color[0].detach().cpu()
        save_image(torch.cat(list(context), dim=-1), output_dir / f"{step:0>6}_{prefix}_context.png")
        save_image(torch.cat(list(target), dim=-1), output_dir / f"{step:0>6}_{prefix}_target.png")
        save_image(torch.cat(list(prediction), dim=-1), output_dir / f"{step:0>6}_{prefix}_prediction.png")

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self._train_step_start = time.time()

    def on_before_backward(self, loss) -> None:
        self._backward_start = time.time()
        return super().on_before_backward(loss)

    def on_after_backward(self) -> None:
        if self._backward_start is not None:
            self._last_backward_time = time.time() - self._backward_start
            self._backward_start = None
        return super().on_after_backward()

    def training_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        with self.benchmarker.time("forward"):
            _, output, _, total_loss = self._forward_training_batch(batch)

        psnr = compute_psnr(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        ).mean()
        self.log("train/psnr", psnr, sync_dist=True)
        self.log("info/global_step", self.global_step, sync_dist=True)

        if (
            self.global_rank == 0
            and (self.global_step + 1) % self.train_cfg.vis_every_n_steps == 0
        ):
            self._save_training_vis(batch, output, "train")

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        encoder_time = (
            self.benchmarker.execution_times["encoder"][-1]
            if self.benchmarker.execution_times["encoder"]
            else 0.0
        )
        decoder_time = (
            self.benchmarker.execution_times["decoder"][-1]
            if self.benchmarker.execution_times["decoder"]
            else 0.0
        )
        step_time = (
            time.time() - self._train_step_start
            if self._train_step_start is not None
            else 0.0
        )
        self._train_step_start = None
        self.log("train/encoder_time", encoder_time)
        self.log("train/decoder_time", decoder_time)
        self.log("train/backward_time", self._last_backward_time)
        self.log("train/step_time", step_time)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            lr = (
                self.trainer.optimizers[0].param_groups[0]["lr"]
                if self.trainer.optimizers
                else self.optimizer_cfg.lr
            )
            print(
                f"[Train Step {self.global_step:6d}] "
                f"loss={float(loss):.6f} lr={float(lr):.2e} "
                f"encoder={encoder_time:.3f}s decoder={decoder_time:.3f}s "
                f"backward={self._last_backward_time:.3f}s step={step_time:.3f}s",
                flush=True,
            )

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

    def validation_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        _, output, _, total_loss = self._forward_training_batch(
            batch,
            log_metrics=False,
        )

        self.log("val/loss", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        rgb_gt = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
        rgb_pred = rearrange(output.color, "b v c h w -> (b v) c h w")
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log("val/psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/ssim", ssim, on_step=False, on_epoch=True, sync_dist=True)

        if (
            self.global_rank == 0
            and (self.global_step + 1) % self.train_cfg.vis_every_n_steps == 0
        ):
            self._save_training_vis(batch, output, "val")

        if self.global_rank == 0 and batch_idx == 0:
            print(
                f"[Val Step {self.global_step:6d}] "
                f"loss={float(total_loss):.6f} psnr={float(psnr):.3f} "
                f"lpips={float(lpips):.3f} ssim={float(ssim):.3f}",
                flush=True,
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        total_steps = int(self.trainer.max_steps)
        if total_steps <= 0:
            total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.optimizer_cfg.lr,
            total_steps=max(1, total_steps + 10),
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def test_step(self, batch, batch_idx):
        batch = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1


        # Downsample context images and mask if needed
        if self.test_cfg.downsample_context_ratio != 1.0:
            h_target = int(h * self.test_cfg.downsample_context_ratio)
            w_target = int(w * self.test_cfg.downsample_context_ratio)

            context_img = batch["context"]["image"]
            b, v, c, h_c, w_c = context_img.shape
            context_img = context_img.reshape(b * v, c, h_c, w_c)
            context_img = F.interpolate(
                context_img,
                size=(h_target, w_target),
                mode="bilinear",
                align_corners=True,
            ).reshape(b, v, c, h_target, w_target)
            batch["context"]["image"] = context_img

            mask = batch["context"].get("mask", None)
            if mask is not None:
                # mask may be (b, v, 1, h, w) or (b, v, h, w)
                orig_shape = mask.shape
                if mask.ndim == 5:
                    mask = mask.reshape(b * v, orig_shape[2], orig_shape[3], orig_shape[4])
                    mask = F.interpolate(
                        mask,
                        size=(h_target, w_target),
                        mode="bilinear",
                        align_corners=True,
                    )
                    mask = mask.reshape(b, v, orig_shape[2], h_target, w_target)
                elif mask.ndim == 4:
                    mask = mask.reshape(b * v, 1, orig_shape[2], orig_shape[3])
                    mask = F.interpolate(
                        mask,
                        size=(h_target, w_target),
                        mode="bilinear",
                        align_corners=True,
                    )
                    mask = mask.reshape(b, v, h_target, w_target)
                batch["context"]["mask"] = mask

        scene_name = f'{batch["scene"][0]}_{batch["timestep"][0]}' if 'timestep' in batch else batch["scene"][0]

        # save input views for visualization
        if self.test_cfg.save_input_images:
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene_name / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        #######################
        # Encoder
        #######################

        with self.benchmarker.time("encoder"):
            encoder_output: EncoderOutput = self.encoder(
                batch["context"],
                self.global_step,
                # deterministic=False,
                visualization_dump=visualization_dump,
            )

        pred_depths = encoder_output.depth
        gaussians = encoder_output.gaussians
        encoder_output.context = batch["context"]

        #######################
        # Decoder
        #######################
        with self.benchmarker.time("decoder", num_calls=v):

            camera_poses = batch["target"]["extrinsics"]

            if self.test_cfg.stablize_camera:
                stable_poses = render_stabilization_path(
                    camera_poses[0].detach().cpu().numpy(),
                    k_size=self.test_cfg.stab_camera_kernel,
                )

                stable_poses = list(
                    map(
                        lambda x: np.concatenate(
                            (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                        ),
                        stable_poses,
                    )
                )
                stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                    camera_poses
                )
                camera_poses = stable_poses.unsqueeze(0)


            if self.test_cfg.render_chunk_size is not None:
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)

                    render_intrinsics = batch["target"]["intrinsics"]
                    render_near = batch["target"]["near"]
                    render_far = batch["target"]["far"]

                    curr_output: DecoderOutput = self.decoder.forward(
                        encoder_output,
                        camera_poses[:, start:end],
                        render_intrinsics[:, start:end],
                        render_near[:, start:end],
                        render_far[:, start:end],
                        (h, w),
                        depth_mode=None
                    )

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat(
                            (output.color, curr_output.color), dim=1
                        )

                    # Free intermediate chunk output
                    if i > 0:
                        del curr_output
            else:
                output: DecoderOutput = self.decoder.forward(
                    encoder_output,
                    camera_poses,
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

        # ============================================
        # Step 1: Optionally move inference results to CPU to prevent GPU memory accumulation
        # ============================================

        # Determine target device based on config
        target_device = 'cpu' if self.test_cfg.move_to_cpu else self.device

        # Move rendering results to target device
        images_prob = output.color[0].detach().to(target_device)  # [v, c, h, w]
        rgb_gt = batch["target"]["image"][0].detach().to(target_device)  # [v, c, h, w]

        if self.test_cfg.mask_output and batch["target"].get("mask", None) is not None:
            mask = batch["target"]["mask"][0].detach().to(target_device).expand_as(rgb_gt)
            background_color = torch.tensor(self.dataset_cfg.background_color, dtype=torch.float32, device=target_device)[None, :, None, None]
            rgb_gt = rgb_gt * mask + (1.0 -mask) * background_color
            images_prob = images_prob * mask + (1.0 - mask) * background_color
        else:
            mask = None

        # Move depth results to target device
        if self.test_cfg.save_depth:
            if pred_depths is not None:
                depth = pred_depths[0].detach().to(target_device)  # [V, H, W]
            else:
                depth = visualization_dump["depth"][0, :, :, :, 0, 0].detach().to(target_device)  # [V, H, W]

            context_index = batch["context"]["index"][0].to(target_device)

        # Store other needed info
        target_index = batch["target"]["index"][0].to(target_device)
        context_index_for_video = batch["context"]["index"][0].to(target_device)

        if encoder_output.gaussians_mask is not None:
            gaussians_num = (encoder_output.gaussians_mask > 0).sum().item()
        else:
            gaussians_num = gaussians.means.size(1)
        # save gaussians BEFORE clearing GPU (needs GPU tensors)
        if self.test_cfg.save_gaussian:
            path_for_gaussian = Path(get_cfg()["output_dir"])
            save_path = path_for_gaussian / 'gaussians' / (scene_name + '.ply')
            save_gaussians(gaussians, encoder_output.gaussians_mask, batch, save_path)

        # Clear GPU memory if data was moved to CPU
        if self.test_cfg.move_to_cpu:
            del encoder_output, gaussians, output
            if pred_depths is not None:
                del pred_depths
            if visualization_dump is not None:
                del visualization_dump
            torch.cuda.empty_cache()

        # ============================================
        # Step 2: Process data (on CPU or GPU depending on config)
        # ============================================

        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0].detach().cpu()  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.numpy() * 255).astype(np.uint8)  # [H, VW, 3]
                depth_concat = []

            for idx, depth_i in zip(context_index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene_name / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.numpy()
                    save_path = path / "images" / scene_name / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene_name / "depth" /  f"img_depth_{scene_name}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            # Clear depth data
            del depth
            if self.test_cfg.save_depth_concat_img:
                del image_concat, depth_concat, concat

        # Save renderings (all from CPU)
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(target_index, images_prob, rgb_gt):
                    save_image(color, path / "images" / scene_name / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene_name / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(target_index, images_prob):
                    save_image(color, path / "images" / scene_name / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in context_index_for_video])
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene_name}_frame_{frame_str}.mp4",
            )

        # compute scores (move data to GPU only when computing)
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if 'gs_num' not in self.test_step_outputs:
                self.test_step_outputs['gs_num'] = []
            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs['gs_num'].append(gaussians_num)
            if self.test_cfg.mask_output and batch["target"].get("mask", None) is not None:
                psnr_list = []
                ssim_list = []
                lpips_list = []
                for i in range(v):
                    mask_indices = mask[i].nonzero(as_tuple=False)
                    min_h = mask_indices[:, 1].min()
                    max_h = mask_indices[:, 1].max()
                    min_w = mask_indices[:, 2].min()
                    max_w = mask_indices[:, 2].max()
                    # Ensure data is on GPU for metric computation
                    rgb_i = images_prob[i, :, min_h:max_h, min_w:max_w].unsqueeze(0)
                    rgb_gt_i = rgb_gt[i, :, min_h:max_h, min_w:max_w].unsqueeze(0)
                    if target_device == 'cpu':
                        rgb_i = rgb_i.to(self.device)
                        rgb_gt_i = rgb_gt_i.to(self.device)

                    psnr_list.append(compute_psnr(rgb_gt_i, rgb_i).mean().item())
                    ssim_list.append(compute_ssim(rgb_gt_i, rgb_i).mean().item())
                    lpips_list.append(compute_lpips(rgb_gt_i, rgb_i).mean().item())

                self.test_step_outputs[f"psnr"].append(np.mean(psnr_list))
                self.test_step_outputs[f"ssim"].append(np.mean(ssim_list))
                self.test_step_outputs[f"lpips"].append(np.mean(lpips_list))
            else:
                rgb_gpu = images_prob.to(self.device)
                rgb_gt_gpu = rgb_gt.to(self.device)

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt_gpu, rgb_gpu).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt_gpu, rgb_gpu).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt_gpu, rgb_gpu).mean().item()
                )

        # Clean up
        if target_device == 'cpu':
            del images_prob, rgb_gt
            torch.cuda.empty_cache()

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()
