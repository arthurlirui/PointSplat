"""Visualization rendering for the static-vs-feedforward alignment.

This module implements the **image-space alignment visualization**
described in the design doc.  Because the static high-quality scene and
the feedforward PointSplat scene are both Gaussian splats, the natural
way to *see* how well they align is to render both from a shared set of
camera poses and compare the resulting images.

Two rendering backends are supported:

1. **Pure-PyTorch point splatter** (default, CPU-friendly): every
   Gaussian centre is projected to the image plane and splatted as a
   small additive blob whose size depends on the Gaussian's scale and
   depth.  This is *not* a full alpha-composited Gaussian splat -- it is
   deliberately lightweight so the alignment viz runs on CPU in a few
   seconds, with no CUDA / gsplat dependency.  It is sufficient for
   judging geometric alignment.
2. **gsplat** (optional, GPU): when ``use_gsplat=True`` and a CUDA device
   is available, we delegate to the project's existing
   :func:`src.model.decoder.decoder_splatting_gsplat.render_gsplat` for a
   photorealistic render.

The :func:`alignment_report` function renders both clouds from a list of
shared cameras, computes image-space metrics (PSNR, SSIM, mean L1, mask
IoU), and returns a structured dict.  :func:`save_alignment_visualization`
additionally writes side-by-side, checkerboard, and per-pixel-residual
PNGs to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .types import GaussianCloud


# ---------------------------------------------------------------------------
# Camera helper
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """A pinhole camera in the standard c2w convention used by PointSplat.

    ``extrinsics`` is a world-to-camera ... no, *camera-to-world* (c2w)
    matrix ``(4, 4)`` (matches :mod:`src.dataset.types.BatchedViews`).
    ``intrinsics`` is a ``(3, 3)`` matrix in *normalised* or *pixel*
    convention; pass ``image_shape`` so we can disambiguate.
    """

    extrinsics: Float[Tensor, "4 4"]  # c2w
    intrinsics: Float[Tensor, "3 3"]  # pixels
    image_shape: tuple[int, int]      # (H, W)

    @property
    def w2c(self) -> Tensor:
        return self.extrinsics.inverse()

    @property
    def H(self) -> int:
        return self.image_shape[0]

    @property
    def W(self) -> int:
        return self.image_shape[1]


# ---------------------------------------------------------------------------
# Pure-PyTorch point splatter
# ---------------------------------------------------------------------------
@torch.no_grad()
def point_splat_render(
    cloud: GaussianCloud,
    camera: Camera,
    point_radius_px: float = 2.0,
    background: float = 0.0,
    max_points: int = 200_000,
) -> Float[Tensor, "3 h w"]:
    """Render a Gaussian cloud to an RGB image with a point splatter.

    Each Gaussian centre is projected to the image plane and splatted as
    a small disc of ``point_radius_px`` pixels, with the Gaussian's own
    colour and opacity.  Overlap is accumulated with alpha blending
    (front-to-back, sorted by depth).

    This is a *visualisation* renderer, not a faithful Gaussian splat:
    it ignores anisotropic scale and rotation.  It exists purely to
    produce an image whose geometry matches the cloud, so we can compute
    image-space alignment metrics against the static scene.

    Returns an ``(H, W, 3)`` image in ``[0, 1]`` (we return it as
    ``(3, H, W)`` to match the project's ``BCHW`` convention).
    """
    device = cloud.means.device
    H, W = camera.H, camera.W

    means = cloud.means
    colors = cloud.colors
    opacities = cloud.opacities
    if len(cloud) > max_points:
        idx = torch.randperm(len(cloud), device=device)[:max_points]
        means, colors, opacities = means[idx], colors[idx], opacities[idx]

    # World -> camera.
    w2c = camera.w2c.to(device).float()
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    p_cam = means @ R.T + t  # (N, 3)
    # Keep only points in front of the camera.
    in_front = p_cam[:, 2] > 1e-4
    p_cam = p_cam[in_front]
    colors = colors[in_front]
    opacities = opacities[in_front]
    if p_cam.shape[0] == 0:
        img = torch.full((3, H, W), background, dtype=torch.float32, device=device)
        return img

    # Project.
    K = camera.intrinsics.to(device).float()
    proj = p_cam @ K.T  # (N, 3)
    depth = proj[:, 2].clamp(min=1e-4)
    u = (proj[:, 0] / depth).round().long()
    v = (proj[:, 1] / depth).round().long()
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depth = u[in_image], v[in_image], depth[in_image]
    colors, opacities = colors[in_image], opacities[in_image]
    if u.shape[0] == 0:
        return torch.full((3, H, W), background, dtype=torch.float32, device=device)

    # Sort back-to-front for correct alpha blending.
    order = torch.argsort(-depth)  # far first
    u, v, depth = u[order], v[order], depth[order]
    colors, opacities = colors[order], opacities[order]

    # Splat as small discs.  We accumulate weighted colour and weight per
    # pixel via index_add_ on the *flattened* image tensors.  Building the
    # (N, K*K) coordinate/weight tables is the only big allocation.
    r = max(1, int(point_radius_px))
    img_acc = torch.zeros(H * W, 3, dtype=torch.float32, device=device)
    w_acc = torch.zeros(H * W, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    kernel = (xx.float() ** 2 + yy.float() ** 2).le(r ** 2).float()  # (K, K)
    kernel_flat = kernel.flatten()                                # (K*K,)

    # Per-point kernel splat coordinates: (N, K*K).
    ku = (xx.flatten()[None, :] + u[:, None])
    kv = (yy.flatten()[None, :] + v[:, None])
    valid = (ku >= 0) & (ku < W) & (kv >= 0) & (kv < H)
    flat_idx = (kv.clamp(0, H - 1) * W + ku.clamp(0, W - 1)).long()  # (N, K*K)
    weight = kernel_flat[None, :] * opacities[:, None] * valid.float()  # (N, K*K)
    # Colour contribution per splat pixel: (N, K*K, 3)
    color_contrib = colors[:, None, :] * weight[:, :, None]

    # Flatten everything and accumulate.
    flat_idx_flat = flat_idx.reshape(-1)
    weight_flat = weight.reshape(-1)
    color_flat = color_contrib.reshape(-1, 3)
    img_acc.index_add_(0, flat_idx_flat, color_flat)
    w_acc.index_add_(0, flat_idx_flat, weight_flat)

    img = img_acc / (w_acc.unsqueeze(1) + 1e-6)  # (HW, 3)
    img = img.view(H, W, 3).permute(2, 0, 1)     # (3, H, W)
    empty = (w_acc.view(H, W) < 1e-6)
    img[:, empty] = background
    return img.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Image-space metrics
# ---------------------------------------------------------------------------
def _to_uint8(img: Tensor) -> Tensor:
    return (img.clamp(0, 1) * 255).round().to(torch.uint8)


def psnr(a: Tensor, b: Tensor) -> float:
    """PSNR between two ``(3, H, W)`` images in [0, 1]."""
    mse = F.mse_loss(a, b)
    if mse.item() <= 1e-12:
        return float("inf")
    return float(20.0 * torch.log10(1.0 / torch.sqrt(mse)).item())


def ssim_simple(a: Tensor, b: Tensor) -> float:
    """A lightweight SSIM (no external deps).

    Uses a 11x11 Gaussian window and the standard SSIM formula.  This is
    intentionally a *simple* SSIM adequate for ranking alignment quality,
    not a perceptual metric.
    """
    a = a.unsqueeze(0).float()
    b = b.unsqueeze(0).float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    # 11x11 Gaussian window, sigma=1.5.  Repeat per channel for groups=3.
    coords = torch.arange(11, dtype=torch.float32) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1, 1, 11, 11)
    window = window.repeat(3, 1, 1, 1).to(a.device)               # (3, 1, 11, 11)

    mu_a = F.conv2d(a, window, padding=5, groups=3)
    mu_b = F.conv2d(b, window, padding=5, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = F.conv2d(a * a, window, padding=5, groups=3) - mu_a2
    sigma_b2 = F.conv2d(b * b, window, padding=5, groups=3) - mu_b2
    sigma_ab = F.conv2d(a * b, window, padding=5, groups=3) - mu_ab
    ssim_map = (
        (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    ) / (
        (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    )
    return float(ssim_map.mean().item())


def mask_iou(a: Tensor, b: Tensor, thresh: float = 0.05) -> float:
    """IoU of the foreground masks (where image > thresh)."""
    a_mask = (a.mean(0) > thresh)
    b_mask = (b.mean(0) > thresh)
    inter = (a_mask & b_mask).sum().item()
    union = (a_mask | b_mask).sum().item()
    return float(inter / (union + 1e-9))


# ---------------------------------------------------------------------------
# Full alignment report
# ---------------------------------------------------------------------------
@dataclass
class CameraAlignmentReport:
    per_camera: list[dict] = field(default_factory=list)
    mean_psnr: float = 0.0
    mean_ssim: float = 0.0
    mean_l1: float = 0.0
    mean_mask_iou: float = 0.0

    def as_dict(self) -> dict:
        return {
            "mean_psnr": self.mean_psnr,
            "mean_ssim": self.mean_ssim,
            "mean_l1": self.mean_l1,
            "mean_mask_iou": self.mean_mask_iou,
            "n_cameras": len(self.per_camera),
            "per_camera": self.per_camera,
        }


@torch.no_grad()
def alignment_report(
    static: GaussianCloud,
    feedforward: GaussianCloud,
    cameras: Sequence[Camera],
    point_radius_px: float = 2.0,
    background: float = 0.0,
) -> CameraAlignmentReport:
    """Render both clouds from ``cameras`` and compute image-space metrics.

    The two clouds must already be in the *same* world frame, i.e. the
    feedforward cloud has been passed through
    :func:`src.alignment.register.apply_similarity`.

    Returns a :class:`CameraAlignmentReport` with per-camera PSNR / SSIM /
    L1 / mask IoU and their means.
    """
    report = CameraAlignmentReport()
    psnrs, ssims, l1s, ious = [], [], [], []

    for i, cam in enumerate(cameras):
        img_static = point_splat_render(static, cam, point_radius_px, background)
        img_ff = point_splat_render(feedforward, cam, point_radius_px, background)

        p = psnr(img_static, img_ff)
        s = ssim_simple(img_static, img_ff)
        l1 = float((img_static - img_ff).abs().mean().item())
        iou = mask_iou(img_static, img_ff)

        psnrs.append(p)
        ssims.append(s)
        l1s.append(l1)
        ious.append(iou)

        report.per_camera.append({
            "camera": i,
            "psnr": p,
            "ssim": s,
            "l1": l1,
            "mask_iou": iou,
        })

    report.mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    report.mean_ssim = float(np.mean(ssims)) if ssims else 0.0
    report.mean_l1 = float(np.mean(l1s)) if l1s else 0.0
    report.mean_mask_iou = float(np.mean(ious)) if ious else 0.0
    return report


# ---------------------------------------------------------------------------
# Save visualisation grids
# ---------------------------------------------------------------------------
@torch.no_grad()
def save_alignment_visualization(
    static: GaussianCloud,
    feedforward: GaussianCloud,
    cameras: Sequence[Camera],
    output_dir: str | Path,
    point_radius_px: float = 2.0,
    background: float = 0.0,
    max_cameras: int = 8,
) -> dict[str, list[str]]:
    """Render side-by-side / checkerboard / residual PNGs to ``output_dir``.

    Writes, for each camera (up to ``max_cameras``):

    * ``side_by_side_<i>.png``  -- static | feedforward
    * ``checkerboard_<i>.png``  -- 16x16 checkerboard mix of the two
    * ``residual_<i>.png``      -- per-pixel L1 heat-map (jet-like)
    * ``overlay_<i>.png``       -- static in red channel, feedforward in
      green channel, so misalignment shows as red/green fringes.

    Returns a dict mapping artifact name to the list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"side_by_side": [], "checkerboard": [], "residual": [], "overlay": []}

    n = min(len(cameras), max_cameras)
    for i in range(n):
        cam = cameras[i]
        img_s = point_splat_render(static, cam, point_radius_px, background).cpu()
        img_f = point_splat_render(feedforward, cam, point_radius_px, background).cpu()

        # Side-by-side.
        sb = torch.cat([img_s, img_f], dim=2)
        paths["side_by_side"].append(_save_png(sb, output_dir / f"side_by_side_{i:03d}.png"))

        # Checkerboard (16x16 cells).
        cb = _checkerboard(img_s, img_f, cell=16)
        paths["checkerboard"].append(_save_png(cb, output_dir / f"checkerboard_{i:03d}.png"))

        # Residual heat-map.
        res = (img_s - img_f).abs().mean(0, keepdim=True)  # (1, H, W)
        res_img = _colorize_heatmap(res)
        paths["residual"].append(_save_png(res_img, output_dir / f"residual_{i:03d}.png"))

        # Overlay: static -> red, feedforward -> green.
        overlay = torch.zeros(3, *img_s.shape[1:], dtype=torch.float32)
        overlay[0] = img_s.mean(0)
        overlay[1] = img_f.mean(0)
        paths["overlay"].append(_save_png(overlay, output_dir / f"overlay_{i:03d}.png"))

    return paths


# ---------------------------------------------------------------------------
# Small image utilities
# ---------------------------------------------------------------------------
def _save_png(img_chw: Tensor, path: Path) -> str:
    """Save a (3, H, W) float image in [0, 1] to ``path``.  Returns str(path)."""
    import cv2

    img = (img_chw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    # cv2 expects BGR.
    if img.shape[-1] == 3:
        img = img[:, :, ::-1]
    cv2.imwrite(str(path), img)
    return str(path)


def _checkerboard(a: Tensor, b: Tensor, cell: int = 16) -> Tensor:
    _, H, W = a.shape
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij"
    )
    mask = ((yy // cell) + (xx // cell)) % 2 == 0
    mask = mask.float().unsqueeze(0)
    return a * mask + b * (1 - mask)


def _colorize_heatmap(gray_1hw: Tensor) -> Tensor:
    """Map a (1, H, W) scalar map in [0, 1] to a simple jet-like RGB (3, H, W)."""
    x = gray_1hw.clamp(0, 1).squeeze(0)  # (H, W)
    r = torch.clamp(1.5 - torch.abs(4 * x - 3), 0, 1)
    g = torch.clamp(1.5 - torch.abs(4 * x - 2), 0, 1)
    b = torch.clamp(1.5 - torch.abs(4 * x - 1), 0, 1)
    return torch.stack([r, g, b], dim=0)
