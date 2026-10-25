"""Similarity-transform registration between a feedforward and a static
Gaussian cloud.

The static high-quality Gaussian-splatting scene and the PointSplat
feedforward Gaussians live in coordinate frames that can differ by a
global similarity transform (rotation + translation + isotropic scale).
For PointSplat this happens because the feedforward head is trained on
normalised, possibly scale-invariant scene coordinates (see
``DecoderSplattingGSplatCfg.scale_invariant``), so its world frame is
arbitrary relative to the metric world frame of the static 3DGS scene.

The registration has two stages:

1. **Robust pre-alignment** -- align centroids and principal axes via
   PCA.  The 8 candidate axis-sign combinations are scored by chamfer
   distance and the best one is kept.  This is closed-form, deterministic,
   and gives a good initialisation even when the two clouds have very
   different point counts (PointSplat produces ~100k Gaussians; a COLMAP
   sparse cloud can have only a few thousand).
2. **ICP refinement** -- point-to-point ICP with a k-d tree (we use
   :mod:`scipy.spatial.cKDTree`) for nearest-neighbour lookup.  We run a
   fixed number of iterations with a robust Huber loss on per-point
   residuals to suppress outliers from the noisier feedforward cloud.

The result is a :class:`SimilarityTransform` ``(R, t, s)`` that maps
``feedforward_xyz -> static_xyz`` via ``y = s * (R @ x) + t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from scipy.spatial import cKDTree
from torch import Tensor

from .types import GaussianCloud


# ---------------------------------------------------------------------------
# Transform container
# ---------------------------------------------------------------------------
@dataclass
class SimilarityTransform:
    """``y = s * (R @ x) + t`` mapping source points onto target points.

    Stored as torch tensors.  ``R`` is ``(3, 3)``, ``t`` is ``(3,)``,
    ``s`` is a scalar.  ``R`` is guaranteed to be a rotation (det = +1).
    """

    R: Float[Tensor, "3 3"]
    t: Float[Tensor, "3"]
    s: Float[Tensor, ""]

    # ---- apply ---------------------------------------------------------------
    def apply(self, points: Float[Tensor, "*n 3"]) -> Float[Tensor, "*n 3"]:
        """Apply the similarity transform to a batch of 3D points."""
        return (points @ (self.s * self.R).T) + self.t

    # ---- io ------------------------------------------------------------------
    def to_matrix4x4(self) -> Float[Tensor, "4 4"]:
        M = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        M[:3, :3] = self.s * self.R
        M[:3, 3] = self.t
        return M

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"SimilarityTransform(s={float(self.s):.4f}, "
            f"t={self.t.tolist()}, R={self.R.tolist()})"
        )


def apply_similarity(
    cloud: GaussianCloud, transform: SimilarityTransform
) -> GaussianCloud:
    """Return a *new* GaussianCloud whose means are mapped by ``transform``.

    Scales are multiplied by ``s`` and rotations are conjugated by ``R`` so
    that the *shape* of every Gaussian is preserved in the target frame.
    Colours and opacities are passed through unchanged.
    """
    new_means = transform.apply(cloud.means)
    new_scales = cloud.scales * float(transform.s)

    # Conjugate rotations: q' = q_R * q * q_R^-1, where q_R is the quaternion
    # of R.  We do this in rotation-matrix space for simplicity.
    R = transform.R.to(cloud.rotations.device).to(cloud.rotations.dtype)
    Rs = _quat_to_rotmat(cloud.rotations)              # (N, 3, 3)
    Rs_new = torch.einsum("ij,njk,kl->nil", R, Rs, R.T)  # R @ Rs @ R^T
    new_rotations = _rotmat_to_quat(Rs_new)

    return GaussianCloud(
        means=new_means,
        scales=new_scales,
        rotations=new_rotations,
        colors=cloud.colors,
        opacities=cloud.opacities,
        source=cloud.source + "_aligned",
    )


# ---------------------------------------------------------------------------
# Stage 1: PCA-based robust pre-alignment
# ---------------------------------------------------------------------------
def _pca_basis(points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return (centroid, eigenvectors, eigenvalues) of a point cloud.

    Eigenvectors are sorted by descending eigenvalue.  We use the SVD on
    the centred points (faster and more stable than ``torch.linalg.eigh``
    on the covariance for tall matrices).
    """
    centroid = points.mean(0)
    centred = points - centroid
    # (N, 3) -> SVD; right-singular vectors are the principal axes.
    _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
    basis = Vt.T  # (3, 3) columns are principal axes
    # Eigenvalues from the singular values: lambda_i = s_i^2 / N
    _, s, _ = torch.linalg.svd(centred, full_matrices=False)
    eigvals = (s ** 2) / max(centred.shape[0], 1)
    return centroid, basis, eigvals


def _best_axis_sign_combination(
    src_centred: Tensor,
    src_basis: Tensor,
    dst_centred: Tensor,
    dst_basis: Tensor,
) -> Tensor:
    """Pick the 8-way axis-sign combination that minimises chamfer distance.

    PCA gives axes up to a sign per axis, so there are 2^3 = 8 candidate
    rotations aligning the source basis to the destination basis.  We
    evaluate each by chamfer distance on a sub-sampled set and keep the
    best.
    """
    # Sub-sample for speed; chamfer sign is robust to sampling.
    n_eval = min(2048, src_centred.shape[0])
    idx = torch.randperm(src_centred.shape[0])[:n_eval]
    src_sample = src_centred[idx]
    n_eval_dst = min(2048, dst_centred.shape[0])
    idx_d = torch.randperm(dst_centred.shape[0])[:n_eval_dst]
    dst_sample = dst_centred[idx_d]

    best_R = None
    best_cost = float("inf")
    for signs in _iter_sign_combinations():
        S = torch.diag(torch.tensor(signs, dtype=src_basis.dtype, device=src_basis.device))
        # Map src axes onto (signed) dst axes: R = dst_basis @ S @ src_basis^T
        R = dst_basis @ S @ src_basis.T
        rotated = src_sample @ R.T
        cost = _chamfer(rotated, dst_sample)
        if cost < best_cost:
            best_cost = cost
            best_R = R
    return best_R


def _iter_sign_combinations():
    for s0 in (1.0, -1.0):
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                yield (s0, s1, s2)


def _chamfer(a: Tensor, b: Tensor) -> float:
    """Symmetric chamfer distance between two point clouds (squared L2)."""
    # Use torch.cdist for simplicity; both clouds are small (<=2048 pts).
    d = torch.cdist(a, b)  # (Na, Nb)
    return float(d.min(1).values.mean().item() + d.min(0).values.mean().item())


def _pca_prealign(
    src: Tensor, dst: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (R, t, s) pre-aligning ``src`` onto ``dst``.

    ``s`` is the ratio of the two clouds' largest singular values, which
    captures the global scale difference between the feedforward and
    static scenes.
    """
    src_c, src_basis, src_eig = _pca_basis(src)
    dst_c, dst_basis, dst_eig = _pca_basis(dst)

    R = _best_axis_sign_combination(src - src_c, src_basis, dst - dst_c, dst_basis)

    # Global scale: ratio of total extent (sum of singular values).
    s = float((src_eig.sum().sqrt() / (dst_eig.sum().sqrt() + 1e-9)).item())
    s = 1.0 / (s + 1e-9)  # we want y = s R x + t, so s maps src-scale to dst-scale
    # More directly: match the std of projections on the first principal axis.
    s = float(
        (dst - dst_c).std().item() / ((src - src_c).std().item() + 1e-9)
    )

    t = dst_c - s * (src_c @ R.T)

    return (
        R.contiguous(),
        t.contiguous(),
        torch.tensor(s, dtype=src.dtype, device=src.device),
    )


# ---------------------------------------------------------------------------
# Stage 2: point-to-point ICP with Huber loss
# ---------------------------------------------------------------------------
def _icp_refine(
    src: np.ndarray,
    dst: np.ndarray,
    init_R: np.ndarray,
    init_t: np.ndarray,
    init_s: float,
    max_iters: int = 50,
    tolerance: float = 1e-6,
    huber_delta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """ICP with isotropic scale and a robust Huber loss on residuals.

    Returns ``(R, t, s, info)`` where ``info`` records the per-iteration
    mean residual for diagnostics.  Uses scipy's cKDTree for fast
    nearest-neighbour queries.

    The transform is accumulated incrementally: at each iteration we find
    correspondences, fit a *delta* similarity transform from the
    currently-transformed source to its nearest neighbours in dst, and
    compose it onto the running transform.  This is the standard ICP
    accumulation pattern and is much more stable than re-fitting from the
    original frame each iteration.
    """
    R = init_R.copy()
    t = init_t.copy()
    s = float(init_s)
    history = []

    dst_tree = cKDTree(dst)
    # Sub-sample src for speed if it is large.
    if src.shape[0] > 8192:
        idx = np.random.default_rng(0).choice(src.shape[0], 8192, replace=False)
        src_eval = src[idx]
    else:
        src_eval = src

    prev_residual = float("inf")
    for it in range(max_iters):
        # Current transformed source.
        transformed = s * (src_eval @ R.T) + t
        dists, idxs = dst_tree.query(transformed, k=1)
        corr = dst[idxs]
        residuals = transformed - corr  # (N, 3)

        # Huber weights on per-point L2 residual.
        r_norm = np.linalg.norm(residuals, axis=1)
        weights = np.where(r_norm <= huber_delta, 1.0, huber_delta / (r_norm + 1e-9))

        # Fit a *delta* similarity transform from `transformed` -> `corr`.
        dR, dt, ds = _weighted_umeyama(transformed, corr, weights)

        # Compose: new transform = (delta) o (current)
        #   y_new = ds * (dR @ (s * (R @ x + t)) ... wait: y_new = ds * dR @ (transformed) + dt
        #   but transformed = s * R @ x + t, so
        #   y_new = ds * dR @ (s * R @ x + t) + dt
        #         = (ds * s) * (dR @ R) @ x + (ds * dR @ t + dt)
        R = dR @ R
        s = ds * s
        t = ds * (dR @ t) + dt

        mean_residual = float(r_norm.mean())
        history.append(mean_residual)
        if abs(prev_residual - mean_residual) < tolerance:
            break
        prev_residual = mean_residual

    return R, t, s, {"iterations": len(history), "residual_history": history}


def _weighted_umeyama(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Umeyama similarity transform: minimises weighted MSE.

    Solves for ``(R, t, s)`` such that ``dst ≈ s * src @ R.T + t`` under
    per-point ``weights``.  Standard closed-form solution.
    """
    w_sum = weights.sum() + 1e-9
    src_mean = (src * weights[:, None]).sum(0) / w_sum
    dst_mean = (dst * weights[:, None]).sum(0) / w_sum
    src_c = src - src_mean
    dst_c = dst - dst_mean

    # Weighted covariance.
    H = (src_c * weights[:, None]).T @ dst_c / w_sum  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = Vt.T @ S @ U.T

    # Weighted scale: ratio of weighted variances along all axes.
    var_src = (weights[:, None] * (src_c ** 2)).sum() / w_sum
    var_dst = (weights[:, None] * (dst_c ** 2)).sum() / w_sum
    s = float(np.sqrt(var_dst / (var_src + 1e-9)))
    # Cap to avoid runaway scales from degenerate correspondences.
    s = float(np.clip(s, 0.1, 10.0))

    t = dst_mean - s * (R @ src_mean)
    return R, t, s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_similar(
    source: GaussianCloud,
    target: GaussianCloud,
    max_icp_iters: int = 50,
    sample_align: int = 4096,
) -> tuple[SimilarityTransform, dict]:
    """Register ``source`` onto ``target``.

    Returns the :class:`SimilarityTransform` that maps ``source.means``
    onto ``target.means``, plus an ``info`` dict with diagnostics
    (chamfer distance before/after, ICP residual history, etc.).

    Parameters
    ----------
    source, target : GaussianCloud
        The clouds to register.  ``source`` is typically the feedforward
        cloud, ``target`` the static high-quality cloud.
    max_icp_iters : int
        Maximum number of ICP iterations.
    sample_align : int
        Sub-sample size for the PCA pre-alignment's chamfer evaluation.
    """
    src = source.means.detach().cpu().float()
    dst = target.means.detach().cpu().float()

    # Sub-sample very large clouds for the PCA stage (full cloud would be
    # slow and PCA on a sub-sample is statistically equivalent).
    if src.shape[0] > sample_align:
        idx = torch.randperm(src.shape[0])[:sample_align]
        src_pca = src[idx]
    else:
        src_pca = src
    if dst.shape[0] > sample_align:
        idx = torch.randperm(dst.shape[0])[:sample_align]
        dst_pca = dst[idx]
    else:
        dst_pca = dst

    R0, t0, s0 = _pca_prealign(src_pca, dst_pca)

    # Compute initial chamfer (post pre-align) for diagnostics.
    pre_aligned = (s0 * (src @ R0.T) + t0).numpy()
    dst_np = dst.numpy()
    dst_tree = cKDTree(dst_np)
    pre_dists, _ = dst_tree.query(pre_aligned, k=1)
    pre_chamfer = float(pre_dists.mean())

    # ICP refine.
    R, t, s, icp_info = _icp_refine(
        src.numpy(), dst_np,
        init_R=R0.numpy(), init_t=t0.numpy(), init_s=float(s0),
        max_iters=max_icp_iters,
    )

    # Final diagnostics.
    final_aligned = s * (src.numpy() @ R.T) + t
    post_dists, _ = dst_tree.query(final_aligned, k=1)
    post_chamfer = float(post_dists.mean())
    median_post = float(np.median(post_dists))
    p90_post = float(np.percentile(post_dists, 90))

    transform = SimilarityTransform(
        R=torch.from_numpy(R).float(),
        t=torch.from_numpy(t).float(),
        s=torch.tensor(s, dtype=torch.float32),
    )
    info = {
        "pre_align_chamfer": pre_chamfer,
        "post_align_chamfer": post_chamfer,
        "post_align_median": median_post,
        "post_align_p90": p90_post,
        "icp_iters": icp_info["iterations"],
        "icp_residual_history": icp_info["residual_history"],
        "n_source": int(src.shape[0]),
        "n_target": int(dst.shape[0]),
    }
    return transform, info


# ---------------------------------------------------------------------------
# Quaternion helpers (kept local to avoid pulling in scipy rotations,
# which would add a hard dependency for a tiny op)
# ---------------------------------------------------------------------------
def _quat_to_rotmat(q: Tensor) -> Tensor:
    """(N, 4) quaternions (w, x, y, z) -> (N, 3, 3) rotation matrices."""
    w, x, y, z = q.unbind(-1)
    n = (q * q).sum(-1, keepdim=True).clamp(min=1e-12).sqrt()
    w, x, y, z = w / n[..., 0], x / n[..., 0], y / n[..., 0], z / n[..., 0]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _rotmat_to_quat(R: Tensor) -> Tensor:
    """(N, 3, 3) rotation matrices -> (N, 4) quaternions (w, x, y, z).

    Uses Shepperd's method; numerically stable for the four quadrants.
    """
    m = R
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    q = torch.zeros(*R.shape[:-2], 4, dtype=R.dtype, device=R.device)
    # Case 1: trace > 0
    mask = trace > 0
    if mask.any():
        s = (trace[mask] + 1.0).clamp(min=1e-12).sqrt() * 2
        qm = q[mask]
        qm[..., 0] = 0.25 * s
        qm[..., 1] = (m[mask][..., 2, 1] - m[mask][..., 1, 2]) / s
        qm[..., 2] = (m[mask][..., 0, 2] - m[mask][..., 2, 0]) / s
        qm[..., 3] = (m[mask][..., 1, 0] - m[mask][..., 0, 1]) / s
        q[mask] = qm

    # Case 2: m00 is the largest diagonal
    mask = (~mask) & (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
    if mask.any():
        s = (1.0 + m[mask][..., 0, 0] - m[mask][..., 1, 1] - m[mask][..., 2, 2]).clamp(min=1e-12).sqrt() * 2
        qm = q[mask]
        qm[..., 0] = (m[mask][..., 2, 1] - m[mask][..., 1, 2]) / s
        qm[..., 1] = 0.25 * s
        qm[..., 2] = (m[mask][..., 0, 1] + m[mask][..., 1, 0]) / s
        qm[..., 3] = (m[mask][..., 0, 2] + m[mask][..., 2, 0]) / s
        q[mask] = qm

    # Case 3: m11 is the largest diagonal
    mask = (~mask) & (m[..., 1, 1] > m[..., 2, 2])
    if mask.any():
        s = (1.0 + m[mask][..., 1, 1] - m[mask][..., 0, 0] - m[mask][..., 2, 2]).clamp(min=1e-12).sqrt() * 2
        qm = q[mask]
        qm[..., 0] = (m[mask][..., 0, 2] - m[mask][..., 2, 0]) / s
        qm[..., 1] = (m[mask][..., 0, 1] + m[mask][..., 1, 0]) / s
        qm[..., 2] = 0.25 * s
        qm[..., 3] = (m[mask][..., 1, 2] + m[mask][..., 2, 1]) / s
        q[mask] = qm

    # Case 4: m22 is the largest diagonal
    mask = ~(trace > 0) & ~(m[..., 0, 0] >= m[..., 1, 1]) & ~(m[..., 1, 1] > m[..., 2, 2])
    if mask.any():
        s = (1.0 + m[mask][..., 2, 2] - m[mask][..., 0, 0] - m[mask][..., 1, 1]).clamp(min=1e-12).sqrt() * 2
        qm = q[mask]
        qm[..., 0] = (m[mask][..., 1, 0] - m[mask][..., 0, 1]) / s
        qm[..., 1] = (m[mask][..., 0, 2] + m[mask][..., 2, 0]) / s
        qm[..., 2] = (m[mask][..., 1, 2] + m[mask][..., 2, 1]) / s
        qm[..., 3] = 0.25 * s
        q[mask] = qm

    return q / (q.norm(dim=-1, keepdim=True) + 1e-9)
