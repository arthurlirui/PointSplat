"""Iterative optimization module for PointSplat.

This module provides per-scene iterative refinement of Gaussians predicted
by the feed-forward PointSplat encoder. After the initial single-pass
prediction, the Gaussians (or the latent features that produce them) are
optimized against the target views for a fixed number of steps.

Two optimization modes are supported:

1. **Latent optimization** (recommended): Freeze the encoder, optimize the
   per-scene latent tensor through the Gaussian decoder head. The latent
   space is compact and converges fast.

2. **Direct Gaussian optimization**: Detach the predicted Gaussians and
   optimize them directly as free parameters (classical 3DGS-style).

Usage
-----
::

    from src.modules.iterative_optim import IterativeOptimizer

    optimizer = IterativeOptimizer(
        encoder=encoder,
        decoder=decoder,
        mode="latent",
        num_steps=200,
        lr=1e-3,
    )
    refined_gaussians = optimizer.refine(batch)
"""

from .optimizer import IterativeOptimizer, GaussianOptimizer, LatentOptimizer

__all__ = [
    "IterativeOptimizer",
    "GaussianOptimizer",
    "LatentOptimizer",
]
