"""Standard MPPCA denoising.

Marchenko-Pastur PCA over a moving box patch, assuming the noise is independent across
voxels.

Per masked voxel:

1. Take the patch box around it, giving ``X`` of shape ``(n_vox, n_vols)``. Rows are patch
   voxels, columns are diffusion volumes.
2. Find the Marchenko-Pastur edge in the eigenvalues of ``X``. No demeaning.
3. Drop everything below the edge, rebuild, and keep only the center voxel's signal.

Complex data are denoised as complex: the Gram matrix is ``X^H X`` and the rebuild uses the
conjugated eigenvectors, both no-ops on real data. The Marchenko-Pastur law is the same,
with the variance of a complex entry, twice that of one channel. So ``sigma_mppca`` is divided
by sqrt(2) on the way out and stays the noise level of one channel.

Center-voxel only by default, with no averaging across the overlapping patches. The noise
estimate is a per-patch quantity, and averaging reconstructions departs from the theory it
rests on. ``aggregate="mean"`` averages them anyway, as an option; see
:mod:`camppca.denoiser.aggregate`.

The core eigendecomposes the ``n_vols x n_vols`` Gram matrix instead of taking the SVD of
``X``. For the center row that is exact rather than an approximation: ``x_center @ V``
equals ``u[center] * s``, so projecting the center row onto the retained eigenvectors
gives what reconstructing the whole patch and reading one row out would give.

It runs the covariance-aware engine with no patch correlation and no relative noise map, which is
exactly this method: see :func:`~camppca.denoiser.camppca.denoise_whitened`.

References: Veraart 2016, Cordero-Grande 2019, Ades-Aron 2020.
"""

from __future__ import annotations

import numpy as np

from .. import __version__
from ..patch import resolve_patch
from .aggregate import check_aggregate
from .camppca import denoise_whitened
from .gather import as_data, check_dwi
from .result import DenoiseResult
from .shrink import check_shrink, rule_text

__all__ = ["denoise_mppca"]


def denoise_mppca(dwi: np.ndarray, mask: np.ndarray | None = None, *, patch=None,
                  algorithm: str = "cordero-grande", shrink: str = "hard",
                  workers: int | None = None, aggregate: str = "center",
                  stride: int = 1) -> DenoiseResult:
    """Denoise a 4D volume with standard MPPCA.

    Args:
        dwi: (X, Y, Z, N) data, real or complex. Complex data are denoised as complex.
        mask: (X, Y, Z) boolean; only these voxels are denoised, the rest are passed
            through. Defaults to the whole volume. In camppca the mask says where the
            noise model is estimated, and every voxel is denoised unless ``mask_only``.
            Comparing the two on one region takes ``mask_only=True`` there.
        patch: box shape, three odd integers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        algorithm: which Marchenko-Pastur variant to cut with, see
            :data:`~camppca.denoiser.mp_threshold.ALGORITHMS`.
        shrink: weight on the kept components, "hard" (default), "gd" or "lr"; see
            :mod:`camppca.denoiser.shrink`. With white noise "lr" is "gd".
        workers: CPU processes to split the voxels over, see
            :mod:`camppca.pool`. The default is one per physical core. 1 runs in this
            process.
        aggregate: "center" (default) writes back each patch's center voxel; "mean"
            averages the overlapping patch reconstructions, see
            :mod:`camppca.denoiser.aggregate`.
        stride: with "mean", a patch center every ``stride`` voxels per axis.

    Returns:
        A :class:`~camppca.denoiser.result.DenoiseResult` with ``model`` unset, since standard
        MPPCA does not use a noise model.

    Raises:
        ValueError: if ``dwi`` is not 4D, ``mask`` does not match its spatial shape,
            ``patch`` is not three odd positive integers, or ``shrink`` is unknown.
    """
    shrink = check_shrink(shrink)
    aggregate, stride = check_aggregate(aggregate, stride)
    tile_shrink = "gd" if shrink == "lr" else shrink   # C = I: the correction is exactly 1
    dwi = as_data(check_dwi(dwi))
    patch = resolve_patch(patch, dwi.shape)
    if mask is None:
        mask = np.ones(dwi.shape[:3], bool)
    else:
        mask = np.asarray(mask, bool)
        check_dwi(dwi, mask.shape)

    denoised, sigma_mppca, rank = denoise_whitened(
        dwi, mask, None, patch=patch, algorithm=algorithm, shrink=tile_shrink,
        workers=workers, aggregate=aggregate, stride=stride)
    return DenoiseResult(
        denoised=denoised, sigma=sigma_mppca, rank=rank,
        provenance=dict(method="mppca", version=__version__,
                        patch=tuple(int(v) for v in patch),
                        algorithm=algorithm, shrink=shrink, aggregate=aggregate,
                        stride=stride, rule=rule_text(algorithm, tile_shrink, aggregate, stride)))
