"""Standard MP-PCA denoising.

Marchenko-Pastur PCA over a moving box patch, assuming the noise is independent across
voxels.

Per masked voxel:

1. Take the patch box around it, giving ``X`` of shape ``(n_vox, n_vols)``. Rows are patch
   voxels, columns are diffusion volumes.
2. Find the Marchenko-Pastur edge in the eigenvalues of ``X``. No demeaning.
3. Drop everything below the edge, rebuild, and keep only the centre voxel's signal.

Centre-voxel only, with no averaging across the overlapping patches. The noise estimate is
a per-patch quantity, and averaging reconstructions would break the theory it rests on.

The core eigendecomposes the ``n_vols x n_vols`` Gram matrix instead of taking the SVD of
``X``. For the centre row that is exact rather than an approximation: ``x_centre @ V``
equals ``u[centre] * s``, so projecting the centre row onto the retained eigenvectors
gives what reconstructing the whole patch and reading one row out would give.

References: Veraart 2016, Cordero-Grande 2019, Ades-Aron 2020.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from ..patch import check_patch, patch_size
from ..pool import resolve_workers, run_slabs
from .gather import TILE, patch_tiles
from .mp_threshold import mp_threshold
from .result import DenoiseResult
from .shrink import RULES, check_shrink, shrink_weight
from .spectrum import eigh_descending

__all__ = ["denoise_mppca"]


def _denoise_tile(patches: np.ndarray, center: int, n_small: int, n_large: int,
                  algorithm: str, shrink: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Denoise a batch of patches, returning the centre voxel of each.

    Args:
        patches: (B, n_vox, n_vols) patch matrices.
        center: row index of the centre voxel.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algorithm: which Marchenko-Pastur variant to cut with.
        shrink: "hard" or "gd"; see :mod:`camppca.denoiser.shrink`.

    Returns:
        ``(signal, sigma_hat, rank)``: (B, n_vols) denoised centre spectra, (B,) fitted
        noise level, (B,) retained component count.
    """
    n_vox, n_vols = patches.shape[1], patches.shape[2]

    gram = np.matmul(patches.transpose(0, 2, 1), patches)
    evals, evecs = eigh_descending(gram)

    rank, sigma_hat = mp_threshold(evals, n_small, n_large, algorithm)

    coords = np.einsum("bn,bnm->bm", patches[:, center, :], evecs)
    keep = np.arange(n_vols)[None, :] < rank[:, None]
    kept = np.where(keep, coords, 0.0)
    if shrink != "hard":
        kept = kept * shrink_weight(evals, sigma_hat, n_vox, n_vols)
    signal = np.einsum("bm,bnm->bn", kept, evecs)
    return signal, sigma_hat, rank


def _denoise_arrays(dwi, mask, patch, algorithm, tile, shrink):
    """The tile loop of :func:`denoise_mppca`, in this process, on checked inputs.

    Returns:
        ``(denoised, sigma_hat, rank)``, full-size, with voxels outside ``mask`` left as
        the input and zero.
    """
    n_vox, n_vols = patch_size(patch), dwi.shape[3]
    center = n_vox // 2
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)

    denoised = dwi.copy()
    sigma_hat = np.zeros(dwi.shape[:3], np.float32)
    rank = np.zeros(dwi.shape[:3], np.int32)

    for coords, patches in patch_tiles(dwi, mask, patch, tile=tile):
        denoised[coords], sigma_hat[coords], rank[coords] = _denoise_tile(
            patches, center, n_small, n_large, algorithm, shrink)
    return denoised, sigma_hat, rank


def denoise_mppca(dwi: np.ndarray, mask: np.ndarray | None = None, patch=(5, 5, 5),
                  algorithm: str = "cordero-grande", tile: int = TILE,
                  shrink: str = "hard", workers: int | None = None) -> DenoiseResult:
    """Denoise a 4D volume with standard MP-PCA.

    Args:
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) boolean; only these voxels are denoised, the rest are passed
            through. Defaults to the whole volume.
        patch: box shape, three odd integers.
        algorithm: which Marchenko-Pastur variant to cut with, see
            :data:`~camppca.denoiser.mp_threshold.ALGORITHMS`.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        shrink: weight on the kept components, "hard" (default), "gd" or "lr"; see
            :mod:`camppca.denoiser.shrink`. With white noise "lr" is "gd".
        workers: pinned CPU processes to split the volume over, see
            :mod:`camppca.pool`. The default is one per physical core. 1 runs in this
            process.

    Returns:
        A :class:`~camppca.denoiser.result.DenoiseResult` with ``model`` unset, since standard
        MP-PCA does not use a noise model.

    Raises:
        ValueError: if ``dwi`` is not 4D, ``mask`` does not match its spatial shape,
            ``patch`` is not three odd positive integers, or ``shrink`` is unknown.
    """
    patch = check_patch(patch)
    shrink = check_shrink(shrink)
    tile_shrink = "gd" if shrink == "lr" else shrink   # C = I: the correction is exactly 1
    dwi = np.asarray(dwi, np.float32)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X, Y, Z, N), got shape {dwi.shape}")
    if mask is None:
        mask = np.ones(dwi.shape[:3], bool)
    else:
        mask = np.asarray(mask, bool)
        if mask.shape != dwi.shape[:3]:
            raise ValueError(
                f"mask shape {mask.shape} does not match dwi spatial shape {dwi.shape[:3]}")

    workers = resolve_workers(workers)

    if workers > 1:
        outputs = (dwi.copy(), np.zeros(dwi.shape[:3], np.float32),
                   np.zeros(dwi.shape[:3], np.int32))
        run_slabs(partial(_denoise_arrays, patch=patch, algorithm=algorithm, tile=tile,
                          shrink=tile_shrink),
                  dwi, mask, patch, outputs, workers, tile)
        denoised, sigma_hat, rank = outputs
    else:
        denoised, sigma_hat, rank = _denoise_arrays(dwi, mask, patch, algorithm, tile,
                                                    tile_shrink)

    return DenoiseResult(
        denoised=denoised, sigma_hat=sigma_hat, rank=rank,
        provenance=dict(method="mppca", patch=tuple(int(v) for v in patch),
                        algorithm=algorithm, shrink=shrink,
                        rule=f"{RULES[tile_shrink]}, centre-voxel only"))
