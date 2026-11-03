"""Covariance-aware MP-PCA.

Whitens each patch with ``C^(-1/2)``, runs one Marchenko-Pastur fit on the whitened patch,
and unwhitens. Standard MP-PCA assumes the patch noise is i.i.d. across voxels; here it is
``Sigma = sigma^2 D C D``, so the fit happens where that assumption holds::

    Y[:, n] = signal + eps_n,   Cov(eps_n) = sigma^2 C
    Yw[:, n] = C^(-1/2) Y[:, n]        ->   Cov = sigma^2 I

UNWHITENING THE CENTRE VOXEL. Whitening mixes voxels, so the centre voxel of the
reconstruction is a dense combination of all ``n_vox`` whitened rows::

    xhat_c = (e_c^T C^(1/2)) Yw Pi

Only the CENTRE ROW of ``C^(1/2)`` is ever needed, which is what keeps unwhitening cheap.

WHITENER FACTORIZATION. The paper applies ``W = (DCD)^(-1/2)`` in one symmetric step. This
code divides rows by ``d`` and then applies ``C^(-1/2)``, i.e. ``W = C^(-1/2) D^(-1)``.
``W^T W = (DCD)^(-1)`` either way, so the Gram, the spectrum, the threshold, the retained
subspace and the reconstruction are all identical. Do NOT "fix" one to match the other.

WHERE C COMES FROM. :meth:`~camppca.noise_model.NoiseModel.patch_correlation` derives it,
because it belongs to the model.

PATCH ORDER. C is built from explicit (dx, dy, dz) lags in C order with z fastest, not as a
Kronecker product, so it is Toeplitz in the true relative lags with no wraparound.
"""

from functools import partial

import numpy as np

from ..patch import check_patch, patch_size
from ..pool import resolve_workers, run_slabs
from .gather import TILE, patch_tiles
from .mp_threshold import mp_threshold
from .shrink import shrink_weight, signal_alignment
from .spectrum import eigh_descending, eigvalsh_descending


def sqrtm_psd(A, p=0.5, floor=1e-10):
    """Matrix power of a symmetric PSD matrix, with the spectrum floored before raising."""
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor * max(w.max(), 1e-30))
    B = (V * w ** p) @ V.T
    return 0.5 * (B + B.T)


def _whiten_split(patches, whitener, n_small: int, n_large: int, algo: str):
    """Whiten a batch of patches and split each spectrum at the Marchenko-Pastur edge.

    Shared with the loop kernel estimator, so both split a patch with the same code. It
    runs in the dtype of ``patches``: float32 here, float64 in the loop.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algo: which Marchenko-Pastur variant to cut with.

    Returns:
        ``(whitened, evals, evecs, rank, sigma_hat)``: (B, n_vox, n_vols) whitened
        patches, (B, n_vols) descending Gram eigenvalues, (B, n_vols, n_vols) their
        eigenvectors as columns in the same order, (B,) retained component count, (B,)
        fitted noise level.
    """
    whitened = np.matmul(whitener, patches)
    gram = np.matmul(whitened.transpose(0, 2, 1), whitened)
    evals, evecs = eigh_descending(gram)
    rank, sigma_hat = mp_threshold(evals, n_small, n_large, algo)
    return whitened, evals, evecs, rank, sigma_hat


def _denoise_tile(patches, whitener, centre_row, C, n_small: int, n_large: int, algo: str,
                  shrink: str):
    """Denoise a batch of whitened patches, returning the centre voxel of each.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``.
        centre_row: (n_vox,) the centre row of ``C^(1/2)``, all that unwhitening needs.
        C: (n_vox, n_vox) the patch correlation itself; only "lr" reads it.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algo: which Marchenko-Pastur variant to cut with.
        shrink: "hard", "gd" or "lr"; see :mod:`camppca.denoiser.shrink`.

    Returns:
        ``(signal, sigma_hat, rank)``: (B, n_vols) denoised centre spectra in whitened
        units, (B,) fitted noise level, (B,) retained component count.
    """
    n_vox, n_vols = patches.shape[1], patches.shape[2]

    whitened, evals, evecs, rank, sigma_hat = _whiten_split(
        patches, whitener, n_small, n_large, algo)

    centre = np.einsum("p,bpm->bm", centre_row, whitened)    # e_c^T C^(1/2) Yw
    coords = np.einsum("bm,bmk->bk", centre, evecs)
    keep = np.arange(n_vols)[None, :] < rank[:, None]
    kept = np.where(keep, coords, 0.0)
    if shrink != "hard":
        alignment = signal_alignment(whitened, evecs, evals, C) if shrink == "lr" else None
        kept = kept * shrink_weight(evals, sigma_hat, n_vox, n_vols, alignment)
    signal = np.einsum("bk,bnk->bn", kept, evecs)
    return signal, sigma_hat, rank


def _with_level_column(dwi, d_field):
    """The level field as one extra column, wrap-padded and gathered with the data."""
    if d_field is None:
        return dwi
    return np.concatenate([dwi, np.asarray(d_field, np.float32)[..., None]], axis=3)


def whitened_level(dwi, mask, C, patch=(5, 5, 5), algo="cordero-grande", tile=TILE,
                   d_field=None, workers=None):
    """The MP noise level of every mask voxel's whitened patch, without denoising.

    The split :func:`denoise_whitened` makes, stopped once ``sigma_hat`` is known: only the
    eigenvalues are computed and nothing is reconstructed. With ``C`` None and no
    ``d_field`` this is standard MP-PCA's ``sigma_hat``.

    Args:
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) boolean.
        C: (n_vox, n_vox) patch correlation with unit diagonal, or None for white noise,
            which skips the whitening multiply.
        patch: box shape, three odd integers.
        algo: which Marchenko-Pastur variant to cut with.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        d_field: (X, Y, Z) level field, already guarded, or None.
        workers: pinned CPU processes to split the volume over, see :mod:`camppca.pool`.

    Returns:
        (X, Y, Z) ``sigma_hat`` in d-normalized units, zero outside ``mask``.
    """
    dwi = np.asarray(dwi, np.float32)
    mask = np.asarray(mask, bool)
    patch = check_patch(patch)
    workers = resolve_workers(workers)
    if workers > 1:
        return run_slabs(partial(whitened_level, C=C, patch=patch, algo=algo, workers=1),
                         dwi, mask, patch, (np.zeros(dwi.shape[:3], np.float32),),
                         workers, tile, d_field=d_field)[0]
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    whitener = None if C is None else sqrtm_psd(
        np.asarray(C, np.float64), -0.5).astype(np.float32)

    level = np.zeros(dwi.shape[:3], np.float32)
    for coords, block in patch_tiles(_with_level_column(dwi, d_field), mask, patch,
                                     tile=tile):
        patches = block if d_field is None else block[:, :, :n_vols] / block[:, :, n_vols:]
        if whitener is not None:
            patches = np.matmul(whitener, patches)
        evals = eigvalsh_descending(np.matmul(patches.transpose(0, 2, 1), patches))
        level[coords] = mp_threshold(evals, n_small, n_large, algo)[1]
    return level


def denoise_whitened(dwi, mask, C, patch=(5, 5, 5), algo="cordero-grande", tile=TILE,
                     d_field=None, shrink="hard", workers=None):
    """MP-PCA on the patch whitened by ``C^(-1/2)``.

    Args:
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) boolean; only these voxels are denoised.
        C: (n_vox, n_vox) patch correlation with unit diagonal.
        patch: box shape, three odd integers.
        algo: which Marchenko-Pastur variant to cut with.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        d_field: (X, Y, Z) level field, ALREADY guarded by the caller
            (:func:`~camppca.denoiser._effective_d`). Each voxel row is divided by its own level before whitening, which
            is exact for ``Sigma = sigma^2 D C D``, and the centre row is multiplied back
            on the way out. What it adds over a per-patch sigma is the within-patch
            heteroscedasticity, which widens the noise bulk the same way off-diagonal
            colour does.
        shrink: weight on the kept components; see :mod:`camppca.denoiser.shrink`.
        workers: pinned CPU processes to split the volume over, see :mod:`camppca.pool`.

    Returns:
        ``(denoised, sigma_hat, rank)``. ``sigma_hat`` is in d-normalized units; the
        caller multiplies by ``d`` to get data units.
    """
    dwi = np.asarray(dwi, np.float32)
    mask = np.asarray(mask, bool)          # non-bool would make `& mask` an integer op
    patch = check_patch(patch)             # odd dims: centre = n_vox // 2 must BE the centre
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    center = n_vox // 2
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)

    C = np.asarray(C, np.float64)
    if C.shape != (n_vox, n_vox):
        raise ValueError(f"C must be ({n_vox}, {n_vox}) for patch "
                         f"{tuple(int(v) for v in patch)}, got {C.shape}")

    workers = resolve_workers(workers)
    if workers > 1:
        outputs = (dwi.copy(), np.zeros(dwi.shape[:3], np.float32),
                   np.zeros(dwi.shape[:3], np.int32))
        return run_slabs(partial(denoise_whitened, C=C, patch=patch, algo=algo,
                                 shrink=shrink, workers=1),
                         dwi, mask, patch, outputs, workers, tile, d_field=d_field)

    whitener = sqrtm_psd(C, -0.5).astype(np.float32)
    centre_row = sqrtm_psd(C, 0.5)[center].astype(np.float32)
    C32 = C.astype(np.float32)

    # The level field rides along as one extra column, so it is wrap-padded and gathered
    # by exactly the same indices as the data.
    volumes = _with_level_column(dwi, d_field)

    denoised = dwi.copy()
    sigma_hat = np.zeros(dwi.shape[:3], np.float32)
    rank = np.zeros(dwi.shape[:3], np.int32)

    for coords, block in patch_tiles(volumes, mask, patch, tile=tile):
        if d_field is None:
            patches, levels = block, None
        else:
            patches, levels = block[:, :, :n_vols], block[:, :, n_vols]
            patches = patches / levels[:, :, None]          # D^(-1): homoscedastic rows
        signal, tile_sigma, tile_rank = _denoise_tile(
            patches, whitener, centre_row, C32, n_small, n_large, algo, shrink)
        if levels is not None:
            signal = signal * levels[:, center, None]       # back to data units at the centre
        denoised[coords], sigma_hat[coords], rank[coords] = signal, tile_sigma, tile_rank
    return denoised, sigma_hat, rank
