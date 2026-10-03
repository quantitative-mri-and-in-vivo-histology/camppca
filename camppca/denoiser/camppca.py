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

import jax
import jax.numpy as jnp
import numpy as np

from ..patch import check_patch, patch_size
from .gather import patch_tiles
from .mp_threshold import mp_threshold


def sqrtm_psd(A, p=0.5, floor=1e-10):
    """Matrix power of a symmetric PSD matrix, with the spectrum floored before raising."""
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor * max(w.max(), 1e-30))
    B = (V * w ** p) @ V.T
    return 0.5 * (B + B.T)


def _whiten_split(patches, whitener, n_small: int, n_large: int, algo: str):
    """Whiten a batch of patches and split each spectrum at the Marchenko-Pastur edge.

    Shared with the loop kernel estimator, so both split a patch with the same code.

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
    whitened = jnp.einsum("pq,bqn->bpn", whitener, patches)
    gram = jnp.einsum("bpn,bpm->bnm", whitened, whitened)
    evals, evecs = jnp.linalg.eigh(gram)                    # ascending
    evals = jnp.maximum(evals[:, ::-1], 0.0)                # descending
    evecs = evecs[:, :, ::-1]

    rank, sigma_hat = mp_threshold(evals, n_small, n_large, algo)
    return whitened, evals, evecs, rank, sigma_hat


def _denoise_tile(patches, whitener, centre_row, n_small: int, n_large: int, algo: str):
    """Denoise a batch of whitened patches, returning the centre voxel of each.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``.
        centre_row: (n_vox,) the centre row of ``C^(1/2)``, all that unwhitening needs.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algo: which Marchenko-Pastur variant to cut with.

    Returns:
        ``(signal, sigma_hat, rank)``: (B, n_vols) denoised centre spectra in whitened
        units, (B,) fitted noise level, (B,) retained component count.
    """
    n_vols = patches.shape[2]

    whitened, _, evecs, rank, sigma_hat = _whiten_split(
        patches, whitener, n_small, n_large, algo)

    centre = jnp.einsum("p,bpm->bm", centre_row, whitened)  # e_c^T C^(1/2) Yw
    coords = jnp.einsum("bm,bmk->bk", centre, evecs)
    keep = jnp.arange(n_vols)[None, :] < rank[:, None]
    signal = jnp.einsum("bk,bnk->bn", jnp.where(keep, coords, 0.0), evecs)
    return signal, sigma_hat, rank


_denoise_tile_jit = partial(jax.jit, static_argnums=(3, 4, 5))(_denoise_tile)


def denoise_whitened(dwi, mask, C, patch=(5, 5, 5), algo="cordero-grande", tile=256,
                     d_field=None):
    """MP-PCA on the patch whitened by ``C^(-1/2)``.

    Args:
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) boolean; only these voxels are denoised.
        C: (n_vox, n_vox) patch correlation with unit diagonal.
        patch: box shape, three odd integers.
        algo: which Marchenko-Pastur variant to cut with.
        tile: maximum patches per device batch.
        d_field: (X, Y, Z) relative level field, ALREADY median-normalized and clipped by
            the caller. Each voxel row is divided by its own level before whitening, which
            is exact for ``Sigma = sigma^2 D C D``, and the centre row is multiplied back
            on the way out. What it adds over a per-patch sigma is the within-patch
            heteroscedasticity, which widens the noise bulk the same way off-diagonal
            colour does.

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

    whitener = jnp.asarray(sqrtm_psd(C, -0.5), jnp.float32)
    centre_row = jnp.asarray(sqrtm_psd(C, 0.5)[center], jnp.float32)

    # The level field rides along as one extra column, so it is wrap-padded and gathered
    # by exactly the same indices as the data.
    if d_field is None:
        volumes = dwi
    else:
        volumes = np.concatenate([dwi, np.asarray(d_field, np.float32)[..., None]], axis=3)

    denoised = dwi.copy()
    sigma_hat = np.zeros(dwi.shape[:3], np.float32)
    rank = np.zeros(dwi.shape[:3], np.int32)

    for coords, block in patch_tiles(volumes, mask, patch, tile=tile):
        if d_field is None:
            patches, levels = block, None
        else:
            patches, levels = block[:, :, :n_vols], block[:, :, n_vols]
            patches = patches / levels[:, :, None]          # D^(-1): homoscedastic rows
        signal, tile_sigma, tile_rank = _denoise_tile_jit(
            patches, whitener, centre_row, n_small, n_large, algo)
        if levels is not None:
            signal = signal * levels[:, center, None]       # back to data units at the centre
        denoised[coords] = np.asarray(signal)
        sigma_hat[coords] = np.asarray(tile_sigma)
        rank[coords] = np.asarray(tile_rank)
    return denoised, sigma_hat, rank
