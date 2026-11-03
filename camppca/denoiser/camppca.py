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

from ..patch import patch_size, resolve_patch
from ..pool import resolve_workers, run_boxes, run_voxels, voxel_chunks
from .aggregate import MEAN_CHUNKS, BoxSum, MeanAssembly, check_aggregate, patch_centres
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
                  shrink: str, full: bool = False, levels=None, root=None):
    """Denoise a batch of whitened patches, returning the centre voxel of each, or with
    ``full`` each whole rebuilt patch.

    The whole patch unwhitens without ``C^(1/2)``: ``C^(1/2) (C^(-1/2) X) V diag(w) V^T``
    is ``X V diag(w) V^T``, because both roots come from one eigendecomposition in
    :func:`sqrtm_psd`. ``V`` are the whitened Gram's eigenvectors, ``w`` the kept and
    shrink weights. The centre row is the same identity for one row.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``.
        centre_row: (n_vox,) the centre row of ``C^(1/2)``, all that unwhitening needs.
        C: (n_vox, n_vox) the patch correlation itself; only "lr" reads it.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algo: which Marchenko-Pastur variant to cut with.
        shrink: "hard", "gd" or "lr"; see :mod:`camppca.denoiser.shrink`.
        full: rebuild the whole patch instead of the centre row.
        levels: with ``full`` and "lr", (B, n_vox) the patch voxels' levels ``d``, so
            "lr" weighs the error over the whole patch in data units; None for ``D = I``.
        root: ``C^(1/2)``, needed with ``levels``.

    Returns:
        ``(signal, sigma_hat, rank)``: (B, n_vols) denoised centre spectra, or with
        ``full`` (B, n_vox, n_vols) rebuilt patches, in the units of ``patches``, (B,)
        fitted noise level, (B,) retained component count.
    """
    n_vox, n_vols = patches.shape[1], patches.shape[2]

    whitened, evals, evecs, rank, sigma_hat = _whiten_split(
        patches, whitener, n_small, n_large, algo)

    if full:
        r = int(rank.max())
        w = (np.arange(r)[None, :] < rank[:, None]).astype(patches.dtype)
        if shrink != "hard":
            alignment = signal_alignment(whitened, evecs, evals, C, levels, root) \
                if shrink == "lr" else None
            w = w * shrink_weight(evals, sigma_hat, n_vox, n_vols, alignment)[:, :r]
        V = evecs[:, :, :r]
        return np.matmul(np.matmul(patches, V) * w[:, None, :], V.transpose(0, 2, 1)), \
            sigma_hat, rank

    centre = np.einsum("p,bpm->bm", centre_row, whitened)    # e_c^T C^(1/2) Yw
    coords = np.einsum("bm,bmk->bk", centre, evecs)
    keep = np.arange(n_vols)[None, :] < rank[:, None]
    kept = np.where(keep, coords, 0.0)
    if shrink != "hard":
        alignment = signal_alignment(whitened, evecs, evals, C) if shrink == "lr" else None
        kept = kept * shrink_weight(evals, sigma_hat, n_vox, n_vols, alignment)
    signal = np.einsum("bk,bnk->bn", kept, evecs)
    return signal, sigma_hat, rank


def _level_voxels(dwi, d_field, voxels, whitener, patch, algo, tile):
    """The tile loop of :func:`whitened_level` for some voxels.

    Returns:
        ``(sigma_hat,)``: (V,) in d-normalized units, in the order of ``voxels``.
    """
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    level = np.empty(voxels.size, np.float32)
    for sl, index in patch_tiles(dwi.shape[:3], voxels, patch, tile=tile):
        patches = np.asarray(dwi[index], np.float32)
        if d_field is not None:
            patches = patches / d_field[index][:, :, None]
        if whitener is not None:
            patches = np.matmul(whitener, patches)
        evals = eigvalsh_descending(np.matmul(patches.transpose(0, 2, 1), patches))
        level[sl] = mp_threshold(evals, n_small, n_large, algo)[1]
    return (level,)


def whitened_level(dwi, mask, C, patch=None, algo="cordero-grande", tile=TILE,
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
        patch: box shape, three odd integers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        algo: which Marchenko-Pastur variant to cut with.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        d_field: (X, Y, Z) level field, already guarded, or None.
        workers: CPU processes to split the voxels over, see :mod:`camppca.pool`.

    Returns:
        (X, Y, Z) ``sigma_hat`` in d-normalized units, zero outside ``mask``.
    """
    dwi = np.asarray(dwi, np.float32)
    mask = np.asarray(mask, bool)
    patch = resolve_patch(patch, dwi.shape)
    whitener = None if C is None else sqrtm_psd(
        np.asarray(C, np.float64), -0.5).astype(np.float32)
    d_field = None if d_field is None else np.asarray(d_field, np.float32)

    voxels = np.flatnonzero(mask)
    coords = np.unravel_index(voxels, mask.shape)
    level = np.zeros(dwi.shape[:3], np.float32)

    def write(sl, outputs):
        level[tuple(c[sl] for c in coords)] = outputs[0]

    run_voxels(partial(_level_voxels, whitener=whitener, patch=patch, algo=algo, tile=tile),
               (dwi, d_field), voxels, write, resolve_workers(workers), tile)
    return level


def _denoise_whitened_voxels(dwi, d_field, voxels, whitener, centre_row, C, patch, algo,
                             tile, shrink):
    """The tile loop of :func:`denoise_whitened` for some voxels.

    Each voxel row is divided by its level before whitening, and the centre row is
    multiplied back on the way out.

    Returns:
        ``(signal, sigma_hat, rank)``: (V, N) in data units, (V,) in d-normalized units,
        (V,), in the order of ``voxels``.
    """
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    center = n_vox // 2
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    signal = np.empty((voxels.size, n_vols), np.float32)
    sigma_hat = np.empty(voxels.size, np.float32)
    rank = np.empty(voxels.size, np.int32)
    for sl, index in patch_tiles(dwi.shape[:3], voxels, patch, tile=tile):
        patches = np.asarray(dwi[index], np.float32)
        levels = None if d_field is None else d_field[index]
        if levels is not None:
            patches = patches / levels[:, :, None]          # D^(-1): homoscedastic rows
        tile_signal, sigma_hat[sl], rank[sl] = _denoise_tile(
            patches, whitener, centre_row, C, n_small, n_large, algo, shrink)
        if levels is not None:
            tile_signal = tile_signal * levels[:, center, None]   # data units at the centre
        signal[sl] = tile_signal
    return signal, sigma_hat, rank


def _mean_whitened_voxels(dwi, d_field, mask, centres, whitener, centre_row, root, C, patch,
                          algo, tile, shrink):
    """Sum the rebuilt patches around ``centres`` over their box, for the mean.

    Rows are divided by their level before whitening and multiplied back after. With
    "lr", the shrink weights see the whole patch's levels, since every voxel of it
    reaches the output.

    Returns:
        :meth:`~camppca.denoiser.aggregate.BoxSum.result`.
    """
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    box = BoxSum(dwi.shape, centres, patch, mask)
    for _, index in patch_tiles(dwi.shape[:3], centres, patch, tile=tile):
        patches = np.asarray(dwi[index], np.float32)
        levels = None if d_field is None else d_field[index]
        if levels is not None:
            patches = patches / levels[:, :, None]
        recon, sigma_hat, rank = _denoise_tile(patches, whitener, centre_row, C, n_small,
                                               n_large, algo, shrink, full=True,
                                               levels=levels, root=root)
        if levels is not None:
            recon = recon * levels[:, :, None]
        box.add(index, recon, sigma_hat, rank)
    return box.result()


def denoise_whitened(dwi, mask, C, patch=None, algo="cordero-grande", tile=TILE,
                     d_field=None, shrink="hard", workers=None, aggregate="centre",
                     stride=1):
    """MP-PCA on the patch whitened by ``C^(-1/2)``.

    Args:
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) boolean; only these voxels are denoised.
        C: (n_vox, n_vox) patch correlation with unit diagonal.
        patch: box shape, three odd integers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        algo: which Marchenko-Pastur variant to cut with.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        d_field: (X, Y, Z) level field, ALREADY guarded by the caller
            (:func:`~camppca.denoiser._effective_d`). Each voxel row is divided by its own level before whitening, which
            is exact for ``Sigma = sigma^2 D C D``, and the centre row is multiplied back
            on the way out. What it adds over a per-patch sigma is the within-patch
            heteroscedasticity, which widens the noise bulk the same way off-diagonal
            colour does.
        shrink: weight on the kept components; see :mod:`camppca.denoiser.shrink`.
        workers: CPU processes to split the voxels over, see :mod:`camppca.pool`.
        aggregate: "centre" (default) or "mean" over the overlapping patches; see
            :mod:`camppca.denoiser.aggregate`.
        stride: with "mean", a patch centre every ``stride`` voxels per axis.

    Returns:
        ``(denoised, sigma_hat, rank)``. ``sigma_hat`` is in d-normalized units; the
        caller multiplies by ``d`` to get data units. With "mean", ``rank`` is the mean
        rank as float32.
    """
    dwi = np.asarray(dwi, np.float32)
    mask = np.asarray(mask, bool)          # non-bool would make `& mask` an integer op
    patch = resolve_patch(patch, dwi.shape)  # odd dims: centre = n_vox // 2 must BE the centre
    aggregate, stride = check_aggregate(aggregate, stride)
    n_vox = patch_size(patch)
    center = n_vox // 2

    C = np.asarray(C, np.float64)
    if C.shape != (n_vox, n_vox):
        raise ValueError(f"C must be ({n_vox}, {n_vox}) for patch "
                         f"{tuple(int(v) for v in patch)}, got {C.shape}")

    whitener = sqrtm_psd(C, -0.5).astype(np.float32)
    centre_row = sqrtm_psd(C, 0.5)[center].astype(np.float32)
    d_field = None if d_field is None else np.asarray(d_field, np.float32)
    workers = resolve_workers(workers)

    if aggregate == "mean":
        assembly = MeanAssembly(dwi, mask)
        run_boxes(partial(_mean_whitened_voxels, whitener=whitener, centre_row=centre_row,
                          root=sqrtm_psd(C, 0.5).astype(np.float32),
                          C=C.astype(np.float32), patch=patch, algo=algo, tile=tile,
                          shrink=shrink),
                  (dwi, d_field, mask),
                  voxel_chunks(patch_centres(mask, patch, stride), MEAN_CHUNKS, tile),
                  assembly.add, workers)
        return assembly.finish()

    voxels = np.flatnonzero(mask)
    coords = np.unravel_index(voxels, mask.shape)
    denoised = dwi.copy()
    sigma_hat = np.zeros(dwi.shape[:3], np.float32)
    rank = np.zeros(dwi.shape[:3], np.int32)

    def write(sl, outputs):
        where = tuple(c[sl] for c in coords)
        denoised[where], sigma_hat[where], rank[where] = outputs

    run_voxels(partial(_denoise_whitened_voxels, whitener=whitener, centre_row=centre_row,
                       C=C.astype(np.float32), patch=patch, algo=algo, tile=tile,
                       shrink=shrink),
               (dwi, d_field), voxels, write, workers, tile)
    return denoised, sigma_hat, rank
