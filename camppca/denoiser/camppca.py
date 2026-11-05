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

STANDARD MP-PCA IS THE CASE C = None. The whitening multiply is skipped and the centre row
is the patch's own, so :func:`~camppca.denoiser.mppca.denoise_mppca` runs this same engine.

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
from .gather import TILE, as_data, patch_tiles, restore_nonfinite, zero_nonfinite
from .mp_threshold import mp_threshold
from .shrink import check_shrink, shrink_weight, signal_alignment
from .spectrum import eigh_descending, eigvalsh_descending


def sqrtm_psd(A, p=0.5, floor=1e-10):
    """Matrix power of a symmetric PSD matrix, with the spectrum floored before raising."""
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor * max(w.max(), 1e-30))
    B = (V * w ** p) @ V.T
    return 0.5 * (B + B.T)


def _whiten_split(patches, whitener, n_small: int, n_large: int, algorithm: str,
                  values_only: bool = False):
    """Whiten a batch of patches and split each spectrum at the Marchenko-Pastur edge.

    The one place the split happens: both denoisers, the level pass and the loop kernel
    estimator all call it. It runs in the dtype of ``patches``.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``, or None for white noise, which skips the
            multiply.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algorithm: which Marchenko-Pastur variant to cut with.
        values_only: compute the eigenvalues only, for the level pass. ``evecs`` is then
            None.

    Returns:
        ``(whitened, evals, evecs, rank, sigma_hat)``: (B, n_vox, n_vols) whitened
        patches, (B, n_vols) descending Gram eigenvalues, (B, n_vols, n_vols) their
        eigenvectors as columns in the same order, (B,) retained component count, (B,)
        fitted noise level.
    """
    whitened = patches if whitener is None else np.matmul(whitener, patches)
    gram = np.matmul(whitened.conj().transpose(0, 2, 1), whitened)
    if values_only:
        evals, evecs = eigvalsh_descending(gram), None
    else:
        evals, evecs = eigh_descending(gram)
    rank, sigma_hat = mp_threshold(evals, n_small, n_large, algorithm)
    return whitened, evals, evecs, rank, sigma_hat


def _tile_patches(dwi, d_field, index):
    """A tile's patches, rows divided by their level when there is a level field.

    Returns:
        ``(patches, levels)``: (B, n_vox, n_vols) in the dtype of ``dwi``, and (B, n_vox)
        levels or None.
    """
    patches = np.asarray(dwi[index], dwi.dtype)
    if d_field is None:
        return patches, None
    levels = d_field[index]
    return patches / levels[:, :, None], levels                # D^(-1): homoscedastic rows


def _denoise_tile(patches, whitener, root, C, n_small: int, n_large: int, algorithm: str,
                  shrink: str, full: bool = False, levels=None, rows=None):
    """Denoise a batch of whitened patches, returning the centre voxel of each, or with
    ``full`` each whole rebuilt patch.

    The whole patch unwhitens without ``C^(1/2)``: ``C^(1/2) (C^(-1/2) X) V diag(w) V^T``
    is ``X V diag(w) V^T``, because both roots come from one eigendecomposition in
    :func:`sqrtm_psd`. ``V`` are the whitened Gram's eigenvectors, ``w`` the kept and
    shrink weights. The centre row is the same identity for one row.

    Args:
        patches: (B, n_vox, n_vols) patch matrices, already divided by ``d``.
        whitener: (n_vox, n_vox) ``C^(-1/2)``, or None for white noise.
        root: (n_vox, n_vox) ``C^(1/2)``, or None with no whitener. Unwhitening one voxel
            needs only its row.
        C: (n_vox, n_vox) the patch correlation itself; only "lr" reads it.
        n_small: the smaller of ``n_vox`` and ``n_vols``.
        n_large: the larger of them.
        algorithm: which Marchenko-Pastur variant to cut with.
        shrink: "hard", "gd" or "lr"; see :mod:`camppca.denoiser.shrink`.
        full: rebuild the whole patch instead of the centre row.
        levels: with ``full`` and "lr", (B, n_vox) the patch voxels' levels ``d``, so
            "lr" weighs the error over the whole patch in data units; None for ``D = I``.
        rows: (B,) the row of the voxel to return in each patch, from
            :func:`~camppca.denoiser.gather.patch_tiles`. None is the centre row.

    Returns:
        ``(signal, sigma_hat, rank)``: (B, n_vols) denoised centre spectra, or with
        ``full`` (B, n_vox, n_vols) rebuilt patches, in the units of ``patches``, (B,)
        fitted noise level, (B,) retained component count.
    """
    n_vox, n_vols = patches.shape[1], patches.shape[2]

    whitened, evals, evecs, rank, sigma_hat = _whiten_split(
        patches, whitener, n_small, n_large, algorithm)

    if full:
        r = int(rank.max())
        w = (np.arange(r)[None, :] < rank[:, None]).astype(patches.dtype)
        if shrink != "hard":
            alignment = signal_alignment(whitened, evecs, evals, C, levels, root) \
                if shrink == "lr" else None
            w = w * shrink_weight(evals, sigma_hat, n_vox, n_vols, alignment)[:, :r]
        V = evecs[:, :, :r]
        return np.matmul(np.matmul(patches, V) * w[:, None, :], V.conj().transpose(0, 2, 1)), \
            sigma_hat, rank

    center = n_vox // 2
    if root is None:                                            # no whitening: Yw = Y
        centre = whitened[:, center, :] if rows is None else \
            whitened[np.arange(len(rows)), rows, :]
    else:
        centre = np.einsum("p,bpm->bm", root[center], whitened)  # e_c^T C^(1/2) Yw
        edge = None if rows is None else np.flatnonzero(rows != center)
        if edge is not None and edge.size:   # off centre near the image edge: its own row
            centre[edge] = np.einsum("bp,bpm->bm", root[rows[edge]], whitened[edge])
    coords = np.einsum("bm,bmk->bk", centre, evecs)
    keep = np.arange(n_vols)[None, :] < rank[:, None]
    kept = np.where(keep, coords, 0.0)
    if shrink != "hard":
        alignment = signal_alignment(whitened, evecs, evals, C) if shrink == "lr" else None
        kept = kept * shrink_weight(evals, sigma_hat, n_vox, n_vols, alignment)
    signal = np.einsum("bk,bnk->bn", kept, evecs.conj())
    return signal, sigma_hat, rank


def _level_voxels(dwi, d_field, voxels, whitener, patch, algorithm, tile):
    """The tile loop of :func:`whitened_level` for some voxels.

    Returns:
        ``(sigma_hat,)``: (V,) in d-normalized units, in the order of ``voxels``.
    """
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    level = np.empty(voxels.size, np.float32)
    for sl, index, _ in patch_tiles(dwi.shape[:3], voxels, patch, tile=tile):
        patches, _ = _tile_patches(dwi, d_field, index)
        level[sl] = _whiten_split(patches, whitener, n_small, n_large, algorithm,
                                  values_only=True)[4]
    return (level,)


def whitened_level(dwi, mask, C, patch=None, algorithm="cordero-grande", tile=TILE,
                   d_field=None, workers=None):
    """The MP noise level of every mask voxel's whitened patch, without denoising.

    The split :func:`denoise_whitened` makes, stopped once ``sigma_hat`` is known: only the
    eigenvalues are computed and nothing is reconstructed. With ``C`` None and no
    ``d_field`` this is standard MP-PCA's ``sigma_hat``.

    Args:
        dwi: (X, Y, Z, N) data, real or complex. For complex data ``sigma_hat`` is one
            channel's level, as in the denoisers.
        mask: (X, Y, Z) boolean.
        C: (n_vox, n_vox) patch correlation with unit diagonal, or None for white noise,
            which skips the whitening multiply.
        patch: box shape, three odd integers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        algorithm: which Marchenko-Pastur variant to cut with.
        tile: patches per batch; see :mod:`camppca.denoiser.gather`.
        d_field: (X, Y, Z) level field, already guarded, or None.
        workers: CPU processes to split the voxels over, see :mod:`camppca.pool`.

    Returns:
        (X, Y, Z) ``sigma_hat`` in d-normalized units, zero outside ``mask``.
    """
    dwi = as_data(dwi)
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

    run_voxels(partial(_level_voxels, whitener=whitener, patch=patch, algorithm=algorithm,
                       tile=tile),
               (dwi, d_field), voxels, write, resolve_workers(workers), tile)
    if np.iscomplexobj(dwi):
        level /= np.sqrt(2.0)              # one channel's level, as in the denoisers
    return level


def _denoise_whitened_voxels(dwi, d_field, voxels, whitener, root, C, patch, algorithm,
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
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    signal = np.empty((voxels.size, n_vols), dwi.dtype)
    sigma_hat = np.empty(voxels.size, np.float32)
    rank = np.empty(voxels.size, np.int32)
    for sl, index, rows in patch_tiles(dwi.shape[:3], voxels, patch, tile=tile):
        patches, levels = _tile_patches(dwi, d_field, index)
        tile_signal, sigma_hat[sl], rank[sl] = _denoise_tile(
            patches, whitener, root, C, n_small, n_large, algorithm, shrink, rows=rows)
        if levels is not None:
            tile_signal = tile_signal * levels[np.arange(len(rows)), rows, None]  # data units
        signal[sl] = tile_signal
    return signal, sigma_hat, rank


def _mean_whitened_voxels(dwi, d_field, mask, centres, whitener, root, C, patch,
                          algorithm, tile, shrink):
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
    box = BoxSum(dwi.shape, centres, patch, mask, dwi.dtype)
    for _, index, _ in patch_tiles(dwi.shape[:3], centres, patch, tile=tile):
        patches, levels = _tile_patches(dwi, d_field, index)
        recon, sigma_hat, rank = _denoise_tile(patches, whitener, root, C, n_small,
                                               n_large, algorithm, shrink, full=True,
                                               levels=levels)
        if levels is not None:
            recon = recon * levels[:, :, None]
        box.add(index, recon, sigma_hat, rank)
    return box.result()


def denoise_whitened(dwi, mask, C, patch=None, algorithm="cordero-grande", tile=TILE,
                     d_field=None, shrink="hard", workers=None, aggregate="centre",
                     stride=1):
    """MP-PCA on the patch whitened by ``C^(-1/2)``.

    Args:
        dwi: (X, Y, Z, N) data, real or complex; see :mod:`camppca.denoiser.mppca` for
            complex data.
        mask: (X, Y, Z) boolean; only these voxels are denoised.
        C: (n_vox, n_vox) patch correlation with unit diagonal, or None for white noise:
            standard MP-PCA, where "lr" is "gd".
        patch: box shape, three odd integers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        algorithm: which Marchenko-Pastur variant to cut with.
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
    dwi = as_data(dwi)
    mask = np.asarray(mask, bool)          # non-bool would make `& mask` an integer op
    patch = resolve_patch(patch, dwi.shape)  # odd dims: centre = n_vox // 2 must BE the centre
    dwi, bad, values = zero_nonfinite(dwi)
    aggregate, stride = check_aggregate(aggregate, stride)
    n_vox = patch_size(patch)

    shrink = check_shrink(shrink)
    if C is None:
        whitener = root = None
        shrink = "gd" if shrink == "lr" else shrink   # C = I: the correction is exactly 1
    else:
        C = np.asarray(C, np.float64)
        if C.shape != (n_vox, n_vox):
            raise ValueError(f"C must be ({n_vox}, {n_vox}) for patch "
                             f"{tuple(int(v) for v in patch)}, got {C.shape}")
        whitener = sqrtm_psd(C, -0.5).astype(np.float32)
        root = sqrtm_psd(C, 0.5).astype(np.float32)
        C = C.astype(np.float32)
    d_field = None if d_field is None else np.asarray(d_field, np.float32)
    workers = resolve_workers(workers)

    if aggregate == "mean":
        assembly = MeanAssembly(dwi, mask)
        run_boxes(partial(_mean_whitened_voxels, whitener=whitener, root=root, C=C, patch=patch, algorithm=algorithm, tile=tile,
                          shrink=shrink),
                  (dwi, d_field, mask),
                  voxel_chunks(patch_centres(mask, patch, stride), MEAN_CHUNKS, tile),
                  assembly.add, workers)
        denoised, sigma_hat, rank = assembly.finish()
        if np.iscomplexobj(dwi):
            sigma_hat /= np.sqrt(2.0)      # one channel's level, as for real data
        restore_nonfinite(denoised, bad, values)
        return denoised, sigma_hat, rank

    voxels = np.flatnonzero(mask)
    coords = np.unravel_index(voxels, mask.shape)
    denoised = dwi.copy()
    sigma_hat = np.zeros(dwi.shape[:3], np.float32)
    rank = np.zeros(dwi.shape[:3], np.int32)

    def write(sl, outputs):
        where = tuple(c[sl] for c in coords)
        denoised[where], sigma_hat[where], rank[where] = outputs

    run_voxels(partial(_denoise_whitened_voxels, whitener=whitener, root=root, C=C, patch=patch, algorithm=algorithm, tile=tile, shrink=shrink),
               (dwi, d_field), voxels, write, workers, tile)
    if np.iscomplexobj(dwi):
        sigma_hat /= np.sqrt(2.0)
    restore_nonfinite(denoised, bad, values)
    return denoised, sigma_hat, rank
