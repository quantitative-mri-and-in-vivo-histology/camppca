"""Per-voxel passes in numpy and SciPy alone, for the pool workers.

A worker that loads JAX holds about 290 MB of its own memory before any work, most of it
JAX's CPU runtime. One that loads only numpy and SciPy holds about 45 MB. SciPy's LAPACK
wrapper is also faster here than JAX's eigh, which calls the same OpenBLAS routine: at
n = 297 on one core, 1.2 ms against 2.2 ms for the eigenvalues alone.

Each function computes what its JAX counterpart computes, on the same patches, and is
tested against it. Nothing here may import JAX, directly or through the package.
"""

from functools import partial

import numpy as np
import scipy.linalg

from ..patch import check_patch, patch_offsets, patch_size
from .mp_threshold import mp_threshold

__all__ = ["whitened_level"]

# scipy.linalg.eigh queries LAPACK for its best workspace. The raw lapack.ssyevd wrapper
# defaults to the smallest one, which forces the unblocked reduction: 1.55 ms against
# 1.2 ms at n = 297.
_eigvalsh = partial(scipy.linalg.eigh, eigvals_only=True, driver="evd", check_finite=False)


def _tiles(volumes, mask, patch, tile):
    """Yield ``(coords, patches)`` like :func:`~camppca.denoiser.gather.patch_tiles`.

    Same wrap padding, same voxel order, same patch-row order, gathered on the host.
    """
    n_cols = volumes.shape[3]
    half = patch // 2
    padded = np.pad(volumes, [(h, h) for h in half] + [(0, 0)], mode="wrap")
    pad_y, pad_z = padded.shape[1:3]
    flat = padded.reshape(-1, n_cols)
    offsets = patch_offsets(patch)
    step = offsets[:, 0] * pad_y * pad_z + offsets[:, 1] * pad_z + offsets[:, 2]
    xs, ys, zs = np.where(mask)
    base = xs * pad_y * pad_z + ys * pad_z + zs
    for start in range(0, xs.size, tile):
        sl = slice(start, start + tile)
        yield (xs[sl], ys[sl], zs[sl]), flat[base[sl, None] + step[None, :]]


def whitened_level(dwi, mask, whitener, patch, algo, tile, d_field=None):
    """:func:`camppca.denoiser.camppca.whitened_level` on checked inputs, in numpy.

    Args:
        dwi: (X, Y, Z, N) float32 data.
        mask: (X, Y, Z) bool.
        whitener: (n_vox, n_vox) float32 ``C^(-1/2)``, or None for white noise.
        patch: box shape, three odd integers.
        algo: which Marchenko-Pastur variant to cut with.
        tile: patches per batch.
        d_field: (X, Y, Z) level field, already guarded, or None.

    Returns:
        (X, Y, Z) ``sigma_hat`` in d-normalized units, zero outside ``mask``.
    """
    patch = check_patch(patch)
    n_vols = dwi.shape[3]
    n_vox = patch_size(patch)
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    volumes = dwi if d_field is None else np.concatenate(
        [dwi, np.asarray(d_field, np.float32)[..., None]], axis=3)

    level = np.zeros(dwi.shape[:3], np.float32)
    for coords, block in _tiles(volumes, np.asarray(mask, bool), patch, tile):
        patches = block if d_field is None else block[:, :, :n_vols] / block[:, :, n_vols:]
        if whitener is not None:
            patches = np.matmul(whitener, patches)
        gram = np.matmul(patches.transpose(0, 2, 1), patches)
        evals = np.stack([_eigvalsh(g) for g in gram])
        level[coords] = mp_threshold(np.maximum(evals[:, ::-1], 0.0), n_small, n_large,
                                     algo, xp=np)[1]
    return level
