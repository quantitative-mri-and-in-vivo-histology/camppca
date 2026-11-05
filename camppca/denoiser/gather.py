"""Patch extraction: the box of voxels around each requested voxel, a tile at a time.

At the image edge the box moves inward until it fits, as in dwidenoise, so a voxel at the
border still gets a full box of real neighbours. It then sits off center in its box, and
the gather says in which row. Wrapping around to the opposite face instead would mix in
voxels that are not neighbours, and ``C`` would treat them as if they were. The box is
built by index arithmetic on the original volume, so no padded copy is ever made. Rows
come out in :func:`camppca.patch.patch_offsets` order, C order with z fastest, which is
the order ``rho`` is indexed in. The two must agree.

Tiles are small on purpose. A 256-patch tile at 297 volumes is about 100 MB, and 16 pool
workers streaming such tiles fought over memory bandwidth: a whitened voxel cost 6.8 ms
at 16 workers against 3.2 ms at 2. With 8-patch tiles it cost 3.2 ms at 16, flat from
2 to 8.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np

from ..patch import check_patch, patch_offsets

__all__ = ["TILE", "as_data", "box_centers", "patch_tiles", "restore_nonfinite",
           "zero_nonfinite"]

#: Patches per tile.
TILE = 8


def as_data(dwi) -> np.ndarray:
    """``dwi`` as the denoisers hold it: complex64 if it is complex, float32 otherwise."""
    dwi = np.asarray(dwi)
    return dwi.astype(np.complex64 if np.iscomplexobj(dwi) else np.float32, copy=False)


def zero_nonfinite(dwi):
    """``dwi`` with NaN and inf set to zero, and what it takes to put them back.

    One NaN would otherwise poison every patch that holds it. Resampled images often carry
    NaN in the background.

    Returns:
        ``(dwi, bad, values)``. ``bad`` is the (X, Y, Z) voxels that held any, ``values``
        their original (n_bad, N) data. When every entry is finite, ``bad`` and ``values``
        are None and ``dwi`` is returned as it is, without a copy.
    """
    bad = np.zeros(dwi.shape[:3], bool)
    for x in range(dwi.shape[0]):          # a slab at a time: no full-size bool array
        bad[x] = ~np.isfinite(dwi[x]).all(-1)
    if not bad.any():
        return dwi, None, None
    warnings.warn(f"{int(bad.sum())} voxels hold NaN or inf. They enter the fit as zero "
                  "and keep their NaN or inf in the output.", RuntimeWarning, stacklevel=3)
    values = dwi[bad]
    dwi = dwi.copy()
    dwi[bad] = np.where(np.isfinite(values), values, 0)
    return dwi, bad, values


def restore_nonfinite(denoised, bad, values):
    """Put the NaN and inf :func:`zero_nonfinite` removed back into ``denoised``, in place."""
    if bad is not None:
        denoised[bad] = np.where(np.isfinite(values), denoised[bad], values)


def box_centers(coords, shape, patch):
    """The center of each voxel's box, per axis: the voxel itself, moved inward at the
    image edge until the box fits. ``coords`` is a sequence of three index arrays."""
    half = np.asarray(patch, int) // 2
    return tuple(np.clip(c, h, int(n) - 1 - h) for c, h, n in zip(coords, half, shape,
                                                                 strict=True))


def patch_tiles(shape, voxels: np.ndarray, patch, tile: int = TILE
                ) -> Iterator[tuple[slice, tuple[np.ndarray, np.ndarray, np.ndarray],
                                    np.ndarray]]:
    """Yield, a tile at a time, the indices of the requested voxels' patches.

    Args:
        shape: the volume's spatial shape (X, Y, Z). No side smaller than ``patch``.
        voxels: flat voxel indices, C order, into that shape.
        patch: length-3 odd box shape.
        tile: voxels per yielded tile.

    Yields:
        ``(sl, index, rows)``. ``sl`` slices ``voxels`` to this tile. ``index`` is three
        (B, n_vox) integer arrays, so ``volume[index]`` is the tile's (B, n_vox, ...)
        patches of any array with that spatial shape, rows in patch-offset order.
        ``rows`` (B,) is the row of each requested voxel in its patch: the center row,
        ``n_vox // 2``, except within half a patch of the image edge.
    """
    patch = check_patch(patch)
    shape = tuple(int(s) for s in shape)
    off = patch_offsets(patch) - patch // 2
    coords = np.unravel_index(np.asarray(voxels), shape)
    centers = box_centers(coords, shape, patch)
    shift = [c - b + h for c, b, h in zip(coords, centers, patch // 2, strict=True)]
    rows = (shift[0] * patch[1] + shift[1]) * patch[2] + shift[2]
    for start in range(0, coords[0].size, int(tile)):
        sl = slice(start, start + int(tile))
        yield sl, tuple(b[sl, None] + off[None, :, k] for k, b in enumerate(centers)), \
            rows[sl]
