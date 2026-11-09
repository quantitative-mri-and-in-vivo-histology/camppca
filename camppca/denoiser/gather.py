"""Patch extraction: the box of voxels around each requested voxel, a tile at a time.

At the image edge the box moves inward until it fits, as in dwidenoise, so a voxel at the
border still gets a full box of real neighbours. It then sits off center in its box, and
the gather says in which row. Wrapping around to the opposite face instead would mix in
voxels that are not neighbours, and ``C`` would treat them as if they were. The box is
built by index arithmetic on the original volume, so no padded copy is ever made. Rows
come out in :func:`camppca.patch.patch_offsets` order, C order with z fastest, which is
the order ``rho`` is indexed in. The two must agree.

Tiles are small on purpose. Large ones, 100 MB at 256 patches, made the pool workers
fight over memory bandwidth and doubled the cost per voxel at 16 workers. 8-patch tiles
cost the same at 2 and at 16.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..patch import check_patch, patch_offsets, patch_size
from ..warn import warn

__all__ = ["TILE", "as_data", "box_centers", "check_dwi", "check_range", "patch_tiles",
           "restore_nonfinite", "zero_nonfinite"]

#: Patches per tile.
TILE = 8
#: How far below float32's largest value a patch's Gram must stay, for the whitening and
#: the division by ``d`` that come before it.
GRAM_HEADROOM = 1e6


def as_data(dwi) -> np.ndarray:
    """``dwi`` as the denoisers hold it: complex64 if it is complex, float32 otherwise."""
    dwi = np.asarray(dwi)
    return dwi.astype(np.complex64 if np.iscomplexobj(dwi) else np.float32, copy=False)


def check_dwi(dwi, spatial_shape=None):
    """``dwi`` as an array, refused unless 4D and, when given, of ``spatial_shape``."""
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X, Y, Z, N), got shape {dwi.shape}")
    if spatial_shape is not None and dwi.shape[:3] != tuple(spatial_shape):
        raise ValueError(f"dwi spatial shape {dwi.shape[:3]} does not match the mask's "
                         f"{tuple(spatial_shape)}")
    return dwi


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
    warn(f"{int(bad.sum())} voxels hold NaN or inf. They enter the fit as zero "
         "and keep their NaN or inf in the output.")
    values = dwi[bad]
    dwi = dwi.copy()
    dwi[bad] = np.where(np.isfinite(values), values, 0)
    return dwi, bad, values


def check_range(dwi, patch):
    """Raise if a value is so large that a patch's float32 Gram would overflow.

    A Gram entry sums ``n_vox * n_vols`` squares, and float32 ends at 3.4e38: about 6e13
    for 7^3 and 300 volumes, with :data:`GRAM_HEADROOM`. Real data stays far below, so a
    value that large is corrupt, like NaN. Unchecked, one entry of 3e19 made all 27
    neighbours of a 3^3 patch NaN, with sigma 1e-7 and no error.
    """
    peak = max(float(np.abs(dwi[x]).max()) for x in range(dwi.shape[0]))
    limit = float(np.sqrt(np.finfo(np.float32).max
                          / (patch_size(patch) * dwi.shape[3] * GRAM_HEADROOM)))
    if peak > limit:
        raise ValueError(f"the data reach {peak:.3g}, above {limit:.3g}, where a patch's "
                         "float32 Gram could overflow. Check the data, or rescale it.")


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
