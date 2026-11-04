"""Patch extraction: the box of voxels around each requested voxel, a tile at a time.

The box wraps around the volume's edges, so a voxel at the border still gets a full box.
The wrap is done by index arithmetic on the original volume, so no padded copy is ever
made: neither of the whole volume nor of a worker's share. Rows come out in
:func:`camppca.patch.patch_offsets` order, C order with z fastest, which is the order
``rho`` is indexed in. The two must agree.

Tiles are small on purpose. A 256-patch tile at 297 volumes is about 100 MB, and 16 pool
workers streaming such tiles fought over memory bandwidth: a whitened voxel cost 6.8 ms
at 16 workers against 3.2 ms at 2. With 8-patch tiles it cost 3.2 ms at 16, flat from
2 to 8.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..patch import check_patch, patch_offsets

__all__ = ["TILE", "as_data", "patch_tiles"]

#: Patches per tile.
TILE = 8


def as_data(dwi) -> np.ndarray:
    """``dwi`` as the denoisers hold it: complex64 if it is complex, float32 otherwise."""
    dwi = np.asarray(dwi)
    return dwi.astype(np.complex64 if np.iscomplexobj(dwi) else np.float32, copy=False)


def patch_tiles(shape, voxels: np.ndarray, patch, tile: int = TILE
                ) -> Iterator[tuple[slice, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Yield, a tile at a time, the indices of the requested voxels' patches.

    Args:
        shape: the volume's spatial shape (X, Y, Z).
        voxels: flat voxel indices, C order, into that shape.
        patch: length-3 odd box shape.
        tile: voxels per yielded tile.

    Yields:
        ``(sl, index)``. ``sl`` slices ``voxels`` to this tile. ``index`` is three (B, n_vox)
        integer arrays, so ``volume[index]`` is the tile's (B, n_vox, ...) patches of any
        array with that spatial shape, rows in patch-offset order.
    """
    patch = check_patch(patch)
    shape = tuple(int(s) for s in shape)
    off = patch_offsets(patch) - patch // 2
    ix, iy, iz = np.unravel_index(np.asarray(voxels), shape)
    for start in range(0, ix.size, int(tile)):
        sl = slice(start, start + int(tile))
        yield sl, ((ix[sl, None] + off[None, :, 0]) % shape[0],
                   (iy[sl, None] + off[None, :, 1]) % shape[1],
                   (iz[sl, None] + off[None, :, 2]) % shape[2])
