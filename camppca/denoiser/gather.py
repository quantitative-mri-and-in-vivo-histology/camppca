"""Patch extraction: turn a masked volume into tiles of patch matrices.

Visit every masked voxel, pull the box of voxels around it, and hand a batch of those
boxes to the denoiser.

The volume is wrap-padded first, so a voxel at the border still gets a full box. When
every requested voxel already has a half-patch border inside the volume, as in the pool's
slabs, the padding would never be read and is skipped: it is a copy of the whole volume.
Rows come
out in :func:`camppca.patch.patch_offsets` order, C order with z fastest, which is the
order ``rho`` is indexed in. The two must agree.

Tiles are small on purpose. A 256-patch tile at 297 volumes is about 100 MB, and 16 pool
workers streaming such tiles fought over memory bandwidth: a whitened voxel cost 6.8 ms
at 16 workers against 3.2 ms at 2. With 8-patch tiles it cost 3.2 ms at 16, flat from
2 to 8.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..patch import check_patch, patch_offsets

__all__ = ["TILE", "patch_tiles"]

#: Patches per tile.
TILE = 8


def patch_tiles(dwi: np.ndarray, mask: np.ndarray, patch, tile: int = TILE
                ) -> Iterator[tuple[tuple[np.ndarray, ...], np.ndarray]]:
    """Yield the masked voxels' patch matrices, one tile at a time.

    Args:
        dwi: (X, Y, Z, N) volume.
        mask: (X, Y, Z) boolean; only these voxels are visited.
        patch: length-3 odd box shape.
        tile: voxels per yielded tile.

    Yields:
        ``(coords, patches)``. ``coords`` is ``(ix, iy, iz)``, three index arrays of length
        B addressing the original volume. ``patches`` is ``(B, n_vox, N)``, rows in
        patch-offset order, the dtype of ``dwi``.
    """
    patch = check_patch(patch)
    mask = np.asarray(mask, bool)
    n_cols = dwi.shape[3]
    half = patch // 2
    xs, ys, zs = np.where(mask)
    bordered = all(c.size and c.min() >= h and c.max() < n - h
                   for c, h, n in zip((xs, ys, zs), half, dwi.shape[:3], strict=True))
    if bordered:
        src, offsets = dwi, patch_offsets(patch) - half
    else:
        src = np.pad(dwi, [(h, h) for h in half] + [(0, 0)], mode="wrap")
        offsets = patch_offsets(patch)
    pad_y, pad_z = src.shape[1:3]
    flat = src.reshape(-1, n_cols)
    step = offsets[:, 0] * pad_y * pad_z + offsets[:, 1] * pad_z + offsets[:, 2]
    base = xs * pad_y * pad_z + ys * pad_z + zs
    for start in range(0, xs.size, int(tile)):
        sl = slice(start, start + int(tile))
        yield (xs[sl], ys[sl], zs[sl]), flat[base[sl, None] + step[None, :]]
