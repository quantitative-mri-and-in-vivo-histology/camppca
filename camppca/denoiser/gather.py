"""Patch extraction: turn a masked volume into tiles of patch matrices.

Visit every masked voxel, pull the box of voxels around it, and hand a batch of those
boxes to the denoiser.

The volume is wrap-padded first, so a voxel at the border still gets a full box. Rows come
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
    padded = np.pad(dwi, [(h, h) for h in half] + [(0, 0)], mode="wrap")
    pad_y, pad_z = padded.shape[1:3]
    flat = padded.reshape(-1, n_cols)
    offsets = patch_offsets(patch)
    step = offsets[:, 0] * pad_y * pad_z + offsets[:, 1] * pad_z + offsets[:, 2]
    xs, ys, zs = np.where(mask)
    base = xs * pad_y * pad_z + ys * pad_z + zs
    for start in range(0, xs.size, int(tile)):
        sl = slice(start, start + int(tile))
        yield (xs[sl], ys[sl], zs[sl]), flat[base[sl, None] + step[None, :]]
