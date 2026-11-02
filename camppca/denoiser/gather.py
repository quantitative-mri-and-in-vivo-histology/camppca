"""Patch extraction: turn a masked volume into tiles of patch matrices.

Visit every masked voxel, pull the box of voxels around it, hand a batch of those boxes to
a JAX kernel.

The volume is wrap-padded first, so a voxel at the border still gets a full box. Rows come
out in :func:`camppca.patch.patch_offsets` order, C order with z fastest, which is the
order ``rho`` is indexed in. The two must agree.

Two ways to gather. Keeping the padded volume on the device and reading each tile's boxes
with an indexed gather is about 1.8x faster and bit-identical, but the padded volume has to
fit in device memory alongside the eigendecomposition workspace. The fallback gathers each
tile on the host and ships only that tile. The choice is automatic and can be forced.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np

from ..patch import check_patch, patch_offsets

__all__ = ["effective_tile", "gather_budget_bytes", "patch_tiles"]

# Bytes of Gram and eigh workspace one tile may use. Caps the tile size on top of the
# caller's request, so a long volume series does not blow up the device.
_TILE_WORKSPACE_BYTES = 5.0e7

_CPU_BUDGET_BYTES = 2.0e9
_DEVICE_HEADROOM_BYTES = 1.0e9


def gather_budget_bytes() -> float:
    """Bytes available for holding a padded volume on the device.

    On CUDA this queries free memory and leaves headroom for the per-tile Gram and eigh
    workspace. On CPU, or when the device cannot report its memory, it falls back to a
    conservative fixed budget.
    """
    try:
        device = jax.devices()[0]
        if device.platform != "gpu":
            return _CPU_BUDGET_BYTES
        stats = device.memory_stats() or {}
        free = stats.get("bytes_limit", 0) - stats.get("bytes_in_use", 0)
        return max(_CPU_BUDGET_BYTES, free - _DEVICE_HEADROOM_BYTES)
    except Exception:                       # noqa: BLE001 - any probe failure means fall back
        return _CPU_BUDGET_BYTES


def effective_tile(tile: int, n_vols: int) -> int:
    """The tile :func:`patch_tiles` uses: ``tile``, lowered to bound the per-tile workspace."""
    return min(int(tile), max(1, int(_TILE_WORKSPACE_BYTES / (n_vols * n_vols))))


def patch_tiles(dwi: np.ndarray, mask: np.ndarray, patch, tile: int = 256,
                on_device: bool | None = None
                ) -> Iterator[tuple[tuple[np.ndarray, ...], jax.Array]]:
    """Yield the masked voxels' patch matrices, one tile at a time.

    Args:
        dwi: (X, Y, Z, N) volume.
        mask: (X, Y, Z) boolean; only these voxels are visited.
        patch: length-3 odd box shape.
        tile: maximum voxels per yielded tile, lowered further to bound the per-tile
            device workspace.
        on_device: True to keep the padded volume on the device, False to gather each tile
            on the host, None to decide from free device memory.

    Yields:
        ``(coords, patches)``. ``coords`` is ``(ix, iy, iz)``, three index arrays of length
        B addressing the original volume. ``patches`` is a device array of shape
        ``(B, n_vox, N)``, rows in patch-offset order.
    """
    patch = check_patch(patch)
    dwi = np.asarray(dwi, np.float32)
    mask = np.asarray(mask, bool)
    n_vols = dwi.shape[3]
    offsets = patch_offsets(patch)
    half = patch // 2

    padded = np.pad(dwi, ((half[0], half[0]), (half[1], half[1]), (half[2], half[2]), (0, 0)),
                    mode="wrap")
    pad_x, pad_y, pad_z = padded.shape[:3]
    xs, ys, zs = np.where(mask)

    tile = effective_tile(tile, n_vols)
    if on_device is None:
        on_device = (pad_x * pad_y * pad_z * n_vols * 4) < gather_budget_bytes()

    if on_device:
        flat = jnp.asarray(padded.reshape(-1, n_vols))
        step = jnp.asarray((offsets[:, 0] * pad_y * pad_z + offsets[:, 1] * pad_z
                            + offsets[:, 2]).astype(np.int32))
        base = jnp.asarray((xs * pad_y * pad_z + ys * pad_z + zs).astype(np.int32))

    for start in range(0, xs.size, tile):
        sl = slice(start, start + tile)
        coords = (xs[sl], ys[sl], zs[sl])
        if on_device:
            patches = flat[base[sl][:, None] + step[None, :]]
        else:
            block = np.empty((coords[0].size, len(offsets), n_vols), np.float32)
            for row, (dx, dy, dz) in enumerate(offsets):
                block[:, row, :] = padded[coords[0] + dx, coords[1] + dy, coords[2] + dz, :]
            patches = jnp.asarray(block)
        yield coords, patches
