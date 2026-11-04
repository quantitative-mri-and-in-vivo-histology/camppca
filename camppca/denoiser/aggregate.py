"""Patch averaging: the mean of the overlapping patch reconstructions, as an option.

The default writes back the centre voxel of each patch only. ``aggregate="mean"``
rebuilds every patch whole and averages, at each voxel, all the patches that cover it:

- only patches centred in the mask are used, and their contributions to voxels outside
  the mask are dropped, so those voxels pass through unchanged as in the default;
- ``stride`` keeps a patch centre every ``stride`` voxels per axis; mask voxels the grid
  leaves uncovered, near a thin edge say, get their own centred patch;
- plain mean, no weights;
- ``sigma_hat`` is the root of the mean ``sigma_hat^2`` over the covering patches, and
  ``rank`` the mean rank.

Each chunk of patch centres sums its rebuilt patches over its box: the x-slices its
patches reach, all of y and z. The boxes are added in chunk order, and the chunks do not
depend on the number of workers, so the result does not either, bit for bit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["AGGREGATES", "MEAN_CHUNKS", "check_aggregate", "patch_centres", "BoxSum",
           "MeanAssembly"]

AGGREGATES = ("centre", "mean")

#: Chunks of patch centres in mean mode, fixed so the sums do not depend on the workers.
MEAN_CHUNKS = 32


def check_aggregate(aggregate: str, stride) -> tuple[str, int]:
    """Validate ``aggregate`` and ``stride``. Returns them, ``stride`` as int."""
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > 1 and aggregate == "centre":
        raise ValueError("stride > 1 needs aggregate='mean': with the centre voxel only, "
                         "every voxel needs its own patch")
    return aggregate, stride


def patch_centres(mask: np.ndarray, patch, stride: int) -> np.ndarray:
    """Flat indices (C order) of the patch centres for mean mode.

    Mask voxels on the stride grid, plus mask voxels no grid patch covers. Coverage wraps
    at the volume's edges, as the gather does.
    """
    mask = np.asarray(mask, bool)
    if stride == 1:
        return np.flatnonzero(mask)
    grid = mask.copy()
    for axis, n in enumerate(mask.shape):
        on = (np.arange(n) % stride == 0).reshape([-1 if a == axis else 1 for a in range(3)])
        grid &= on
    cover = ndimage.convolve(grid.astype(np.int32), np.ones(tuple(int(p) for p in patch),
                                                            np.int32), mode="wrap")
    return np.flatnonzero(grid | (mask & (cover == 0)))


class BoxSum:
    """One chunk's sums of rebuilt patches over its box, in a worker.

    The box is the x-slices the chunk's patches reach, wrapping in x, with all of y and z.
    Only voxels inside the mask receive anything.
    """

    def __init__(self, shape, centres, patch, mask, dtype=np.float32):
        X, Y, Z, n_vols = shape
        half = int(patch[0]) // 2
        xs = np.asarray(centres) // (Y * Z)
        lo, hi = int(xs.min()) - half, int(xs.max()) + half
        if hi - lo + 1 >= X:
            lo, self.x = 0, np.arange(X)
        else:
            self.x = np.arange(lo, hi + 1) % X
        self.lo, self.dims, self.mask = lo, (X, Y, Z), mask
        # A patch wider than the volume wraps onto itself and repeats indices.
        self.repeats = any(int(p) > n for p, n in zip(patch, (X, Y, Z), strict=True))
        n = self.x.size * Y * Z
        self.sum = np.zeros((n, n_vols), dtype)
        self.count = np.zeros(n, np.int32)
        self.var = np.zeros(n, np.float64)
        self.rank = np.zeros(n, np.float64)

    def add(self, index, recon, sigma_hat, rank):
        """Add one tile: ``index`` from :func:`~camppca.denoiser.gather.patch_tiles`,
        ``recon`` (B, n_vox, N) rebuilt patches, ``sigma_hat`` and ``rank`` (B,)."""
        X, Y, Z = self.dims
        px, py, pz = index
        flat = (((px - self.lo) % X) * Y + py) * Z + pz
        inside = self.mask[px, py, pz]
        for b in range(recon.shape[0]):          # one patch's own indices are unique
            f, keep = flat[b][inside[b]], inside[b]
            if self.repeats:
                np.add.at(self.sum, f, recon[b][keep])
                np.add.at(self.count, f, 1)
                np.add.at(self.var, f, float(sigma_hat[b]) ** 2)
                np.add.at(self.rank, f, float(rank[b]))
            else:
                self.sum[f] += recon[b][keep]
                self.count[f] += 1
                self.var[f] += float(sigma_hat[b]) ** 2
                self.rank[f] += float(rank[b])

    def result(self):
        """``(x, sum, count, var, rank)``, the last four shaped (bx, Y, Z[, N])."""
        _, Y, Z = self.dims
        box = (self.x.size, Y, Z)
        return (self.x, self.sum.reshape(box + (-1,)), self.count.reshape(box),
                self.var.reshape(box), self.rank.reshape(box))


class MeanAssembly:
    """The full-size sums in the main process, and the means at the end.

    The output starts as the input with the mask voxels zeroed, and the boxes add into
    it, so no third full image is needed. Outside the mask the boxes add zeros.
    """

    def __init__(self, dwi, mask):
        self.mask = np.asarray(mask, bool)
        self.denoised = dwi.copy()
        for i in range(self.mask.shape[0]):
            self.denoised[i][self.mask[i]] = 0
        self.count = np.zeros(self.mask.shape, np.int32)
        self.var = np.zeros(self.mask.shape, np.float64)
        self.rank = np.zeros(self.mask.shape, np.float64)

    def add(self, outputs):
        x, total, count, var, rank = outputs
        self.denoised[x] += total
        self.count[x] += count
        self.var[x] += var
        self.rank[x] += rank

    def finish(self):
        """``(denoised, sigma_hat, rank)``; sigma_hat and rank zero outside the mask."""
        sigma_hat = np.zeros(self.mask.shape, np.float32)
        rank = np.zeros(self.mask.shape, np.float32)
        for i in range(self.mask.shape[0]):
            m = self.count[i] > 0
            n = self.count[i][m]
            self.denoised[i][m] /= n[:, None].astype(np.float32)
            sigma_hat[i][m] = np.sqrt(self.var[i][m] / n)
            rank[i][m] = self.rank[i][m] / n
        return self.denoised, sigma_hat, rank
