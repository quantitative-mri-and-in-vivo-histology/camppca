"""Patch averaging: the mean of the overlapping patch reconstructions, as an option.

The default writes back the center voxel of each patch only. ``aggregate="mean"``
rebuilds every patch whole and averages, at each voxel, all the patches that cover it:

- only the patches of mask voxels are used, and their contributions to voxels outside
  the mask are dropped, so those voxels pass through unchanged as in the default;
- at the image edge several voxels share one box, moved inward, and it counts once;
- ``stride`` keeps a patch center every ``stride`` voxels per axis; mask voxels the grid
  leaves uncovered, near a thin edge say, get their own centered patch;
- plain mean, no weights;
- ``sigma_mppca`` is the root of the mean ``sigma_mppca^2`` over the covering patches, and
  ``rank`` the mean rank.

Each chunk of patch centers sums its rebuilt patches over its box: the x-slices its
patches reach, all of y and z. The boxes are added in chunk order, and the chunks do not
depend on the number of workers, so the result does not either, bit for bit.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .gather import box_centers

__all__ = ["AGGREGATES", "MEAN_CHUNKS", "check_aggregate", "patch_centers", "BoxSum",
           "MeanAssembly"]

AGGREGATES = ("center", "mean")

#: Chunks of patch centers in mean mode, fixed so the sums do not depend on the workers.
MEAN_CHUNKS = 32


def check_aggregate(aggregate: str, stride) -> tuple[str, int]:
    """Validate ``aggregate`` and ``stride``. Returns them, ``stride`` as int."""
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if stride > 1 and aggregate == "center":
        raise ValueError("stride > 1 needs aggregate='mean': with the center voxel only, "
                         "every voxel needs its own patch")
    return aggregate, stride


def patch_centers(mask: np.ndarray, patch, stride: int) -> np.ndarray:
    """Flat indices (C order) of the patch centers for mean mode, each box once.

    Mask voxels on the stride grid, plus mask voxels no grid patch covers. A box at the
    image edge is moved inward, as the gather moves it, so the centers returned are box
    centers. Voxels that share a box give it once.
    """
    mask = np.asarray(mask, bool)
    if stride == 1:
        return _unique_boxes(mask, patch)
    grid = mask.copy()
    for axis, n in enumerate(mask.shape):
        on = (np.arange(n) % stride == 0).reshape([-1 if a == axis else 1 for a in range(3)])
        grid &= on
    boxes = np.zeros(mask.shape, np.int32)       # grid patches counted at their box centers
    np.add.at(boxes, box_centers(np.nonzero(grid), mask.shape, patch), 1)
    cover = ndimage.convolve(boxes, np.ones(tuple(int(p) for p in patch), np.int32),
                             mode="constant")
    return _unique_boxes(grid | (mask & (cover == 0)), patch)


def _unique_boxes(voxels, patch):
    """Flat indices of the distinct box centers of the voxels in a boolean volume."""
    centers = box_centers(np.nonzero(voxels), voxels.shape, patch)
    return np.unique(np.ravel_multi_index(centers, voxels.shape))


class BoxSum:
    """One chunk's sums of rebuilt patches over its box, in a worker.

    The box is the x-slices the chunk's patches reach, with all of y and z. Only voxels
    inside the mask receive anything.
    """

    def __init__(self, shape, centers, patch, mask, dtype=np.float32):
        X, Y, Z, n_vols = shape
        half = int(patch[0]) // 2
        xs = box_centers((np.asarray(centers) // (Y * Z),), (X,), patch[:1])[0]
        lo, hi = int(xs.min()) - half, int(xs.max()) + half
        self.x = np.arange(lo, hi + 1)
        self.lo, self.dims, self.mask = lo, (X, Y, Z), mask
        n = self.x.size * Y * Z
        self.sum = np.zeros((n, n_vols), dtype)
        self.count = np.zeros(n, np.int32)
        self.var = np.zeros(n, np.float64)
        self.rank = np.zeros(n, np.float64)

    def add(self, index, recon, sigma_mppca, rank):
        """Add one tile: ``index`` from :func:`~camppca.denoiser.gather.patch_tiles`,
        ``recon`` (B, n_vox, N) rebuilt patches, ``sigma_mppca`` and ``rank`` (B,)."""
        _, Y, Z = self.dims
        px, py, pz = index
        flat = ((px - self.lo) * Y + py) * Z + pz
        inside = self.mask[px, py, pz]
        for b in range(recon.shape[0]):          # one patch's own indices are unique
            f, keep = flat[b][inside[b]], inside[b]
            self.sum[f] += recon[b][keep]
            self.count[f] += 1
            self.var[f] += float(sigma_mppca[b]) ** 2
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
        """``(denoised, sigma_mppca, rank)``; sigma_mppca and rank zero outside the mask."""
        sigma_mppca = np.zeros(self.mask.shape, np.float32)
        rank = np.zeros(self.mask.shape, np.float32)
        for i in range(self.mask.shape[0]):
            m = self.count[i] > 0
            n = self.count[i][m]
            self.denoised[i][m] /= n[:, None].astype(np.float32)
            sigma_mppca[i][m] = np.sqrt(self.var[i][m] / n)
            rank[i][m] = self.rank[i][m] / n
        return self.denoised, sigma_mppca, rank
