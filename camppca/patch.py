"""Patch geometry, shared by both halves of the seam.

Neutral by design: the noise-model half needs the offsets to build C, the denoise half
needs them to gather voxels.  Putting them here keeps the engine from importing the
estimator.

⚠ The flattening order is C-order with z fastest, and it is PAIRED with the indexing of
``rho`` (:func:`table_index`).  The two must agree or C is silently wrong.
"""

import numpy as np


def patch_offsets(patch):
    """The engine's flattening order: C order over the box, z fastest."""
    patch = np.asarray(patch, int)
    return np.array([(dx, dy, dz) for dx in range(patch[0])
                     for dy in range(patch[1]) for dz in range(patch[2])])


def patch_size(patch):
    """Number of voxels in the patch (paper M, code ``n_vox``)."""
    return int(np.prod(np.asarray(patch, int)))


def table_index(dx, dy, dz, lag):
    """Where lag ``(dx, dy, dz)`` sits in ``rho``: ``rho[ix, iy, iz]``.

    A correlation has ``rho(h) = rho(-h)``, so the table holds the half with ``dz >= 0``
    and a lag with ``dz < 0`` is read as ``(-dx, -dy, -dz)``. Flipping only ``dz`` would
    add a mirror symmetry in z that the noise need not have. In the ``dz = 0`` plane both
    ``(dx, dy)`` and ``(-dx, -dy)`` are stored, with the same value.
    """
    flip = np.asarray(dz) < 0
    return (np.where(flip, -dx, dx) + lag, np.where(flip, -dy, dy) + lag, np.abs(dz))


def required_lag(patch):
    """The in-plane lag ``rho`` must cover for this patch: ``max(patch[0], patch[1]) - 1``.

    Any smaller and lag lookup runs off the table -- which numpy answers with a
    wraparound, not an error.  The through-slice reach is separate: ``patch[2] - 1``,
    which is ``rho.shape[2] - 1``.
    """
    patch = np.asarray(patch, int)
    return int(max(patch[0], patch[1])) - 1


def default_patch(shape):
    """dwidenoise's default patch for an (X, Y, Z, N) image.

    The smallest odd cube with at least N voxels, so the patch matrix is never wider than
    tall: 5 x 5 x 5 up to 125 volumes, 7 x 7 x 7 up to 343. Each side is clipped to the
    largest odd size that fits the image, since the patch needs a center voxel.

    References:
        Tournier JD, Smith R, Raffelt D, Tabbara R, Dhollander T, Pietsch M,
        Christiaens D, Jeurissen B, Yeh CH, Connelly A. MRtrix3: A fast, flexible and
        open software framework for medical image processing and visualisation.
        NeuroImage 2019;202:116137.
    """
    side = 1
    while side ** 3 < shape[3]:
        side += 2
    return tuple(min(side, n - 1 + n % 2) for n in shape[:3])


def resolve_patch(patch, shape):
    """``patch`` validated, or :func:`default_patch` for ``shape`` when it is None.

    Raises if the patch is larger than the image: no box of that size fits inside it.
    """
    patch = check_patch(default_patch(shape) if patch is None else patch)
    if np.any(patch > np.asarray(shape[:3])):
        raise ValueError(f"patch {tuple(int(v) for v in patch)} is larger than the "
                         f"image {tuple(int(v) for v in shape[:3])}")
    return patch


def check_patch(patch):
    """Validate a patch shape; returns it as a length-3 int array."""
    patch = np.asarray(patch, int)
    if patch.shape != (3,):
        raise ValueError(f"patch must be 3 ints, got {patch.shape}")
    if patch.min() < 1:
        raise ValueError(f"patch dimensions must be >= 1, got {tuple(int(v) for v in patch)}")
    if np.any(patch % 2 == 0):
        raise ValueError(
            f"patch dimensions must be odd, got {tuple(int(v) for v in patch)} -- an even "
            "dimension has no center voxel and the engine is center-voxel-only")
    return patch
