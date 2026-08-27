"""Brain mask and estimation-region helpers.

Ported from ``scripts/paper/multi_dataset/masking.py`` (see reference/), verbatim except
that an empty mask raises ``ValueError`` instead of ``SystemExit`` -- a library must not
kill its caller's interpreter.

WHY brain_mask IS NOT JUST threshold + erode.  That is what it started as, and on GE
MR750 data it collapsed the mask from ~137k voxels to 13.7k: the intensity threshold
leaves a POROUS mask there (interior voxels below the cut, ~1000 connected components),
and three erosions eat it from the inside out.  Every downstream number was then computed
on a sliver of brain, with no 9x9x9 patch fitting inside the mask at all.  Closing the
pores BEFORE eroding fixes it and barely moves the configs that were already fine.

Pure numpy + scipy.ndimage. Nothing here imports JAX.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, label

MASK_FRAC = 0.15      # threshold as a fraction of the mean-b0 99th percentile
CLOSE = 2             # seal pores BEFORE eroding -- see the module docstring
ERODE = 3             # rho_region and patch sampling need a strict interior


def brain_mask(b0_mean, frac=MASK_FRAC, close=CLOSE, erode=ERODE):
    """Eroded brain interior from the mean b0: threshold, largest component, close, fill, erode."""
    m = b0_mean > frac * np.percentile(b0_mean, 99)
    lab, n = label(m)
    if n == 0:
        raise ValueError("empty brain mask -- check the threshold fraction")
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    m = binary_fill_holes(binary_closing(m, iterations=close))
    for z in range(m.shape[2]):                    # 3-D fill misses pores open in-plane only
        m[:, :, z] = binary_fill_holes(m[:, :, z])
    return binary_erosion(m, iterations=erode)


def stable_region(cols, mask):
    """`mask` minus voxels that are (near-)CONSTANT across the given columns.

    rho_region standardises each voxel by its own std, `A / (std + 1e-12)`.  A voxel that
    is constant across the columns but not zero -- which happens with only 4-6
    b0-difference columns on the high-intensity Philips data -- is amplified by up to
    1e12 and swamps every correlation sum, leaving rho ~ 1e-6 at every non-zero lag.
    Such voxels carry no covariance information, so drop them from the estimation region
    rather than let them dominate it.
    """
    s = cols.std(1)
    keep = s > 1e-3 * np.median(s)
    reg = np.zeros(mask.shape, bool)
    reg[mask] = keep
    return reg, int((~keep).sum())
