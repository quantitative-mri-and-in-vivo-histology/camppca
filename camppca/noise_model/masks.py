"""Brain mask.

An empty mask raises ``ValueError``.

WHY brain_mask IS NOT JUST threshold + erode.  That is what it started as, and on GE
MR750 data it collapsed the mask from ~137k voxels to 13.7k: the intensity threshold
leaves a POROUS mask there (interior voxels below the cut, ~1000 connected components),
and three erosions eat it from the inside out.  Every downstream number was then computed
on a sliver of brain, with no 9x9x9 patch fitting inside the mask at all.  Closing the
pores BEFORE eroding fixes it and barely moves the configs that were already fine.

Pure numpy + scipy.ndimage.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, label

MASK_FRAC = 0.15      # threshold as a fraction of the mean-b0 99th percentile
CLOSE = 2             # seal pores BEFORE eroding -- see the module docstring
ERODE = 3

# ⚠ `estimate_noise_model` calls `brain_mask(..., erode=0)`: with `mask_only`, the model's
# mask is also the set of voxels the engine writes, so eroding it would leave a 3-voxel
# rim un-denoised.


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
    if erode <= 0:
        return m           # ⚠ scipy erodes to CONVERGENCE at iterations=0, not zero times
    eroded = binary_erosion(m, iterations=erode)
    if not eroded.any():
        raise ValueError(
            f"brain mask is empty after {erode} erosions (it held {int(m.sum())} voxels "
            f"before). Thin volumes lose every slice this way -- lower `erode`, or pass a "
            f"mask explicitly.")
    return eroded
