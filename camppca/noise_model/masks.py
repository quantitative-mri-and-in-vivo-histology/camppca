"""Brain mask.

An empty mask raises ``ValueError``.

On some scanners the intensity threshold leaves a porous mask, with interior voxels
below the cut. Erosion then eats it from the inside out. So the pores are closed before
eroding.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, label

MASK_FRAC = 0.15      # threshold as a fraction of the mean-b0 99th percentile
CLOSE = 2             # seal pores before eroding, see the module docstring
ERODE = 3

# `estimate_noise_model` calls `brain_mask(..., erode=0)`: with `mask_only`, the model's
# mask is also the set of voxels the engine writes, so eroding it would leave a 3-voxel
# rim un-denoised.


def brain_mask(b0_mean, *, frac=MASK_FRAC, close=CLOSE, erode=ERODE):
    """Eroded brain interior from the mean b0: threshold, largest component, close, fill, erode.

    Args:
        b0_mean: (X, Y, Z) mean b0 image.
        frac: the threshold, as a fraction of the image's 99th percentile.
        close: closing iterations that seal pores before eroding, see the module
            docstring. 0 skips it.
        erode: erosion iterations. 0 keeps the filled mask as it is.

    Returns:
        (X, Y, Z) bool mask.

    Raises:
        ValueError: if the mask is empty after thresholding or after eroding.
    """
    m = b0_mean > frac * np.percentile(b0_mean, 99)
    lab, n = label(m)
    if n == 0:
        raise ValueError("empty brain mask. Check the threshold fraction.")
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    if close > 0:
        # Padded with the edge slices: scipy erodes against a zero border, so the closing
        # cut two slices off every face the brain touches.
        m = binary_closing(np.pad(m, close, mode="edge"), iterations=close)
        m = m[(slice(close, -close),) * 3]
    m = binary_fill_holes(m)
    for z in range(m.shape[2]):                    # 3-D fill misses pores open in-plane only
        m[:, :, z] = binary_fill_holes(m[:, :, z])
    if erode <= 0:
        return m           # scipy erodes to convergence at iterations=0, not zero times
    eroded = binary_erosion(m, iterations=erode)
    if not eroded.any():
        raise ValueError(
            f"brain mask is empty after {erode} erosions (it held {int(m.sum())} voxels "
            f"before). Thin volumes lose every slice this way. Lower `erode`, or pass a "
            f"mask explicitly.")
    return eroded
