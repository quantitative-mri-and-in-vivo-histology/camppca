"""Phase stabilization: complex diffusion data rotated onto the real axis, for real-valued
denoising.

Every diffusion-weighted volume carries its own phase, from motion during the diffusion
gradients, on top of a fixed phase from k-space sampling and the coil combination. The
phase differs between voxels and between volumes, so the complex patch matrix is not low
rank, and the magnitude carries a Rician noise floor. Removing a smooth estimate of the
phase leaves the signal near the real axis. The real part then has Gaussian noise and no
floor, and denoises like a magnitude image.

In each slice, in plane, for every volume, as in the EDDEN preprocessing:

1. remove the phase of the complex mean over all volumes, the fixed part;
2. estimate the smooth phase that is left with a filter;
3. rotate by minus that phase and keep the real part.

The filters:

- ``"tukey"``: a low-pass in k-space, a Tukey window set by its full width at half
  maximum as a fraction of k-space. EDDEN used 58%.
- ``"dk"``: the decorrelated kernel of Sprenger et al., their Opt5. A 5 x 5 smoother in
  image space that gives almost no weight to the centre voxel and its direct neighbours,
  whose noise is correlated with the centre's. The estimate then barely sees the voxel's
  own noise, so the real part stays unbiased.

On complex white noise of unit level, the real part's mean over pure noise is 0.28 with
"dk" and 0.91 with "tukey" at 58%, against 1.25 for the magnitude. At a signal of 1 it is
1.02, 1.27 and 1.55. Hence "dk" is the default.

References:
    Eichner C et al. Real diffusion-weighted MRI enabling true signal averaging and
    increased diffusion contrast. NeuroImage 2015;122:373-384.
    Sprenger T et al. Real valued diffusion-weighted imaging using decorrelated phase
    filtering. Magn Reson Med 2017;77:559-570.
    Manzano-Patron JP et al. Denoising diffusion MRI: considerations and implications for
    analysis. Imaging Neuroscience 2024;2. doi:10.1162/imag_a_00060 (EDDEN).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["FILTERS", "stabilize_phase", "phase_scale"]

FILTERS = ("tukey", "dk")

#: Sprenger et al.'s Opt5 weights by offset (|dx|, |dy|), sorted. The paper gives them
#: only as a colour map (Fig. 3); these were read off it by colour and sum to 1.008.
_DK_WEIGHTS = {(0, 0): 0.002, (0, 1): 0.006, (1, 1): 0.075,
               (0, 2): 0.053, (1, 2): 0.045, (2, 2): 0.027}


def _dk_kernel():
    """The 5 x 5 decorrelated kernel, normalized to sum 1."""
    k = np.array([[_DK_WEIGHTS[tuple(sorted((abs(dx), abs(dy))))] for dy in range(-2, 3)]
                  for dx in range(-2, 3)])
    return k / k.sum()


def _tukey(n, fwhm):
    """Tukey window over ``n`` k-space lines, in FFT order, with its half maximum at
    ``fwhm`` of the k-space width: flat out to ``2 fwhm - 1``, then a cosine taper to zero
    at the edge."""
    r = 2 * np.abs(np.fft.fftfreq(n))                 # 0 at the centre, 1 at the edge
    flat, taper = 2 * fwhm - 1, 2 * (1 - fwhm)
    return 0.5 * (1 + np.cos(np.pi * np.clip(r - flat, 0, None) / taper))


def phase_scale(phase) -> float:
    """The factor that turns a stored phase image into radians.

    1 when every value lies within [-pi, pi]. Otherwise the image is taken as integers
    spread over a power-of-two range for one full turn, as scanners store it: Siemens
    writes -4096 to 4094 or 0 to 4095. A constant offset is harmless, since the
    stabilization removes it with the rest of the phase.
    """
    lo, hi = float(np.min(phase)), float(np.max(phase))
    if lo >= -np.pi - 1e-3 and hi <= np.pi + 1e-3:
        return 1.0
    return 2 * np.pi / 2 ** np.ceil(np.log2(hi - lo) - 1e-3)   # 8190, 8192 -> 2^13


def stabilize_phase(magnitude, phase, filter="dk", fwhm=0.58, mean_phase=True,
                    slice_axis=2, imag=False):
    """Rotate complex 4D data onto the real axis and return the real part.

    Args:
        magnitude: (X, Y, Z, N) magnitude.
        phase: (X, Y, Z, N) phase in radians, see :func:`phase_scale`.
        filter: "tukey" or "dk", the estimate of the smooth phase; see the module.
        fwhm: "tukey" only, the window's full width at half maximum as a fraction of
            k-space, from 0.5 to below 1. Smaller is smoother.
        mean_phase: first remove the phase of the complex mean over all volumes.
        slice_axis: the spatial axis across slices. The filters work in the other two.
        imag: also return the imaginary part, which should hold noise only.

    Returns:
        (X, Y, Z, N) float32 real part, or ``(real, imag)``.
    """
    if filter not in FILTERS:
        raise ValueError(f"filter must be one of {FILTERS}, got {filter!r}")
    if filter == "tukey" and not 0.5 <= fwhm < 1:
        raise ValueError(f"fwhm must be in [0.5, 1), got {fwhm}")
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1 or 2, got {slice_axis}")
    magnitude, phase = np.asarray(magnitude), np.asarray(phase)
    if magnitude.ndim != 4 or magnitude.shape != phase.shape:
        raise ValueError(f"magnitude and phase must be 4D and alike, got {magnitude.shape} "
                         f"and {phase.shape}")
    plane = tuple(a for a in range(3) if a != slice_axis)
    n_vols = magnitude.shape[3]

    def volume(n):
        return (magnitude[..., n] * np.exp(1j * phase[..., n])).astype(np.complex64)

    fixed = np.ones(magnitude.shape[:3], np.complex64)
    if mean_phase:
        total = np.zeros(magnitude.shape[:3], np.complex128)
        for n in range(n_vols):
            total += volume(n)
        fixed = np.exp(-1j * np.angle(total)).astype(np.complex64)

    if filter == "tukey":
        window = np.ones(magnitude.shape[:3], np.float32)
        for a in plane:
            window = window * _tukey(magnitude.shape[a], fwhm).reshape(
                [-1 if b == a else 1 for b in range(3)]).astype(np.float32)
    else:
        kernel = np.expand_dims(_dk_kernel(), slice_axis)

    real = np.empty(magnitude.shape, np.float32)
    im = np.empty(magnitude.shape, np.float32) if imag else None
    for n in range(n_vols):
        z = volume(n) * fixed
        if filter == "tukey":
            smooth = np.fft.ifftn(np.fft.fftn(z, axes=plane) * window, axes=plane)
        else:
            smooth = (ndimage.convolve(z.real, kernel, mode="constant")
                      + 1j * ndimage.convolve(z.imag, kernel, mode="constant"))
        z = z * np.exp(-1j * np.angle(smooth)).astype(np.complex64)
        real[..., n] = z.real
        if imag:
            im[..., n] = z.imag
    return (real, im) if imag else real
