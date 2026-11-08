"""Phase stabilization: complex diffusion data rotated onto the real axis, for real-valued
denoising.

Every diffusion-weighted volume carries its own phase, from motion during the diffusion
gradients, on top of a fixed phase from k-space sampling and the coil combination. The
phase differs between voxels and between volumes, so the complex patch matrix is not low
rank, and the magnitude carries a Rician noise floor. Removing a smooth estimate of the
phase leaves the signal near the real axis. The real part then has Gaussian noise and no
floor, and denoises like a magnitude image.

The recipe is EDDEN's preprocessing, with a different filter:

1. remove the raw phase of one reference volume from every volume: the first volume whose
   mean magnitude is within 95% of the largest, in practice a b0. That takes out the fixed
   phase from k-space sampling and the coil combination, which can vary faster than any
   smooth estimate follows;
2. in each slice, in plane, for every volume, estimate the remaining smooth phase with the
   decorrelated filter of Sprenger et al., their Opt5;
3. rotate by minus that phase and keep the real part.

The filter is a 5 x 5 smoother in image space that gives almost no weight to the center
voxel and its direct neighbours, whose noise is correlated with the center's. The estimate
then barely sees the voxel's own noise, so the real part stays unbiased. EDDEN used a Hann
window cubed in k-space, which they describe as a Tukey filter with an FWHM of 58%. On
complex white noise of unit level it leaves the real part's mean at 0.54, against 0.01
with the filter and 1.25 for the magnitude.

The reference volume itself ends up as its magnitude, noise floor included. It is a bright
b0, where the floor is small.

References:
    Eichner C et al. Real diffusion-weighted MRI enabling true signal averaging and
    increased diffusion contrast. NeuroImage 2015;122:373-384.
    Sprenger T et al. Real valued diffusion-weighted imaging using decorrelated phase
    filtering. Magn Reson Med 2017;77:559-570.
    Manzano-Patron JP et al. Denoising diffusion MRI: considerations and implications for
    analysis. Imaging Neuroscience 2024;2. doi:10.1162/imag_a_00060 (EDDEN).
"""

from __future__ import annotations

from functools import partial

import numpy as np
from joblib import delayed
from scipy import ndimage

from .pool import resolve_workers, worker_pool

__all__ = ["stabilize_phase", "phase_scale"]

#: Sprenger et al.'s Opt5 weights by offset (|dx|, |dy|), sorted. The paper gives them
#: only as a color map (Fig. 3); these were read off it by color and sum to 1.006.
_DK_WEIGHTS = {(0, 0): 0.002, (0, 1): 0.006, (1, 1): 0.075,
               (0, 2): 0.053, (1, 2): 0.045, (2, 2): 0.027}


def _dk_weights():
    """The 5 x 5 "dk" filter weights, normalized to sum 1."""
    k = np.array([[_DK_WEIGHTS[tuple(sorted((abs(dx), abs(dy))))] for dy in range(-2, 3)]
                  for dx in range(-2, 3)])
    return k / k.sum()


def phase_scale(phase) -> float:
    """The factor that turns a stored phase image into radians.

    1 when the values fit in one turn of radians, within [-pi, 2pi]. Otherwise the range
    from the smallest to the largest value is taken as one full turn. That fits any
    unit: Siemens integers from -4096 to 4094, degrees, milliradians. A real phase image
    covers a full turn, since its noisy background does. A constant offset is harmless,
    since the stabilization removes it with the rest of the phase. NaN is ignored.
    """
    lo, hi = float(np.nanmin(phase)), float(np.nanmax(phase))
    if lo >= -np.pi - 1e-3 and hi <= 2 * np.pi + 1e-3 and hi - lo <= 2 * np.pi + 1e-3:
        return 1.0
    if not np.isfinite(hi - lo) or hi == lo:
        raise ValueError(f"cannot read the phase unit: its values run from {lo} to {hi}")
    return 2 * np.pi / (hi - lo)


def stabilize_phase(magnitude, phase, slice_axis=2, imag=False, workers=None):
    """Rotate complex 4D data onto the real axis and return the real part.

    Args:
        magnitude: (X, Y, Z, N) magnitude.
        phase: (X, Y, Z, N) phase, in radians or in any unit whose range is one full
            turn, as scanners store it. See :func:`phase_scale`.
        slice_axis: the spatial axis across slices. The filter works in the other two.
        imag: also return the imaginary part, which should hold noise only.
        workers: CPU processes the volumes are split over, see :mod:`camppca.pool`. The
            volumes are independent, so the result does not depend on it.

    Returns:
        (X, Y, Z, N) float32 real part, or ``(real, imag)``.
    """
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1 or 2, got {slice_axis}")
    magnitude, phase = np.asarray(magnitude), np.asarray(phase)
    if magnitude.ndim != 4 or magnitude.shape != phase.shape:
        raise ValueError(f"magnitude and phase must be 4D and alike, got {magnitude.shape} "
                         f"and {phase.shape}")
    n_vols = magnitude.shape[3]
    scale = phase_scale(phase)             # per volume: no scaled copy of the phase
    workers = resolve_workers(workers)

    reference = reference_volume(magnitude)
    fixed = np.exp(-1j * scale * phase[..., reference]).astype(np.complex64)

    real = np.empty(magnitude.shape, np.float32)
    im = np.empty(magnitude.shape, np.float32) if imag else None

    def write(chunk, out):
        real[..., chunk] = out[0]
        if imag:
            im[..., chunk] = out[1]

    task = partial(_rotate_volumes, scale=scale, slice_axis=slice_axis, imag=imag)
    chunks = np.array_split(np.arange(n_vols), min(n_vols, 4 * workers))
    if workers == 1:
        for chunk in chunks:
            write(chunk, task(magnitude, phase, fixed, chunk))
    else:
        with worker_pool(workers, return_as="generator_unordered") as parallel:
            for chunk, out in parallel(delayed(_chunk_task)(task, chunk, magnitude, phase,
                                                            fixed) for chunk in chunks):
                write(chunk, out)
    return (real, im) if imag else real


def reference_volume(magnitude) -> int:
    """The first volume whose mean magnitude is within 95% of the largest. NaN is
    ignored."""
    means = np.array([np.nanmean(magnitude[..., n]) for n in range(magnitude.shape[3])])
    return int(np.flatnonzero(means >= 0.95 * means.max())[0])


def _volume(magnitude, phase, n, scale):
    """Volume ``n`` as complex64."""
    return (magnitude[..., n] * np.exp(1j * scale * phase[..., n])).astype(np.complex64)


def _chunk_task(task, chunk, magnitude, phase, fixed):
    return chunk, task(magnitude, phase, fixed, chunk)


def _rotate_volumes(magnitude, phase, fixed, volumes, scale, slice_axis, imag):
    """Stabilize the phase of some volumes. Returns their real and imaginary parts,
    (X, Y, Z, len(volumes)) each, the imaginary part None unless ``imag``."""
    shape = magnitude.shape[:3]
    weights = np.expand_dims(_dk_weights(), slice_axis)
    real = np.empty(shape + (len(volumes),), np.float32)
    im = np.empty_like(real) if imag else None
    for k, n in enumerate(volumes):
        z = _volume(magnitude, phase, n, scale) * fixed
        smooth = (ndimage.convolve(z.real, weights, mode="constant")
                  + 1j * ndimage.convolve(z.imag, weights, mode="constant"))
        size = np.abs(smooth)
        # The unit phasor of the smooth phase, conjugated: exp(-i angle) without the
        # angle. Where the estimate is zero, outside the head, nothing is turned.
        turn = np.where(size > 0, smooth.conj() / np.where(size > 0, size, 1), 1)
        z *= turn.astype(np.complex64)
        real[..., k] = z.real
        if imag:
            im[..., k] = z.imag
    return real, im
