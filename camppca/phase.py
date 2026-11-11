"""Phase stabilization: complex diffusion data rotated onto the real axis, for real-valued
denoising.

Every diffusion-weighted volume carries its own phase, from motion during the diffusion
gradients, on top of a fixed phase from k-space sampling and the coil combination. The
phase differs between voxels and between volumes, so the complex patch matrix is not low
rank, and the magnitude carries a Rician noise floor. Removing a smooth estimate of the
phase leaves the signal near the real axis. The real part then has Gaussian noise and no
floor, and denoises like a magnitude image.

The recipe is the default phase step of NORDIC's NIFTI_NORDIC2:

1. in each slice, in plane, for every volume, estimate the smooth phase with the
   decorrelated filter of Sprenger et al., their Opt5;
2. rotate by minus that phase and keep the real part.

EDDEN's preprocessing first removes the raw phase of one bright reference volume from
every volume. That step is left out.

The filter is a 5 x 5 smoother in image space that gives almost no weight to the center
voxel and its direct neighbours, whose noise is correlated with the center's. The estimate
then barely sees the voxel's own noise, so the real part stays unbiased. EDDEN used a Hann
window cubed in k-space, which they describe as a Tukey filter with an FWHM of 58%. On
complex white noise of unit level it leaves the real part's mean at 0.54, against 0.01
with the filter and 1.25 for the magnitude.

References:
    Moeller S, Pisharady PK, Ramanna S, et al. NOise reduction with DIstribution
    Corrected (NORDIC) PCA in dMRI with complex-valued parameter-free locally
    low-rank processing. NeuroImage 2021;226:117539.

    Moeller S. NORDIC_Raw, https://github.com/SteenMoeller/NORDIC_Raw, commit
    1a893ec. The default phase step of NIFTI_NORDIC2 followed here.

    Eichner C, Cauley SF, Cohen-Adad J, et al. Real diffusion-weighted MRI enabling
    true signal averaging and increased diffusion contrast. NeuroImage
    2015;122:373-384.

    Sprenger T, Sperl JI, Fernandez B, Haase A, Menzel MI. Real valued
    diffusion-weighted imaging using decorrelated phase filtering. Magn Reson Med
    2017;77(2):559-570. The "dk" filter.

    Manzano Patron JP, Moeller S, Andersson JLR, Ugurbil K, Yacoub E, Sotiropoulos SN.
    Denoising diffusion MRI: considerations and implications for analysis. Imaging
    Neuroscience 2024;2:1-29. EDDEN's recipe, the one not taken.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from joblib import delayed
from scipy import ndimage

from .pool import resolve_workers, worker_pool
from .warn import warn

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


#: Radians, the phase format unless the caller declares another.
RADIANS = (-np.pi, np.pi)
#: How far the data may span past the declared turn, as a fraction of it.
SPAN_MARGIN = 0.01


def phase_scale(phase, phase_range=RADIANS) -> float:
    """The factor that turns a stored phase image into radians.

    ``phase_range`` is the pair of values that spans one full turn: radians by default,
    ``(-4096, 4096)`` for Siemens integers, ``(-180, 180)`` for degrees. Only its width
    matters, since a constant offset is removed with the rest of the phase, so radians
    from 0 to 2 pi pass too. The format is declared, not guessed from the range of the
    values, which scales an unwrapped or offset phase wrong without a word.

    Raises:
        ValueError: if the values span more than the declared turn, by over
            :data:`SPAN_MARGIN`, or hold no finite value. A real phase image spans close to
            a full turn, since its noisy background does, so less than half a turn warns.
    """
    lo, hi = (float(v) for v in phase_range)
    if not hi > lo:
        raise ValueError(f"phase_range must run from low to high, got {phase_range}")
    turn = hi - lo
    low, high = float(np.nanmin(phase)), float(np.nanmax(phase))
    if not np.isfinite(high - low):
        raise ValueError("the phase holds no finite value")
    if high - low > turn * (1 + SPAN_MARGIN):
        raise ValueError(
            f"the phase runs from {low:g} to {high:g}, wider than the declared turn of "
            f"{lo:g} to {hi:g}. Declare its format: --phase-range LOW HIGH on the command "
            f"line, phase_range=(low, high) in Python, e.g. -4096 4096 for Siemens integers "
            f"or -180 180 for degrees.")
    if high - low < turn / 2:
        warn(f"the phase runs from {low:g} to {high:g}, less than half the declared "
             f"turn of {lo:g} to {hi:g}. Check its format.")
    return 2 * np.pi / turn


def stabilize_phase(magnitude, phase, *, slice_axis=2, imag=False, workers=None,
                    phase_range=RADIANS):
    """Rotate complex 4D data onto the real axis and return the real part.

    Args:
        magnitude: (X, Y, Z, N) magnitude.
        phase: (X, Y, Z, N) phase, in the format ``phase_range`` declares.
        slice_axis: the spatial axis across slices. The filter works in the other two.
        imag: also return the imaginary part, which should hold noise only.
        workers: CPU processes the volumes are split over, see :mod:`camppca.pool`. The
            volumes are independent, so the result does not depend on it.
        phase_range: the values that span one full turn, radians by default. See
            :func:`phase_scale`.

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
    scale = phase_scale(phase, phase_range)    # per volume: no scaled copy of the phase
    workers = resolve_workers(workers)

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
            write(chunk, task(magnitude, phase, chunk))
    else:
        with worker_pool(workers, return_as="generator_unordered") as parallel:
            for chunk, out in parallel(delayed(_chunk_task)(task, chunk, magnitude, phase)
                                       for chunk in chunks):
                write(chunk, out)
    return (real, im) if imag else real


def _volume(magnitude, phase, n, scale):
    """Volume ``n`` as complex64."""
    return (magnitude[..., n] * np.exp(1j * scale * phase[..., n])).astype(np.complex64)


def _chunk_task(task, chunk, magnitude, phase):
    return chunk, task(magnitude, phase, chunk)


def _rotate_volumes(magnitude, phase, volumes, scale, slice_axis, imag):
    """Stabilize the phase of some volumes. Returns their real and imaginary parts,
    (X, Y, Z, len(volumes)) each, the imaginary part None unless ``imag``."""
    shape = magnitude.shape[:3]
    weights = np.expand_dims(_dk_weights(), slice_axis)
    real = np.empty(shape + (len(volumes),), np.float32)
    im = np.empty_like(real) if imag else None
    for k, n in enumerate(volumes):
        z = _volume(magnitude, phase, n, scale)
        # A zero adds nothing to the weighted sum whose angle is the smooth phase, so
        # zeroing NaN and inf leaves them out exactly. Left in, one NaN spread to its
        # 5 x 5 neighbours, which then went unturned. They stay NaN in the output.
        finite = np.where(np.isfinite(z), z, 0)
        smooth = (ndimage.convolve(finite.real, weights, mode="constant")
                  + 1j * ndimage.convolve(finite.imag, weights, mode="constant"))
        size = np.abs(smooth)
        # The unit phasor of the smooth phase, conjugated: exp(-i angle) without the
        # angle. Where the estimate is zero, outside the head, nothing is turned.
        turn = np.where(size > 0, smooth.conj() / np.where(size > 0, size, 1), 1)
        z *= turn.astype(np.complex64)
        real[..., k] = z.real
        if imag:
            im[..., k] = z.imag
    return real, im
