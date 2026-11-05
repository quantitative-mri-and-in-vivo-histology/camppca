"""The denoisers: the engine half of the seam.

Source-agnostic by design. ``denoise_with_noise_model`` consumes a
:class:`~camppca.noise_model.NoiseModel` and never asks where it came from, so a future
alignment-based estimator feeds the same code.

``camppca.py`` is the covariance-aware method, ``mppca.py`` the standard white-noise one.
What they share lives in ``mp_threshold.py``, ``gather.py`` and ``result.py``, next to
them. The loop kernel estimator in ``noise_model/loop.py`` reuses the split,
``sqrtm_psd`` and ``_effective_d`` from here.

Every public name here is re-exported from ``camppca`` itself.
"""

import warnings

import numpy as np
from scipy import ndimage

from ..noise_model import estimate_noise_model
from ..patch import resolve_patch
from .aggregate import check_aggregate
from .camppca import denoise_whitened
from .gather import TILE
from .mp_threshold import check_algorithm
from .mppca import denoise_mppca
from .result import DenoiseResult
from .shrink import RULES, check_shrink

__all__ = ["DenoiseResult", "denoise_camppca", "denoise_mppca",
           "denoise_with_noise_model"]

#: Levels below this fraction of the median over the mask count as bad. Whitening divides
#: by ``d``, so a patch of near-constant data, whose MP level is near zero, would blow up.
LEVEL_FLOOR = 1e-2

def _good_level(d, mask):
    """Mask voxels whose level is finite and at least :data:`LEVEL_FLOOR` of the median."""
    d = np.asarray(d, np.float32)
    good = np.isfinite(d) & (d > 0) & np.asarray(mask, bool)
    if not good.any():
        raise ValueError("noise field d is empty or non-positive everywhere in the mask")
    return good & (d >= LEVEL_FLOOR * np.median(d[good]))


def _effective_d(d, mask):
    """The level field as the engine uses it. The one place ``d`` is guarded.

    ``d`` is defined up to scale: rows are divided by it and the center voxel multiplied
    back, so a constant cancels in the denoised image, the rank and ``sigma_hat`` in data
    units. Voxels that are non-finite, below :data:`LEVEL_FLOOR` times the median over the
    mask, or outside the mask get the level of the nearest good mask voxel. Edge patches include outside voxels, and by default they
    are denoised too, so a brain voxel the mask missed gets a local level.

    The engine takes the field as given and rescales the denoised image by it at the center
    voxel, but NOT ``sigma_hat``. Putting sigma_hat back into data units therefore means
    applying this same field, which is why it is computed here and passed down.
    """
    d = np.asarray(d, np.float32)
    good = _good_level(d, mask)
    nearest = ndimage.distance_transform_edt(~good, return_distances=False,
                                             return_indices=True)
    return d[tuple(nearest)].astype(np.float32)


def denoise_with_noise_model(dwi, model, patch=None, noise_field=True,
                             algorithm="cordero-grande", tile=TILE, shrink="hard", workers=None,
                             aggregate="center", stride=1, mask_only=False):
    """Denoise in the whitened domain, given a noise model.

    ``model``        a :class:`~camppca.noise_model.NoiseModel`.  Raises if ``patch`` is
                     wider than the lag the model measured -- never silently clipped.
    ``patch``        None takes the patch the model was estimated for, the one its loop
                     fitted to, or dwidenoise's default if the model records none
                     (:func:`~camppca.patch.default_patch`).
    ``noise_field``  apply the model's relative level field ``d`` (Sigma = sigma^2 D C D).
                     Adds the WITHIN-patch heteroscedasticity, which widens the noise bulk
                     the same way off-diagonal color does.
    ``shrink``       weight on the kept components: "hard" (default), "gd" or "lr"; see
                     :mod:`camppca.denoiser.shrink`.
    ``workers``      CPU processes to split the voxels over, see
                     :mod:`camppca.pool`. The default is one per physical core. 1 runs in
                     this process.
    ``aggregate``    "center" (default) writes back each patch's center voxel; "mean"
                     averages the overlapping patch reconstructions, see
                     :mod:`camppca.denoiser.aggregate`.
    ``stride``       with "mean", a patch center every ``stride`` voxels per axis.
    ``mask_only``    denoise only inside the model's mask and pass the rest through.
                     By default every voxel is denoised. The mask only says where the
                     model was estimated, and a brain mask always misses some brain at
                     its edge. Outside it ``d`` takes the nearest mask voxel's level.

    The rule -- Exp2 / Cordero-Grande, center-voxel-only, hard threshold unless ``shrink``
    or ``aggregate`` say otherwise -- is the method.
    """
    shrink = check_shrink(shrink)
    check_algorithm(algorithm)
    aggregate, stride = check_aggregate(aggregate, stride)
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X,Y,Z,N), got shape {dwi.shape}")
    patch = resolve_patch(model.provenance.get("patch") if patch is None else patch,
                          dwi.shape)
    if dwi.shape[:3] != model.mask.shape:
        raise ValueError(f"dwi spatial shape {dwi.shape[:3]} != model mask "
                         f"{model.mask.shape}")

    fitted = (model.provenance.get("loop_algorithm"), model.provenance.get("loop_noise_field"))
    if "loop_algorithm" in model.provenance and fitted != (algorithm, bool(noise_field)):
        warnings.warn(f"the loop kernel was fitted with algorithm={fitted[0]!r}, "
                      f"noise_field={fitted[1]}, but this denoise uses algorithm={algorithm!r}, "
                      f"noise_field={bool(noise_field)}. Re-estimate with the same settings.",
                      RuntimeWarning, stacklevel=2)
    C = model.patch_correlation(patch)
    d_eff = _effective_d(model.d, model.mask) if noise_field else None

    region = model.mask if mask_only else np.ones(model.mask.shape, bool)
    denoised, sigma_hat, rank = denoise_whitened(
        dwi, region, C, patch=tuple(int(v) for v in patch), algorithm=algorithm, tile=tile,
        d_field=d_eff, shrink=shrink, workers=workers, aggregate=aggregate, stride=stride)

    if d_eff is not None:
        # The engine rescales the IMAGE by d at the center voxel but not sigma_hat.
        # Do it here, or the scale of d leaks into the headline number.
        sigma_hat = sigma_hat * d_eff

    where = "center-voxel only" if aggregate == "center" else \
        f"mean over overlapping patches, stride {stride}"
    prov = dict(method="camppca", patch=tuple(int(v) for v in patch), algorithm=algorithm,
                noise_field=bool(noise_field), shrink=shrink, aggregate=aggregate,
                stride=stride, mask_only=bool(mask_only),
                rule=f"Cordero-Grande/Exp2, {RULES[shrink]}, {where}",
                model=dict(model.provenance))
    return DenoiseResult(denoised=denoised, sigma_hat=sigma_hat, rank=rank, model=model,
                         provenance=prov)


def denoise_camppca(dwi, bvals=None, mask=None, patch=None, noise_field=True,
                    algorithm="cordero-grande", tile=TILE, shrink="hard", workers=None,
                    aggregate="center", stride=1, mask_only=False, **estimate_kwargs):
    """Estimate the noise model from the data, then denoise with it.

    Equivalent to ``estimate_noise_model`` followed by ``denoise_with_noise_model``; the
    model is kept on the result (``result.model``), so the one-liner still exposes the
    intermediates. ``mask`` is where the model is estimated. Every voxel is denoised
    unless ``mask_only``.
    """
    check_shrink(shrink)                   # before the estimate, which can take a while
    check_aggregate(aggregate, stride)
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch, algorithm=algorithm,
                                 noise_field=noise_field, workers=workers,
                                 **estimate_kwargs)
    return denoise_with_noise_model(dwi, model, patch=patch, noise_field=noise_field,
                                    algorithm=algorithm, tile=tile, shrink=shrink, workers=workers,
                                    aggregate=aggregate, stride=stride,
                                    mask_only=mask_only)
