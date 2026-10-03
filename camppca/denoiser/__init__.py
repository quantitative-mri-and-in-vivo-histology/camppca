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

import numpy as np

from ..noise_model import estimate_noise_model
from ..patch import check_patch
from .camppca import denoise_whitened
from .result import DenoiseResult
from .mppca import denoise_mppca

__all__ = ["DenoiseResult", "denoise_camppca", "denoise_mppca",
           "denoise_with_noise_model"]

D_CLIP = (0.2, 5.0)


def _effective_d(d, mask):
    """Median-normalize and clip the level field. The one place this happens.

    The engine takes the field as given and rescales the denoised image by it at the centre
    voxel, but NOT ``sigma_hat``. Putting sigma_hat back into data units therefore means
    applying this same field, which is why it is computed here and passed down.
    """
    d = np.asarray(d, np.float32)
    mask = np.asarray(mask, bool)          # a uint8 mask makes `d[good]` a fancy INDEX
    good = np.isfinite(d) & (d > 0) & mask
    if not good.any():
        raise ValueError("noise field d is empty or non-positive everywhere in the mask")
    d = d / np.median(d[good])
    return np.clip(np.where(np.isfinite(d) & (d > 0), d, 1.0), *D_CLIP)


def denoise_with_noise_model(dwi, model, patch=(5, 5, 5), noise_field=True,
                             algo="cordero-grande", tile=256, sigma_units="data"):
    """Denoise in the whitened domain, given a noise model.

    ``model``        a :class:`~camppca.noise_model.NoiseModel`.  Raises if ``patch`` is
                     wider than the lag the model measured -- never silently clipped.
    ``noise_field``  apply the model's relative level field ``d`` (Sigma = sigma^2 D C D).
                     Adds the WITHIN-patch heteroscedasticity, which widens the noise bulk
                     the same way off-diagonal colour does.
    ``sigma_units``  "data" (default) returns ``sigma_hat`` in the input's units; "d" leaves
                     it d-normalized, as the engine produces it.

    The rule -- Exp2 / Cordero-Grande, hard threshold, centre-voxel-only -- is the method,
    not a knob.
    """
    if sigma_units not in ("data", "d"):
        raise ValueError(f"sigma_units must be 'data' or 'd', got {sigma_units!r}")
    patch = check_patch(patch)
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X,Y,Z,N), got shape {dwi.shape}")
    if dwi.shape[:3] != model.mask.shape:
        raise ValueError(f"dwi spatial shape {dwi.shape[:3]} != model mask "
                         f"{model.mask.shape}")

    C = model.patch_correlation(patch)
    d_eff = _effective_d(model.d, model.mask) if noise_field else None

    denoised, sigma_hat, rank = denoise_whitened(
        dwi, model.mask, C, patch=tuple(int(v) for v in patch), algo=algo, tile=tile,
        d_field=d_eff)

    if d_eff is not None and sigma_units == "data":
        # The engine rescales the IMAGE by d at the centre voxel but not sigma_hat.
        # Do it here, or the normalization convention leaks into the headline number.
        sigma_hat = sigma_hat * d_eff

    prov = dict(method="camppca", patch=tuple(int(v) for v in patch), algo=algo,
                noise_field=bool(noise_field), sigma_units=sigma_units,
                rule="Cordero-Grande/Exp2, hard threshold, no shrinkage, centre-voxel only",
                model=dict(model.provenance))
    return DenoiseResult(denoised=denoised, sigma_hat=sigma_hat, rank=rank, model=model,
                         provenance=prov)


def denoise_camppca(dwi, bvals=None, mask=None, patch=(5, 5, 5), noise_field=True,
                    algo="cordero-grande", tile=256, sigma_units="data",
                    **estimate_kwargs):
    """Estimate the noise model from the data, then denoise with it.

    Equivalent to ``estimate_noise_model`` followed by ``denoise_with_noise_model``; the
    model is kept on the result (``result.model``), so the one-liner still exposes the
    intermediates.
    """
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch,
                                 **estimate_kwargs)
    return denoise_with_noise_model(dwi, model, patch=patch, noise_field=noise_field,
                                    algo=algo, tile=tile, sigma_units=sigma_units)
