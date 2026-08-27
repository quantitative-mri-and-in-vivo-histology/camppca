"""The denoiser: the engine half of the seam.

Named ``denoiser`` and not ``denoise`` so the subpackage does not shadow the
``denoise()`` function on ``camppca`` itself.

Source-agnostic by design -- ``denoise_with_noise_model`` consumes a
:class:`~camppca.noise_model.NoiseModel` and never asks where it came from, so a future
alignment-based estimator feeds the same code.

``camppca.py`` is the method; ``mppca.py`` is the plain white-noise baseline arm plus the
Marchenko-Pastur threshold both share. Both now speak CLAUDE.md's notation directly -- see
their module headers for the rename from the reference spelling.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..noise_model import NoiseModel, estimate_noise_model
from ..patch import check_patch
from .camppca import denoise_white_percol

__all__ = ["DenoiseResult", "denoise", "denoise_with_noise_model"]

D_CLIP = (0.2, 5.0)


@dataclass
class DenoiseResult:
    """Denoised data plus the calibrated noise level the MP fit found.

    ``denoised``   (X, Y, Z, N) denoised data.  Voxels outside the mask are passed through
                   unchanged -- the engine is centre-voxel-only, with no overlap averaging.
    ``sigma_hat``  (X, Y, Z) MP-PCA's noise estimate, in the input's units.  An OUTPUT --
                   distinct from ``model.sigma_p2s``, the P2S level that drove whitening.
    ``rank``       (X, Y, Z) retained rank per voxel.
    ``model``      the noise model used, so the one-liner still exposes the intermediates.
    """

    denoised: np.ndarray
    sigma_hat: np.ndarray
    rank: np.ndarray
    model: NoiseModel
    provenance: dict = field(default_factory=dict)


def _effective_d(d, mask):
    """The level field exactly as the engine normalizes and clips it internally.

    ⚠ MIRRORS ``camppca.denoise_white_percol`` (the ``d_field`` block).  It has to: the
    engine rescales the denoised image by this field at the centre voxel but NOT
    ``sigma_hat``, so putting sigma_hat back into data units means applying the same
    field.  Keep the two in sync.
    """
    d = np.asarray(d, np.float32)
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
    if dwi.shape[3] != model.group.size:
        raise ValueError(f"dwi has {dwi.shape[3]} volumes, model covers "
                         f"{model.group.size}")

    Cs = model.patch_correlation(patch)
    d_eff = _effective_d(model.d, model.mask) if noise_field else None

    denoised, sigma_hat, rank = denoise_white_percol(
        dwi, model.mask, Cs, model.group, column_scale=model.column_scale(),
        patch=tuple(int(v) for v in patch), algo=algo, tile=tile, d_field=d_eff)

    if d_eff is not None and sigma_units == "data":
        # The engine rescales the IMAGE by d at the centre voxel but not sigma_hat
        # (camppca.py, the `xh = xh * gp[:, center, None]` line). Do it here, or the
        # normalization convention leaks into the headline number.
        sigma_hat = sigma_hat * d_eff

    prov = dict(patch=tuple(int(v) for v in patch), algo=algo, noise_field=bool(noise_field),
                sigma_units=sigma_units, rule="Cordero-Grande/Exp2, hard threshold, "
                "no shrinkage, centre-voxel only", model=dict(model.provenance))
    return DenoiseResult(denoised=denoised, sigma_hat=sigma_hat, rank=rank, model=model,
                         provenance=prov)


def denoise(dwi, bvals=None, mask=None, patch=(5, 5, 5), noise_field=True,
            per_shell="auto", algo="cordero-grande", tile=256, sigma_units="data",
            **estimate_kwargs):
    """Estimate the noise model from the data, then denoise with it.

    Equivalent to ``estimate_noise_model`` followed by ``denoise_with_noise_model``; the
    model is kept on the result (``result.model``), so the one-liner still exposes the
    intermediates.
    """
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch,
                                 per_shell=per_shell, **estimate_kwargs)
    return denoise_with_noise_model(dwi, model, patch=patch, noise_field=noise_field,
                                    algo=algo, tile=tile, sigma_units=sigma_units)
