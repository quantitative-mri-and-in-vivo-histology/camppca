"""The denoisers: the engine half of the seam.

Source-agnostic by design. ``denoise_with_noise_model`` consumes a
:class:`~camppca.noise_model.NoiseModel` and never asks where it came from, so a future
alignment-based estimator feeds the same code.

``camppca.py`` is the covariance-aware method, ``mppca.py`` the standard white-noise one.
What they share lives in ``mp_threshold.py``, ``gather.py`` and ``result.py``, next to
them. The loop in ``noise_model/loop.py`` reuses the split and
``sqrtm_psd`` from here. The level guard belongs to the noise model:
:meth:`~camppca.noise_model.NoiseModel.effective_d`.

Every public name here is re-exported from ``camppca`` itself.
"""


import numpy as np

from .. import __version__
from ..noise_model import estimate_noise_model
from ..patch import resolve_patch
from ..pool import limit_blas
from ..warn import warn
from .aggregate import check_aggregate
from .camppca import denoise_whitened
from .gather import check_dwi
from .mp_threshold import check_algorithm
from .mppca import denoise_mppca
from .result import DenoiseResult
from .shrink import check_shrink, rule_text

__all__ = ["DenoiseResult", "denoise_camppca", "denoise_mppca",
           "denoise_with_noise_model"]

def _flat_mode_factor(C, aggregate="center"):
    """The factor that takes the flat mode out of the reported noise level.

    The loop cannot see one direction of the table, the same value added at every lag. It
    changes the d-divided patch's noise only along the flat mode u, u_i = 1/sqrt(n_vox),
    and is fixed by assuming the far lags average to zero. A voxel's full noise level
    leans on that assumption, so the reported level leaves the flat mode out:
    ``sigma^2 = sigma_mppca^2 * d^2 * n_vox/(n_vox-1) * [(I - u u^T) C (I - u u^T)]_jj``.
    ``j`` is the center voxel, or with ``aggregate="mean"`` the mean
    over the patch, since a voxel then sits at every position once. The factor is one
    number for the whole image, and 1 for white noise.
    """
    C = np.asarray(C, np.float64)
    n_vox = C.shape[0]
    centered = C - C.mean(0, keepdims=True)
    projected = centered - centered.mean(1, keepdims=True)    # (I - uu^T) C (I - uu^T)
    diag = projected[n_vox // 2, n_vox // 2] if aggregate == "center" else \
        np.trace(projected) / n_vox
    return float(np.sqrt(n_vox / (n_vox - 1) * diag))


@limit_blas
def denoise_with_noise_model(dwi, model, *, patch=None, noise_field=True,
                             algorithm="cordero-grande", shrink="hard", workers=None,
                             aggregate="center", stride=1, mask_only=False):
    """Denoise in the whitened domain, given a noise model.

    The rule is the method: the MP cut ``algorithm``, center voxel only, and a hard
    threshold, unless ``shrink`` or ``aggregate`` say otherwise.

    Args:
        dwi: (X, Y, Z, N) data, real or complex, on the model's voxel grid.
        model: a :class:`~camppca.noise_model.NoiseModel`.
        patch: None takes the patch the model was estimated for, the one its loop fitted
            to, or dwidenoise's default if the model records none
            (:func:`~camppca.patch.default_patch`).
        noise_field: apply the model's relative noise map ``d`` (Sigma = sigma^2 D C D).
            It adds the within-patch change of level, which widens the noise bulk the
            same way correlation does.
        algorithm: which Marchenko-Pastur variant to cut with, see
            :data:`~camppca.denoiser.mp_threshold.ALGORITHMS`. A mismatch with the
            model's ``fitted_to`` warns.
        shrink: weight on the kept components, "hard" (default), "gd" or "lr", see
            :mod:`camppca.denoiser.shrink`.
        workers: CPU processes to split the voxels over, see :mod:`camppca.pool`. The
            default is one per physical core. 1 runs in this process.
        aggregate: "center" (default) writes back each patch's center voxel. "mean"
            averages the overlapping patch reconstructions, see
            :mod:`camppca.denoiser.aggregate`.
        stride: with "mean", a patch center every ``stride`` voxels per axis.
        mask_only: denoise only inside the model's mask and pass the rest through. By
            default every voxel is denoised. The mask only says where the model was
            estimated, and a brain mask always misses some brain at its edge. Outside it
            ``d`` takes the nearest mask voxel's level.

    Returns:
        A :class:`~camppca.denoiser.result.DenoiseResult` carrying ``model``.

    Raises:
        ValueError: if ``dwi`` is not 4D on the model's grid, or ``patch`` needs lags the
            model does not have: wider on any axis than ``fitted_to["patch"]``, or than
            the table reaches. Never silently clipped.
    """
    shrink = check_shrink(shrink)
    check_algorithm(algorithm)
    aggregate, stride = check_aggregate(aggregate, stride)
    dwi = check_dwi(dwi, model.mask.shape)
    patch = resolve_patch(model.fitted_to.get("patch") if patch is None else patch,
                          dwi.shape)

    # Only the settings the model records: a second estimator may leave keys out.
    used = dict(algorithm=algorithm, noise_field=bool(noise_field))
    differ = [k for k in used if k in model.fitted_to and model.fitted_to[k] != used[k]]
    if differ:
        fitted = ", ".join(f"{k}={model.fitted_to[k]!r}" for k in differ)
        now = ", ".join(f"{k}={used[k]!r}" for k in differ)
        warn(f"the loop was fitted with {fitted}, but this denoise uses {now}. "
             f"Re-estimate with the same settings.")
    C = model.patch_correlation(patch)
    d_eff = model.effective_d() if noise_field else None

    region = model.mask if mask_only else np.ones(model.mask.shape, bool)
    denoised, sigma_mppca, rank = denoise_whitened(
        dwi, region, C, patch=tuple(int(v) for v in patch), algorithm=algorithm,
        d_field=d_eff, shrink=shrink, workers=workers, aggregate=aggregate, stride=stride,
        progress_label="denoising")

    if d_eff is not None:
        # The engine rescales the IMAGE by d at the center voxel but not sigma_mppca.
        # Do it here, or the scale of d leaks into the headline number.
        sigma_mppca = sigma_mppca * d_eff
    flat_mode = _flat_mode_factor(C, aggregate)
    sigma = sigma_mppca * flat_mode

    prov = dict(method="camppca", version=__version__, patch=tuple(int(v) for v in patch),
                algorithm=algorithm,
                noise_field=bool(noise_field), shrink=shrink, aggregate=aggregate,
                stride=stride, mask_only=bool(mask_only), flat_mode_factor=flat_mode,
                rule=rule_text(algorithm, shrink, aggregate, stride),
                model=dict(model.provenance))
    return DenoiseResult(denoised=denoised, sigma=sigma, rank=rank, model=model,
                         provenance=prov)


def denoise_camppca(dwi, mask=None, *, bvals=None, patch=None, noise_field=True,
                    algorithm="cordero-grande", shrink="hard", workers=None,
                    aggregate="center", stride=1, mask_only=False, **estimate_kwargs):
    """Estimate the noise model from the data, then denoise with it.

    Equivalent to :func:`~camppca.estimate_noise_model` followed by
    :func:`denoise_with_noise_model`. The model is kept on the result (``result.model``),
    so the one-liner still exposes the intermediates.

    Args:
        dwi: (X, Y, Z, N) data, real or complex.
        mask: where the noise model is estimated, see
            :func:`~camppca.estimate_noise_model`. Every voxel is denoised unless
            ``mask_only``.
        **estimate_kwargs: the estimator's other options, ``lowb`` and ``level_rounds``.

    The other arguments are those of :func:`~camppca.estimate_noise_model` and
    :func:`denoise_with_noise_model`. ``algorithm`` and ``noise_field`` go to both, so the
    loop fits to the settings the denoise uses.

    Returns:
        A :class:`~camppca.denoiser.result.DenoiseResult`.
    """
    check_shrink(shrink)                   # before the estimate, which can take a while
    check_aggregate(aggregate, stride)
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch, algorithm=algorithm,
                                 noise_field=noise_field, workers=workers,
                                 **estimate_kwargs)
    return denoise_with_noise_model(dwi, model, patch=patch, noise_field=noise_field,
                                    algorithm=algorithm, shrink=shrink, workers=workers,
                                    aggregate=aggregate, stride=stride,
                                    mask_only=mask_only)
