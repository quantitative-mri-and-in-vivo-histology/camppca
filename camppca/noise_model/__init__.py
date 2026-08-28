"""The noise model: what the noise IS, and how to estimate it from the data.

The patch noise covariance is assumed to factorize into a level that varies in space and
a correlation pattern that does not::

    Sigma = sigma^2 * D * C * D          D = diag(d)

``rho`` (the lag table) and ``d`` (the relative level field) are the model.  ``C``, the
P x P patch correlation, is DERIVED from rho plus a patch shape -- a denoiser knob, not a
property of the noise -- so it is computed on demand by :meth:`NoiseModel.patch_correlation`
and never stored.

This half of the seam is source-specific: here rho comes from Patch2Self residuals.  A
future alignment-based estimator builds the same :class:`NoiseModel` a different way and
feeds the same engine, so nothing downstream may assume P2S.

Pure numpy/scipy. Nothing here imports JAX.
"""

from dataclasses import dataclass, field

import numpy as np

from ..patch import check_patch, patch_offsets, required_lag
from .masks import brain_mask, interior_region, stable_region
from .p2s import noise_map, p2s_ols, rho_region

__all__ = ["NoiseModel", "estimate_noise_model", "brain_mask", "stable_region"]

#: Fewest diffusion volumes P2S can fit a correlation pattern from. Below this the
#: leave-one-out fit degenerates: at 1 volume there are no predictors at all, every
#: residual is the centred signal, `stable_region` drops every voxel and `rho` comes back
#: all-NaN, which propagates to a fully NaN denoised image. At 2 volumes it does not crash
#: but the level is badly biased, measured ~50% high.
MIN_VOLUMES = 10


@dataclass
class NoiseModel:
    """Spatial noise model: correlation pattern ``rho`` plus level field ``d``.

    ``rho``        (2*lag+1, 2*lag+1, dzmax+1) correlation lag table, unit at zero lag.
                   PATTERN ONLY -- carries no level.
    ``d``          (X, Y, Z) relative level field, median-normalized.  Only ratios matter.
    ``sigma_p2s``  (X, Y, Z) absolute P2S noise level.  An INPUT that drives whitening --
                   not to be confused with ``sigma_hat``, the MP-PCA output.
    ``mask``       (X, Y, Z) bool, the voxels the model was estimated on.
    ``lag``        max lag ``rho`` covers; bounds the patch this model can serve.
    """

    rho: np.ndarray
    d: np.ndarray
    sigma_p2s: np.ndarray
    mask: np.ndarray
    lag: int
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        # A uint8 mask (what `nib.load(...).get_fdata().astype(...)` hands you) would turn
        # `... & mask` into integer fancy indexing downstream, silently mis-normalizing d.
        self.mask = np.asarray(self.mask, bool)
        self.rho = np.asarray(self.rho, float)

    def patch_correlation(self, patch):
        """Derive ``C`` (P, P), the unit-diagonal patch correlation.

        ``C`` is ``rho`` re-indexed into the geometry of this patch, PSD-projected to unit
        diagonal.  Raises if the patch needs lags this model never measured: without the
        check, the index runs off the table and numpy wraps around silently.
        """
        patch = check_patch(patch)
        need = required_lag(patch)
        if need > self.lag:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs lag {need}, but this noise "
                f"model only covers lag {self.lag}. Re-estimate with the larger patch.")
        return _build_C(self.rho, patch, self.lag)


def _build_C(rho, patch, lag):
    """Patch correlation from a lag table ``rho[dx+lag, dy+lag, |dz|]``; no wraparound.

    Projected to the nearest PSD matrix with unit diagonal if the measured table is
    indefinite.  The bounds check that guards it lives in
    :meth:`NoiseModel.patch_correlation`.
    """
    o = patch_offsets(patch)
    dx = o[:, 0][:, None] - o[:, 0][None, :]
    dy = o[:, 1][:, None] - o[:, 1][None, :]
    dz = np.abs(o[:, 2][:, None] - o[:, 2][None, :])
    R = rho[dx + lag, dy + lag, np.minimum(dz, rho.shape[2] - 1)]
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-9:
        R = (V * np.maximum(w, 1e-9)) @ V.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    return R


def estimate_noise_model(dwi, bvals=None, mask=None, patch=(5, 5, 5), ncol=40, lowb=100,
                         seed=0):
    """Estimate the noise model from the data itself. No noise-only scan required.

    ``dwi``    (X, Y, Z, N) diffusion data.
    ``bvals``  (N,) b-values, used only to tell b0s from diffusion volumes.  Without them
               every volume is treated as a diffusion volume.
    ``mask``   (X, Y, Z) bool.  Estimated from the mean b0 when omitted.
    ``patch``  the patch this model must be able to serve; sets the lag rho covers.
    ``ncol``   residual columns subsampled for the rho fit.  Deterministic given ``seed``,
               so re-estimating gives an identical model.
    ``lowb``   b-value at or below which a volume counts as a b0.

    One pattern is fit across all the diffusion volumes.  b0s are excluded from the fit:
    their residual level is not representative of diffusion-weighted noise.
    """
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X,Y,Z,N), got shape {dwi.shape}")
    n_vols = dwi.shape[3]
    lag = required_lag(check_patch(patch))
    dzmax = lag
    rng = np.random.default_rng(seed)

    if bvals is None:
        bvals = np.zeros(n_vols)
        dwi_idx = np.arange(n_vols)
    else:
        bvals = np.asarray(bvals, float).ravel()
        if bvals.size != n_vols:
            raise ValueError(f"bvals has {bvals.size} entries, dwi has {n_vols} volumes")
        dwi_idx = np.where(bvals > lowb)[0]
    if dwi_idx.size < MIN_VOLUMES:
        raise ValueError(
            f"need >= {MIN_VOLUMES} volumes above b={lowb} to fit a correlation pattern, "
            f"found {dwi_idx.size}. Below this the leave-one-out fit degenerates and rho "
            f"comes back all-NaN, which propagates to a fully NaN denoised image.")

    if mask is None:
        b0 = np.where(bvals <= lowb)[0]
        b0_mean = dwi[..., b0].mean(-1) if b0.size else dwi.mean(-1)
        # erode=0: the stored mask is also the set of voxels the engine WRITES, so
        # eroding here would leave an un-denoised rim. The strict interior that
        # rho_region needs is derived separately, below.
        mask = brain_mask(b0_mean, erode=0)
    mask = np.asarray(mask, bool)
    if mask.shape != dwi.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != dwi spatial shape {dwi.shape[:3]}")
    n_mask = int(mask.sum())
    if n_mask <= dwi_idx.size:
        raise ValueError(
            f"mask holds {n_mask} voxels but P2S fits up to {dwi_idx.size - 1} predictors "
            f"per volume -- the regression is singular. Need V >> N; V/N here is "
            f"{n_mask / max(dwi_idx.size, 1):.1f}.")

    interior, erode_used = interior_region(mask)

    # ---- P2S: sigma from every diffusion column, rho from a subsample of them ------------
    residuals, inflation = p2s_ols(dwi[mask][:, dwi_idx].astype(np.float64))
    sigma_vox = noise_map(residuals, inflation, np.arange(dwi_idx.size))

    keep = np.sort(rng.permutation(residuals.shape[1])[:ncol])
    subsample = residuals[:, keep].astype(np.float32)
    del residuals
    region, _ = stable_region(subsample, mask)
    region &= interior           # rho_region wants the strict interior, not the rim
    if not region.any():
        raise ValueError("estimation region is empty after dropping constant voxels "
                         "and eroding to the interior -- check the mask")
    rho = rho_region(subsample, mask, region, lag=lag, dzmax=dzmax)
    del subsample

    sigma_p2s = np.zeros(mask.shape, np.float32)
    sigma_p2s[mask] = sigma_vox
    d = np.zeros(mask.shape, np.float32)
    d[mask] = sigma_vox / np.median(sigma_vox)      # median, not mean -- see CLAUDE.md

    prov = dict(patch=tuple(int(v) for v in np.asarray(patch, int)), lag=lag, dzmax=dzmax,
                ncol=int(ncol), seed=int(seed), lowb=float(lowb),
                n_mask=int(mask.sum()), n_vols=int(n_vols), n_dwi=int(dwi_idx.size),
                erode=int(erode_used), n_interior=int(interior.sum()),
                bvals_sha=_bvals_hash(bvals))
    return NoiseModel(rho=rho, d=d, sigma_p2s=sigma_p2s, mask=mask, lag=lag,
                      provenance=prov)


def _bvals_hash(bvals):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(bvals, np.float64).tobytes()).hexdigest()[:16]
