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

P2S is numpy/scipy. The loop kernel (``loop.py``) runs MP-PCA's own split, so it uses
JAX and the denoiser's internals.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

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

#: Smallest eigenvalue a patch correlation may have. A lag table can be valid for one
#: patch and indefinite for a larger one. `_build_C` then projects it to a C with
#: eigenvalues near zero, whitening divides by them, and sigma_hat collapses to ~1e-7
#: everywhere without an error.
MIN_EIGENVALUE = 1e-3


@dataclass
class NoiseModel:
    """Spatial noise model: correlation pattern ``rho`` plus level field ``d``.

    ``rho``        (2*lag+1, 2*lag+1, dzmax+1) correlation lag table, unit at zero lag.
                   PATTERN ONLY -- carries no level.
    ``d``          (X, Y, Z) relative level field, median-normalized.  Only ratios matter.
    ``sigma_p2s``  (X, Y, Z) absolute P2S noise level.  An INPUT that drives whitening --
                   not to be confused with ``sigma_hat``, the MP-PCA output.
    ``mask``       (X, Y, Z) bool, the voxels the model was estimated on.
    ``lag``        max in-plane lag ``rho`` covers; with ``rho.shape[2] - 1``, the
                   through-slice one, it bounds the patch this model can serve.
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
        check, the index runs off the table and numpy wraps around silently.  Also raises
        if ``C`` is near singular (see :data:`MIN_EIGENVALUE`).
        """
        patch = check_patch(patch)
        need = required_lag(patch)
        if need > self.lag:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs lag {need}, but this noise "
                f"model only covers lag {self.lag}. Re-estimate with the larger patch.")
        dzmax = self.rho.shape[2] - 1
        if patch[2] - 1 > dzmax:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs through-slice lag "
                f"{patch[2] - 1}, but this noise model only covers {dzmax}. Re-estimate "
                f"with the deeper patch.")
        C = _build_C(self.rho, patch, self.lag)
        smallest = float(np.linalg.eigvalsh(C)[0])
        if smallest < MIN_EIGENVALUE:
            raise ValueError(
                f"rho is not a valid correlation for patch "
                f"{tuple(int(v) for v in patch)}: the smallest eigenvalue of C is "
                f"{smallest:.2e}, below {MIN_EIGENVALUE:g}. Whitening divides by it, and "
                f"sigma_hat would collapse toward zero. Estimate the noise model at this "
                f"patch.")
        return C


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


def _pool_level(level, mask, size):
    """Neighbourhood-average a per-voxel level in VARIANCE, over mask voxels only.

    Args:
        level: (X, Y, Z) per-voxel noise level, zero outside ``mask``.
        mask: (X, Y, Z) bool, the voxels that count.
        size: box width in voxels. 1 leaves the field untouched.

    Returns:
        (X, Y, Z) pooled level, zero outside ``mask``.

    The average is over squares because the square is the unbiased quantity, being a
    residual second moment. Averaging levels instead biases low. Each voxel's squared
    level carries a relative sampling variance of 2/nu with nu residual columns, about 7%
    in the level at nu = 40, and averaging n voxels divides that by n. A 3^3 box takes it
    under 2%. The cost is blur, roughly h^2 * lap(d^2) / 24 for a box of width h, so
    ``size`` trades sampling noise against real structure.

    Both sums run over mask voxels only, so the edge is not pulled toward the background.
    ``mode="constant"`` extends that past the array edge: the default, ``reflect``, would
    mirror the rim back in and count it twice.
    """
    if size <= 1:
        return level
    kw = dict(size=size, mode="constant", cval=0.0)
    num = uniform_filter(np.where(mask, level, 0.0) ** 2, **kw)
    den = uniform_filter(mask.astype(np.float64), **kw)
    pooled = np.zeros_like(level)
    ok = mask & (den > 0)
    pooled[ok] = np.sqrt(num[ok] / den[ok])
    return pooled


def estimate_noise_model(dwi, bvals=None, mask=None, patch=(5, 5, 5), ncol=40, lowb=100,
                         seed=0, pool=3, kernel="loop", algo="cordero-grande",
                         noise_field=True):
    """Estimate the noise model from the data itself. No noise-only scan required.

    ``dwi``    (X, Y, Z, N) diffusion data.
    ``bvals``  (N,) b-values, used only to tell b0s from diffusion volumes.  Without them
               every volume is treated as a diffusion volume.
    ``mask``   (X, Y, Z) bool.  Estimated from the mean b0 when omitted.
    ``patch``  the patch this model must be able to serve; sets the lag rho covers.
    ``ncol``   residual columns subsampled for the rho fit.  Deterministic given ``seed``,
               so re-estimating gives an identical model.
    ``lowb``   b-value at or below which a volume counts as a b0.
    ``pool``   box width, in voxels, the relative level field ``d`` is averaged over.
               Averaged in variance, over mask voxels only; see :func:`_pool_level`.
               1 disables it. ``sigma_p2s`` is never pooled.
    ``kernel`` "loop" (default) refines the P2S pattern until it explains the MP-PCA bulk
               of the patches; see :mod:`camppca.noise_model.loop`. "p2s" returns the
               P2S pattern itself, which is always the loop's start.
    ``algo``, ``noise_field``  the denoiser settings the loop fits the kernel to: the MP
               rule, and whether rows are divided by ``d``. Pass the ones you will denoise
               with. Ignored for "p2s".

    One pattern is fit across all the diffusion volumes.  b0s are excluded from the fit:
    their residual level is not representative of diffusion-weighted noise.
    """
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X,Y,Z,N), got shape {dwi.shape}")
    n_vols = dwi.shape[3]
    patch = check_patch(patch)
    if np.any(patch > np.asarray(dwi.shape[:3])):
        # The denoiser wrap-pads, so a patch deeper than the volume holds one slice twice.
        raise ValueError(f"patch {tuple(int(v) for v in patch)} is larger than the "
                         f"volume {dwi.shape[:3]}")
    lag = required_lag(patch)
    dzmax = int(patch[2]) - 1
    if kernel not in ("loop", "p2s"):
        raise ValueError(f"kernel must be 'loop' or 'p2s', got {kernel!r}")
    from ..denoiser.mp_threshold import ALGORITHMS  # here: denoiser imports this package

    if algo not in ALGORITHMS:
        raise ValueError(f"algo must be one of {ALGORITHMS}, got {algo!r}")
    rng = np.random.default_rng(seed)
    pool = int(pool)
    if pool < 1:
        raise ValueError(f"pool must be >= 1 voxel, got {pool}")

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
    level = np.zeros(mask.shape, np.float64)
    level[mask] = sigma_vox
    level = _pool_level(level, mask, pool)
    d = np.zeros(mask.shape, np.float32)
    d[mask] = level[mask] / np.median(level[mask])  # median, not mean -- see CLAUDE.md

    prov = dict(patch=tuple(int(v) for v in np.asarray(patch, int)), lag=lag, dzmax=dzmax,
                ncol=int(ncol), seed=int(seed), lowb=float(lowb), pool=pool,
                n_mask=int(mask.sum()), n_vols=int(n_vols), n_dwi=int(dwi_idx.size),
                erode=int(erode_used), n_interior=int(interior.sum()),
                bvals_sha=_bvals_hash(bvals), kernel=kernel)

    if kernel == "loop":
        # Imported here: loop.py uses denoiser internals, and the denoiser imports this
        # package, so a top-level import would be circular.
        from .loop import loop_kernel
        rho, info = loop_kernel(dwi, mask, d, rho, patch, lag, seed, algo=algo,
                                noise_field=noise_field)
        prov.update(loop_rounds=info["rounds"], loop_converged=info["converged"],
                    u_kernel=info["u_kernel"], n_centres=info["n_centres"],
                    loop_algo=algo, loop_noise_field=bool(noise_field))

    model = NoiseModel(rho=rho, d=d, sigma_p2s=sigma_p2s, mask=mask, lag=lag,
                       provenance=prov)
    if kernel == "loop":
        # Fail here, not at denoising after the estimate is spent.
        try:
            model.patch_correlation(patch)
        except ValueError as err:
            raise ValueError(f"the loop kernel is not usable: {err} kernel='p2s' skips "
                             "the loop.") from err
    return model


def _bvals_hash(bvals):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(bvals, np.float64).tobytes()).hexdigest()[:16]
