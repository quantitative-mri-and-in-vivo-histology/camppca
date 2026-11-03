"""The noise model: what the noise IS, and how to estimate it from the data.

The patch noise covariance is assumed to factorize into a level that varies in space and
a correlation pattern that does not::

    Sigma = sigma^2 * D * C * D          D = diag(d)

``rho`` (the lag table) and ``d`` (the relative level field) are the model.  ``C``, the
P x P patch correlation, is DERIVED from rho plus a patch shape -- a denoiser knob, not a
property of the noise -- so it is computed on demand by :meth:`NoiseModel.patch_correlation`
and never stored.

This half of the seam is source-specific: here both come from MP-PCA itself. ``d`` is
standard MP-PCA's noise level, pooled, and rho is the loop kernel (``loop.py``) started
from white. A future alignment-based estimator builds the same :class:`NoiseModel` a
different way and feeds the same engine, so nothing downstream may assume this one.

The loop runs MP-PCA's own split, so it uses the denoiser's internals.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter

from ..patch import check_patch, patch_offsets, required_lag, resolve_patch
from .masks import brain_mask

__all__ = ["NoiseModel", "estimate_noise_model", "brain_mask"]

#: Fewest volumes the Marchenko-Pastur fit is trusted with. Below this the Exp2 width
#: normalizer degenerates as the rank approaches the volume count: at 125 voxels and 34
#: volumes the noise level came out 52% high against a true 5%.
MIN_VOLUMES = 50

#: Stop the level rounds when the median relative change of ``d`` drops below this.
#: On real data the first round moved ``d`` by 3.5 to 5%, the second by under 1%.
LEVEL_TOL = 5e-3

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
    ``d``          (X, Y, Z) level field, defined up to scale: an MP-PCA noise level
                   pooled over a ``pool``^3 box.  Only ratios matter.  Zero outside
                   ``mask``.
    ``mask``       (X, Y, Z) bool, the voxels the model was estimated on.
    ``lag``        max in-plane lag ``rho`` covers; with ``rho.shape[2] - 1``, the
                   through-slice one, it bounds the patch this model can serve.
    """

    rho: np.ndarray
    d: np.ndarray
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
    second moment. Averaging levels instead biases low. Pooling also smooths the grain an
    MP level map carries where neighbouring voxels cut at different ranks; unpooled, that
    grain made the loop's ``C`` near singular on strongly correlated data. The cost is
    blur, roughly h^2 * lap(d^2) / 24 for a box of width h, so ``size`` trades sampling
    noise against real structure.

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


def estimate_noise_model(dwi, bvals=None, mask=None, patch=None, lowb=100, seed=0,
                         pool=3, algo="cordero-grande", noise_field=True, level_rounds=0,
                         workers=None):
    """Estimate the noise model from the data itself. No noise-only scan required.

    ``dwi``    (X, Y, Z, N) diffusion data.  Every volume enters, b0s included, because the
               denoiser whitens them all.
    ``bvals``  (N,) b-values, used only to find the b0s the default mask is drawn from.
    ``mask``   (X, Y, Z) bool.  Estimated from the mean b0 when omitted.
    ``patch``  the patch this model must be able to serve; sets the lag rho covers.
               None picks dwidenoise's default, see :func:`~camppca.patch.default_patch`.
    ``lowb``   b-value at or below which a volume counts as a b0.
    ``seed``   picks the loop's patch centres, so re-estimating gives an identical model.
    ``pool``   box width, in voxels, the level field ``d`` is averaged over.  Averaged in
               variance, over mask voxels only; see :func:`_pool_level`.  1 disables it.
    ``algo``, ``noise_field``  the denoiser settings the model is fitted to: the MP rule,
               and whether rows are divided by ``d``.  Pass the ones you will denoise with.
    ``level_rounds``  how many times ``d`` is re-estimated from the whitened patches.
               0, the default, keeps standard MP-PCA's level.
    ``workers``  CPU processes for the passes over every voxel and the loop's
               split, see :mod:`camppca.pool`. The default is one per physical core.
               1 runs everything in this process.

    The level field ``d`` is standard MP-PCA's ``sigma_hat`` at every mask voxel, pooled
    ``pool``^3.  Unpooled, its rank grain made ``C`` near singular on strongly correlated
    data.  The kernel is the loop's (:mod:`camppca.noise_model.loop`), started from white
    noise: 1 at lag 0, 0 elsewhere.

    Each level round splits every mask voxel's whitened patch, which gives ``s``, the
    level in d-normalized units.  The new ``d`` is ``d * s`` pooled, and the loop is
    refitted from the previous table.  The rounds stop when ``d`` moves by less than
    :data:`LEVEL_TOL`.  At the fixed point ``s`` is flat: every whitened patch carries the
    same noise level.  On real data this changed U by at most 0.006 and ``sigma_hat`` by
    at most 0.3%, at three to four times the cost, hence the default of 0.

    With ``noise_field=False``, ``d`` is 1 in the mask and only the loop runs.
    """
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError(f"dwi must be 4D (X,Y,Z,N), got shape {dwi.shape}")
    n_vols = dwi.shape[3]
    patch = resolve_patch(patch, dwi.shape)
    if np.any(patch > np.asarray(dwi.shape[:3])):
        # The denoiser wrap-pads, so a patch deeper than the volume holds one slice twice.
        raise ValueError(f"patch {tuple(int(v) for v in patch)} is larger than the "
                         f"volume {dwi.shape[:3]}")
    lag = required_lag(patch)
    dzmax = int(patch[2]) - 1
    from ..denoiser.mp_threshold import ALGORITHMS  # here: denoiser imports this package

    if algo not in ALGORITHMS:
        raise ValueError(f"algo must be one of {ALGORITHMS}, got {algo!r}")
    pool = int(pool)
    if pool < 1:
        raise ValueError(f"pool must be >= 1 voxel, got {pool}")
    level_rounds = int(level_rounds)
    if level_rounds < 0:
        raise ValueError(f"level_rounds must be >= 0, got {level_rounds}")
    if n_vols < MIN_VOLUMES:
        warnings.warn(f"{n_vols} volumes, fewer than {MIN_VOLUMES}: the Marchenko-Pastur "
                      "fit degenerates as the rank approaches the volume count, and the "
                      "noise level can come out far too high.", RuntimeWarning, stacklevel=2)

    if bvals is None:
        bvals = np.zeros(n_vols)
    else:
        bvals = np.asarray(bvals, float).ravel()
        if bvals.size != n_vols:
            raise ValueError(f"bvals has {bvals.size} entries, dwi has {n_vols} volumes")

    if mask is None:
        b0 = np.where(bvals <= lowb)[0]
        b0_mean = dwi[..., b0].mean(-1) if b0.size else dwi.mean(-1)
        # erode=0: the stored mask is also the set of voxels the engine WRITES, so
        # eroding here would leave an un-denoised rim.
        mask = brain_mask(b0_mean, erode=0)
    mask = np.asarray(mask, bool)
    if mask.shape != dwi.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != dwi spatial shape {dwi.shape[:3]}")

    # Imported here: these use denoiser internals, and the denoiser imports this package,
    # so a top-level import would be circular.
    from ..denoiser import _effective_d
    from ..denoiser.camppca import whitened_level
    from .loop import loop_kernel

    rho = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    rho[lag, lag, 0] = 1.0
    if noise_field:
        level = whitened_level(dwi, mask, None, patch, algo, workers=workers)
        d = _level_field(level, mask, pool)
    else:
        d = mask.astype(np.float32)
    rho, info = loop_kernel(dwi, mask, d, rho, patch, lag, seed, algo=algo,
                            noise_field=noise_field, workers=workers)

    rounds_run, change, spread = 0, None, None
    for rounds_run in range(1, (level_rounds if noise_field else 0) + 1):  # noqa: B007
        d_eff = _effective_d(d, mask)
        s = whitened_level(dwi, mask, _build_C(rho, patch, lag), patch, algo,
                           d_field=d_eff, workers=workers)
        q25, q50, q75 = np.percentile(s[mask], (25, 50, 75))
        spread = float((q75 - q25) / q50)
        new = _level_field(d_eff * s, mask, pool)
        change = float(np.median(np.abs((new[mask] / np.median(new[mask]))
                                        / (d[mask] / np.median(d[mask])) - 1)))
        if change < LEVEL_TOL:
            break
        d = new
        rho, info = loop_kernel(dwi, mask, d, rho, patch, lag, seed, algo=algo,
                                noise_field=noise_field, workers=workers)

    prov = dict(patch=tuple(int(v) for v in np.asarray(patch, int)), lag=lag, dzmax=dzmax,
                seed=int(seed), lowb=float(lowb), pool=pool, n_mask=int(mask.sum()),
                n_vols=int(n_vols), bvals_sha=_bvals_hash(bvals),
                loop_rounds=info["rounds"], loop_converged=info["converged"],
                u_kernel=info["u_kernel"], n_centres=info["n_centres"],
                loop_algo=algo, loop_noise_field=bool(noise_field),
                level_rounds=level_rounds, level_rounds_run=rounds_run)
    if rounds_run:
        prov.update(level_converged=change < LEVEL_TOL, level_change=change,
                    s_spread=spread)

    model = NoiseModel(rho=rho, d=d, mask=mask, lag=lag, provenance=prov)
    # Fail here, not at denoising after the estimate is spent.
    try:
        model.patch_correlation(patch)
    except ValueError as err:
        raise ValueError(f"the loop kernel is not usable: {err}") from err
    return model


def _level_field(level, mask, pool):
    """A per-voxel level pooled ``pool``^3 into a level field, zero outside ``mask``."""
    pooled = _pool_level(np.where(mask, level, 0.0).astype(np.float64), mask, pool)
    return np.where(mask, pooled, 0.0).astype(np.float32)


def _bvals_hash(bvals):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(bvals, np.float64).tobytes()).hexdigest()[:16]
