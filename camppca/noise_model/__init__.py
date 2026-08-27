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
from typing import Any

import numpy as np

from ..patch import patch_offsets, patch_size, required_lag, check_patch
from .masks import brain_mask, stable_region
from .p2s import noise_map, p2s_ols, rho_region

__all__ = ["NoiseModel", "estimate_noise_model", "brain_mask", "stable_region"]


@dataclass
class NoiseModel:
    """Spatial noise model: correlation pattern ``rho`` plus level field ``d``.

    ``rho``        (S, 2*lag+1, 2*lag+1, dzmax+1) correlation lag tables, one per shell,
                   unit at zero lag.  PATTERN ONLY -- carries no level.
    ``group``      (N,) volume -> shell index, indexing ``rho``.
    ``d``          (X, Y, Z) relative level field, median-normalized.  Only ratios matter.
    ``sigma_p2s``  (X, Y, Z) absolute P2S noise level.  An INPUT that drives whitening --
                   not to be confused with ``sigma_hat``, the MP-PCA output.
    ``sigma_shell``(S,) noise level per shell.  Only RATIOS matter: whitening with
                   patterns of different amplitude gives C_s^(-1/2) different gains, so
                   the per-shell level must be divided out alongside C_s or spatial
                   correlation is merely traded for column heteroscedasticity.
    ``mask``       (X, Y, Z) bool, the voxels the model was estimated on.
    ``lag``        max lag ``rho`` covers; bounds the patch this model can serve.
    """

    rho: np.ndarray
    group: np.ndarray
    d: np.ndarray
    sigma_p2s: np.ndarray
    sigma_shell: np.ndarray
    mask: np.ndarray
    lag: int
    provenance: dict = field(default_factory=dict)

    @property
    def n_shells(self):
        return int(self.rho.shape[0])

    def column_scale(self):
        """(N,) per-volume noise scale for the engine: ``sigma_shell[group]``, mean 1.

        Mean-normalized so ``sigma_hat`` comes back in the input's units; only ratios
        enter the whitening, so the choice of normalizer cancels there.
        """
        per_vol = np.asarray(self.sigma_shell, float)[self.group]
        return per_vol / per_vol.mean()

    def patch_correlation(self, patch):
        """Derive ``Cs`` (S, P, P), the unit-diagonal patch correlation per shell.

        ``C`` is ``rho`` re-indexed into the geometry of this patch, PSD-projected to unit
        diagonal.  Raises if the patch needs lags this model never measured -- upstream
        this ran off the table and numpy answered with a silent wraparound.
        """
        patch = check_patch(patch)
        need = required_lag(patch)
        if need > self.lag:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs lag {need}, but this noise "
                f"model only covers lag {self.lag}. Re-estimate with the larger patch.")
        return np.stack([_build_C(r, patch, self.lag) for r in self.rho])


def _build_C(rho, patch, lag):
    """Patch correlation from a lag table ``rho[dx+lag, dy+lag, |dz|]``; no wraparound.

    Projected to the nearest PSD matrix with unit diagonal if the measured table is
    indefinite.  Body ported verbatim from ``whitened.build_R``; the bounds check that
    guards it lives in :meth:`NoiseModel.patch_correlation`.
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


def estimate_noise_model(dwi, bvals=None, mask=None, patch=(5, 5, 5), per_shell="auto",
                         ncol=40, lowb=100, shell_tol=250, seed=0):
    """Estimate the noise model from the data itself. No noise-only scan required.

    ``dwi``        (X, Y, Z, N) diffusion data.
    ``bvals``      (N,) b-values.  Required for per-shell estimation; without them
                   everything above ``lowb`` is treated as one shell.
    ``mask``       (X, Y, Z) bool.  Estimated from the mean b0 when omitted.
    ``patch``      the patch this model must be able to serve; sets the lag rho covers.
    ``per_shell``  "auto" (per-shell when bvals resolve >1 shell), True, or False.
    ``ncol``       residual columns subsampled for the rho fit.  Deterministic given
                   ``seed``, so re-estimating gives an identical model.
    ``lowb``       b-value at or below which a volume counts as a b0.
    ``shell_tol``  nominal-shell clustering width.

    Per-shell patterns are fit WITHIN each shell -- b1000 predicted from b1000 only --
    so no shell's pattern borrows structure from another's residuals.
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
    if dwi_idx.size < 2:
        raise ValueError(f"need >= 2 volumes above b={lowb}, found {dwi_idx.size}")

    if mask is None:
        b0 = np.where(bvals <= lowb)[0]
        b0_mean = dwi[..., b0].mean(-1) if b0.size else dwi.mean(-1)
        mask = brain_mask(b0_mean)
    mask = np.asarray(mask, bool)
    if mask.shape != dwi.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != dwi spatial shape {dwi.shape[:3]}")
    n_mask = int(mask.sum())
    if n_mask <= dwi_idx.size:
        raise ValueError(
            f"mask holds {n_mask} voxels but P2S fits up to {dwi_idx.size - 1} predictors "
            f"per volume -- the regression is singular. Need V >> N; V/N here is "
            f"{n_mask / max(dwi_idx.size, 1):.1f}.")

    # ---- shell grouping -----------------------------------------------------------------
    nominal = np.round(bvals / shell_tol) * shell_tol
    shells = sorted(set(nominal[dwi_idx]))
    want_shells = len(shells) > 1 if per_shell == "auto" else bool(per_shell)
    if want_shells:
        col_sets = [np.where(nominal == b)[0] for b in shells]
        group = np.zeros(n_vols, int)
        for s, cols in enumerate(col_sets):
            group[cols] = s
        group[bvals <= lowb] = 0        # b0s ride along with the first shell's pattern
    else:
        col_sets = [dwi_idx]
        group = np.zeros(n_vols, int)

    # ---- P2S per shell: rho from a column subsample, sigma from every column -------------
    rhos = []
    sigma_shell = []
    sumsq = np.zeros(int(mask.sum()))
    ncols_used = 0
    for cols in col_sets:
        Yc = dwi[mask][:, cols].astype(np.float64)
        res, infl = p2s_ols(Yc)
        del Yc
        # sigma uses ALL of this shell's columns; each column's own inflation factor.
        shell_sigma = noise_map(res, infl, np.arange(len(cols)))
        sigma_shell.append(float(np.median(shell_sigma)))
        sumsq += shell_sigma ** 2 * len(cols)
        ncols_used += len(cols)
        keep = np.sort(rng.permutation(res.shape[1])[:ncol])
        sub = res[:, keep].astype(np.float32)
        del res
        reg, _ = stable_region(sub, mask)
        rhos.append(rho_region(sub, mask, reg, lag=lag, dzmax=dzmax))
        del sub

    sigma_vox = np.sqrt(sumsq / ncols_used)
    sigma_p2s = np.zeros(mask.shape, np.float32)
    sigma_p2s[mask] = sigma_vox
    d = np.zeros(mask.shape, np.float32)
    d[mask] = sigma_vox / np.median(sigma_vox)      # median, not mean -- see CLAUDE.md

    prov = dict(patch=tuple(int(v) for v in np.asarray(patch, int)), lag=lag, dzmax=dzmax,
                ncol=int(ncol), seed=int(seed), lowb=float(lowb),
                per_shell=bool(want_shells), shell_tol=float(shell_tol),
                shells=[float(b) for b in shells] if want_shells else None,
                n_mask=int(mask.sum()), n_vols=int(n_vols),
                bvals_sha=_bvals_hash(bvals))
    return NoiseModel(rho=np.stack(rhos), group=group, d=d, sigma_p2s=sigma_p2s,
                      sigma_shell=np.array(sigma_shell), mask=mask, lag=lag,
                      provenance=prov)


def _bvals_hash(bvals):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(bvals, np.float64).tobytes()).hexdigest()[:16]
