"""Patch2Self (radius 0): the measurement half of the noise model.

Leave-one-volume-out OLS across voxels gives signal-free residuals; their FFT
autocorrelation gives ``rho``, the correlation lag table, and their per-voxel RMS
gives ``sigma_p2s``, the absolute noise level.

Radius 0 is mandatory, not a simplification: a radius>0 predictor regresses each voxel
on its spatial neighbors, which biases the very spatial correlation being estimated.

``rho_region`` takes ``lag`` and ``dzmax`` explicitly, with no defaults: both follow
from the patch shape (``LAG = max(patch) - 1``), and any fixed default wraps the lag
lookup around for a patch deeper than 3 slices -- silently, with no error.
"""

import numpy as np


def p2s_ols(Y, ridge=1e-8):
    """Leave-one-volume-out OLS across voxels. Y (V, N) centred per volume.

    Returns residuals (V, N) and the per-volume noise-inflation factor 1 + sum_k w_jk^2.
    """
    V, N = Y.shape
    mu = Y.mean(0)
    Yc = Y - mu
    G = Yc.T @ Yc
    G += ridge * np.trace(G) / N * np.eye(N)
    res = np.empty_like(Yc)
    infl = np.empty(N)
    idx = np.arange(N)
    for j in range(N):
        o = idx != j
        w = np.linalg.solve(G[np.ix_(o, o)], G[o, j])
        res[:, j] = Yc[:, j] - Yc[:, o] @ w
        infl[j] = 1.0 + float((w ** 2).sum())
    return res, infl


def rho_region(res, mask, reg, lag, dzmax):
    """rho(dx,dy,dz) inside `reg`, per-voxel standardised. res (V, N) on `mask` voxels."""
    X, Y, Z = mask.shape
    A = res[reg[mask]]
    A = A / (A.std(1, keepdims=True) + 1e-12)
    vol = np.zeros((X, Y, Z, A.shape[1]), np.float32)
    m3 = np.zeros(mask.shape, np.float32)
    sel = np.zeros(mask.shape, bool)
    sel[mask] = reg[mask]
    vol[sel] = A
    m3[sel] = 1.0
    Mf = np.fft.rfft2(m3, axes=(0, 1))
    num = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    den = np.zeros_like(num)
    F = np.fft.rfft2(vol, axes=(0, 1))
    for dz in range(dzmax + 1):
        a = F[:, :, : Z - dz] if dz else F
        b = F[:, :, dz:] if dz else F
        c = np.fft.irfft2((a * np.conj(b)).sum(3).sum(2), s=(X, Y), axes=(0, 1)).real
        ma = Mf[:, :, : Z - dz] if dz else Mf
        mb = Mf[:, :, dz:] if dz else Mf
        cn = np.fft.irfft2((ma * np.conj(mb)).sum(2), s=(X, Y), axes=(0, 1)).real
        for i, dx in enumerate(range(-lag, lag + 1)):
            for k, dy in enumerate(range(-lag, lag + 1)):
                num[i, k, dz] += c[dx % X, dy % Y]
                den[i, k, dz] += cn[dx % X, dy % Y]
    rho = num / np.maximum(den, 1e-9)
    return rho / rho[lag, lag, 0]


def noise_map(res, infl, dwi_cols):
    """Absolute per-voxel noise level sigma_p2s (V,) from the P2S residuals.

    ``infl[j] = 1 + sum_k w_jk^2`` is the OLS predictor's own noise, folded into the
    residual: Var(r_j) = sigma^2 * infl[j].  Dividing it out is what makes this map
    ABSOLUTE rather than merely relative -- and is exactly what dipy's patch2self cannot
    give you, because it does not expose the weights.

    ``dwi_cols`` selects the columns to average over (b0 residual levels are not
    representative of DWI noise).  Kept explicit: the two canonical upstream call sites
    disagree on the cut, so the policy belongs to the caller.
    """
    return np.sqrt(np.mean(res[:, dwi_cols] ** 2 / infl[None, dwi_cols], axis=1))
