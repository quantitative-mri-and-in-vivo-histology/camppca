"""Builders several test files share: correlated noise, separable correlations, volumes."""

import numpy as np

from camppca.patch import patch_offsets


def filter_rho(h, lag):
    """The normalized autocorrelation of filter ``h``, lags -lag to lag."""
    full = np.correlate(h, h, mode="full")
    mid = len(h) - 1
    return np.array([full[mid + k] if abs(k) < len(h) else 0.0
                     for k in range(-lag, lag + 1)]) / full[mid]


def colored_noise(rng, shape, n_vols, hx, hy):
    """Unit noise filtered by ``hx`` along x and ``hy`` along y: correlation
    filter_rho(hx) x filter_rho(hy), z left uncorrelated."""
    w = rng.standard_normal(shape + (n_vols,))
    out = np.apply_along_axis(lambda v: np.convolve(v, hx, "same"), 0, w)
    out = np.apply_along_axis(lambda v: np.convolve(v, hy, "same"), 1, out)
    return out / np.sqrt((hx ** 2).sum() * (hy ** 2).sum())


def separable_correlation(patch, ax, ay, az):
    """Unit-diagonal patch correlation from a separable AR(1) pattern, PSD by construction."""
    off = patch_offsets(patch)
    lags = [off[:, i][:, None] - off[:, i][None, :] for i in range(3)]
    return (ax ** np.abs(lags[0])) * (ay ** np.abs(lags[1])) * (az ** np.abs(lags[2]))


def separable_rho(lag, dzmax, ax, ay, az=0.0):
    """The same separable AR(1) pattern as a table, the half dz >= 0."""
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            for k in range(dzmax + 1):
                rho[i + lag, j + lag, k] = ax ** abs(i) * ay ** abs(j) * az ** k
    return rho


def interior_mask(shape, margin=2):
    """Voxels whose whole patch fits inside the volume, so no box moves inward."""
    mask = np.zeros(shape, bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    return mask


def low_rank_volume(rng, rank, shape, n_vols, scale=100.0):
    """A real volume whose every patch matrix has rank at most ``rank``."""
    coeffs = rng.standard_normal(shape + (rank,))
    basis = rng.standard_normal((rank, n_vols))
    return (scale * coeffs @ basis).astype(np.float32)
