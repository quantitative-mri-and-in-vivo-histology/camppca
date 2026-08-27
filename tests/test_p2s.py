"""Patch2Self residuals, the rho lag table, and the absolute noise map."""

import numpy as np
import pytest

from camppca.noise_model.masks import stable_region
from camppca.noise_model.p2s import noise_map, p2s_ols, rho_region


def _sep_filter_rho(h, lag):
    """Analytic normalized autocorrelation of a 1-D filter, out to `lag`."""
    full = np.correlate(h, h, mode="full")
    mid = len(h) - 1
    return np.array([full[mid + k] if abs(k) < len(h) else 0.0
                     for k in range(-lag, lag + 1)]) / full[mid]


def _correlated_noise(rng, shape, hx, hy, n_cols):
    """White noise filtered separably in x and y -> known spatial correlation."""
    w = rng.standard_normal(shape + (n_cols,))
    out = np.apply_along_axis(lambda v: np.convolve(v, hx, "same"), 0, w)
    return np.apply_along_axis(lambda v: np.convolve(v, hy, "same"), 1, out)


def test_p2s_ols_shapes_and_inflation():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((400, 10)) * 5 + 100
    res, infl = p2s_ols(Y)
    assert res.shape == Y.shape
    assert infl.shape == (10,)
    assert np.all(infl >= 1.0), "inflation factor is 1 + sum(w^2), never below 1"


def test_noise_map_recovers_sigma_and_needs_the_inflation_correction():
    rng = np.random.default_rng(1)
    n_vox, n_vols, sigma_true = 4000, 40, 7.0
    signal = rng.standard_normal((n_vox, 3)) @ rng.standard_normal((3, n_vols)) * 30 + 400
    Y = signal + rng.standard_normal((n_vox, n_vols)) * sigma_true

    res, infl = p2s_ols(Y)
    corrected = np.median(noise_map(res, infl, np.arange(n_vols)))
    assert corrected == pytest.approx(sigma_true, rel=0.05)

    # Without dividing by infl the residual carries the predictor's own noise too,
    # so the level comes out HIGH. This is what makes the map absolute.
    raw = np.median(np.sqrt(np.mean(res ** 2, axis=1)))
    assert raw > corrected


def test_rho_region_recovers_an_injected_correlation():
    rng = np.random.default_rng(2)
    shape, lag = (32, 32, 6), 3
    hx, hy = np.array([0.4, 1.0, 0.4]), np.array([0.2, 1.0, 0.2])
    vol = _correlated_noise(rng, shape, hx, hy, 60)

    mask = np.zeros(shape, bool)
    mask[4:28, 4:28, 1:5] = True
    res = vol[mask]
    rho = rho_region(res, mask, mask, lag=lag, dzmax=lag)

    assert rho[lag, lag, 0] == pytest.approx(1.0), "rho is unit at zero lag by construction"
    want_x, want_y = _sep_filter_rho(hx, lag), _sep_filter_rho(hy, lag)
    for k in range(-lag, lag + 1):
        assert rho[k + lag, lag, 0] == pytest.approx(want_x[k + lag], abs=0.03)
        assert rho[lag, k + lag, 0] == pytest.approx(want_y[k + lag], abs=0.03)
    # z was never filtered, so through-plane correlation must vanish.
    assert abs(rho[lag, lag, 1]) < 0.05


def test_stable_region_drops_constant_voxels_that_would_swamp_rho():
    rng = np.random.default_rng(3)
    shape, lag = (24, 24, 6), 2
    vol = _correlated_noise(rng, shape, np.array([0.5, 1.0, 0.5]), np.array([1.0]), 40)
    mask = np.zeros(shape, bool)
    mask[3:21, 3:21, 1:5] = True

    res = vol[mask]
    res[7] = 1e4                      # constant but non-zero: std ~ 0, value huge
    reg, dropped = stable_region(res, mask)
    assert dropped == 1

    bad = rho_region(res, mask, mask, lag=lag, dzmax=1)
    good = rho_region(res, mask, reg, lag=lag, dzmax=1)
    # The rogue voxel flattens every non-zero lag toward zero; dropping it restores them.
    assert abs(bad[lag + 1, lag, 0]) < abs(good[lag + 1, lag, 0])
    assert good[lag + 1, lag, 0] > 0.2
