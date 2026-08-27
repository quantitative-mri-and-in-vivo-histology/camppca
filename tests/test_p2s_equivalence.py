"""Is our P2S really Patch2Self?

`p2s_ols` is asserted against the exact estimator dipy's `patch2self` specifies for
`model="ols"`: leave-one-volume-out `Ridge(alpha=1e-10)` regression across voxels, at
`patch_radius=0`.  We match it to ~1e-6.

We do NOT assert against dipy's `patch2self` OUTPUT, and the second test pins why.
dipy constructs `Ridge(copy_X=False, alpha=1e-10)` with `fit_intercept=True`; sklearn
centres X in place when `copy_X=False`, and dipy then calls `predict()` on that same,
now-centred array (`patch2self.py`, `_fit_denoising_model` -> `model.predict(cur_x.T)`).
Its predictions are therefore short by `mean(X) @ w`.  Verified against dipy 1.12.1 to
3.6e-11 -- that is a reproduction of the bug, not a tolerance.

Keeps dipy and scikit-learn as TEST-ONLY dev dependencies.
"""

import numpy as np
import pytest

from camppca.noise_model.p2s import p2s_ols

linear_model = pytest.importorskip("sklearn.linear_model")


def _data(seed=0, shape=(8, 8, 5), n_vols=12, rank=3, sigma=6.0):
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(shape + (rank,)) @ rng.standard_normal((rank, n_vols))
    return (signal * 20 + 300 + rng.standard_normal(shape + (n_vols,)) * sigma)


def _loo_ridge_residuals(D, **ridge_kw):
    """Leave-one-volume-out Ridge across voxels -- the estimator patch2self specifies."""
    out = np.empty_like(D)
    n_vols = D.shape[1]
    for j in range(n_vols):
        others = np.arange(n_vols) != j
        cur_x = np.ascontiguousarray(D[:, others])
        model = linear_model.Ridge(alpha=1e-10, **ridge_kw).fit(cur_x, D[:, j])
        out[:, j] = D[:, j] - model.predict(cur_x)
    return out


def test_p2s_ols_matches_the_patch2self_ols_estimator():
    D = _data().reshape(-1, 12)
    ours, infl = p2s_ols(D)
    reference = _loo_ridge_residuals(D)

    assert np.abs(ours - reference).max() < 1e-4 * np.abs(reference).std()
    assert np.corrcoef(ours.ravel(), reference.ravel())[0, 1] > 1 - 1e-12
    assert infl.shape == (12,)


def test_dipy_patch2self_output_carries_a_known_in_place_centring_bug():
    """Pins WHY we compare against the estimator rather than dipy's output."""
    patch2self = pytest.importorskip("dipy.denoise.patch2self").patch2self
    data = _data()
    n_vols = data.shape[-1]
    D = data.reshape(-1, n_vols)

    denoised = patch2self(data, np.full(n_vols, 1000.0), patch_radius=(0, 0, 0),
                          model="ols", shift_intensity=False, b0_denoising=False,
                          version=1)
    dipy_residuals = (data - denoised).reshape(-1, n_vols)

    # Reproduce dipy exactly by replaying the in-place mutation.
    replayed = _loo_ridge_residuals(D, copy_X=False)
    assert np.abs(dipy_residuals - replayed).max() < 1e-8

    # And confirm that is what separates it from the correct estimator.
    correct = _loo_ridge_residuals(D)
    assert np.abs(dipy_residuals - correct).max() > np.abs(correct).std()
