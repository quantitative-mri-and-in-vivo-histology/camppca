"""End to end on synthetic data with a KNOWN noise covariance.

Injects spatially correlated noise with an analytically known rho and sigma, then checks
the whole chain: the estimator recovers both, whitening collapses the patch off-diagonals
toward the i.i.d. floor, and the MP fit lands on the true level -- where standard,
white-noise MP-PCA does not.

⚠ The mask here is deliberately large. P2S fits N-1 predictors across the mask voxels, so
at small V the in-sample overfitting deflates BOTH rho and sigma. Measured at N = 60:

    V = 576    p/V = 0.10    rho_x1 0.589 / 0.665    sigma 7.95 / 10
    V = 2904   p/V = 0.020   rho_x1 0.649 / 0.665    sigma 9.53 / 10
    V = 11560  p/V = 0.005   rho_x1 0.657 / 0.665    sigma 9.82 / 10

Real data sits near p/V = 1e-4, so this never bites in practice -- but a small synthetic
test will read as a broken estimator if you let it.
"""

import numpy as np
import pytest

from camppca import denoise_camppca, denoise_mppca, estimate_noise_model
from camppca.denoiser import denoise_with_noise_model
from camppca.patch import patch_offsets

SIGMA_TRUE = 10.0
HX = np.array([0.5, 1.0, 0.5])          # in-plane correlation along x
HY = np.array([0.25, 1.0, 0.25])        # and along y; z left uncorrelated


def _filter_rho(h, lag):
    full = np.correlate(h, h, mode="full")
    mid = len(h) - 1
    return np.array([full[mid + k] if abs(k) < len(h) else 0.0
                     for k in range(-lag, lag + 1)]) / full[mid]


def _colored_noise(rng, shape, n_vols, sigma):
    """Noise with correlation rho = acf(HX) x acf(HY), and marginal std exactly `sigma`."""
    w = rng.standard_normal(shape + (n_vols,))
    out = np.apply_along_axis(lambda v: np.convolve(v, HX, "same"), 0, w)
    out = np.apply_along_axis(lambda v: np.convolve(v, HY, "same"), 1, out)
    gain = np.sqrt((HX ** 2).sum() * (HY ** 2).sum())
    return out / gain * sigma


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    shape, n_vols, rank = (40, 40, 12), 60, 4
    bvals = np.concatenate([np.zeros(6), np.full(n_vols - 6, 1000.0)])
    signal = rng.standard_normal(shape + (rank,)) @ rng.standard_normal((rank, n_vols))
    dwi = signal * 25 + 400 + _colored_noise(rng, shape, n_vols, SIGMA_TRUE)
    mask = np.zeros(shape, bool)
    mask[4:36, 4:36, 2:10] = True          # V = 8192, p/V = 0.007
    return dwi, bvals, mask


@pytest.fixture(scope="module")
def model(dataset):
    dwi, bvals, mask = dataset
    return estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))


def test_estimator_recovers_the_injected_correlation(model):
    lag = model.lag
    rho = model.rho
    want_x, want_y = _filter_rho(HX, lag), _filter_rho(HY, lag)

    assert rho[lag, lag, 0] == pytest.approx(1.0)
    for k in (1, 2):
        assert rho[lag + k, lag, 0] == pytest.approx(want_x[lag + k], abs=0.03)
        assert rho[lag, lag + k, 0] == pytest.approx(want_y[lag + k], abs=0.03)
    assert abs(rho[lag, lag, 1]) < 0.05, "z was never correlated"


def test_estimator_recovers_the_injected_noise_level(model, dataset):
    _, _, mask = dataset
    assert np.median(model.sigma_p2s[mask]) == pytest.approx(SIGMA_TRUE, rel=0.05)


def test_whitening_collapses_the_patch_off_diagonals(model, dataset):
    """The point of the whole method: after C^(-1/2) the patch looks i.i.d."""
    from camppca.denoiser.camppca import sqrtm_psd

    dwi, _, mask = dataset
    patch = (5, 5, 3)
    C = model.patch_correlation(patch)
    W = sqrtm_psd(C, -0.5)

    rng = np.random.default_rng(1)
    n_cols = 200
    noise = _colored_noise(rng, dwi.shape[:3], n_cols, SIGMA_TRUE)
    offs = patch_offsets(patch)
    kk = np.asarray(patch) // 2
    centres = np.argwhere(mask)[rng.permutation(int(mask.sum()))[:60]]
    stack = np.stack([np.stack([noise[x - kk[0] + dx, y - kk[1] + dy, z - kk[2] + dz, :]
                                for dx, dy, dz in offs]) for x, y, z in centres])

    def patch_corr(patches):
        """Empirical P x P correlation, pooled over patches and columns."""
        flat = patches.transpose(1, 0, 2).reshape(patches.shape[1], -1)
        return np.corrcoef(flat)

    P = stack.shape[1]
    off = ~np.eye(P, dtype=bool)
    before = patch_corr(stack)
    after = patch_corr(np.einsum("pq,bqn->bpn", W, stack))
    floor = 1.0 / np.sqrt(stack.shape[0] * n_cols)      # i.i.d. sampling floor

    # Adjacent-in-x pairs carry the injected correlation; they are the sharp probe.
    adjacent = (np.abs(offs[:, 0][:, None] - offs[:, 0][None, :]) == 1) & \
               (offs[:, 1][:, None] == offs[:, 1][None, :]) & \
               (offs[:, 2][:, None] == offs[:, 2][None, :])
    want_x1 = _filter_rho(HX, 1)[2]

    assert np.abs(before[adjacent]).mean() == pytest.approx(want_x1, abs=0.05)
    collapsed = np.abs(after[adjacent]).mean()
    assert collapsed < 0.10, f"adjacent correlation must collapse, got {collapsed}"
    assert collapsed < np.abs(before[adjacent]).mean() / 8
    assert np.abs(after[off]).max() < np.abs(before[off]).max() / 5
    assert np.abs(after[off]).mean() < 20 * floor


def test_camppca_lands_on_the_true_level_where_white_mppca_does_not(dataset):
    """The paper's core claim: colour makes standard MP-PCA under-estimate sigma."""
    dwi, bvals, mask = dataset
    patch = (5, 5, 3)

    ours = denoise_camppca(dwi, bvals, mask, patch=patch).sigma_hat
    white = denoise_mppca(dwi, mask, patch=patch).sigma_hat

    ours_med, white_med = np.median(ours[mask]), np.median(white[mask])
    # Measured: camppca ~10.0, white MP-PCA ~5.3 against a true 10.0.
    assert ours_med == pytest.approx(SIGMA_TRUE, rel=0.15)
    assert white_med < 0.7 * SIGMA_TRUE, "white MP-PCA under-estimates on coloured noise"
    assert abs(ours_med - SIGMA_TRUE) < abs(white_med - SIGMA_TRUE) / 3


def test_denoise_reduces_variance_and_keeps_shape(dataset):
    dwi, bvals, mask = dataset
    result = denoise_camppca(dwi, bvals, mask, patch=(5, 5, 3))

    assert result.denoised.shape == dwi.shape
    assert result.sigma_hat.shape == dwi.shape[:3]
    passthrough = dwi.astype(result.denoised.dtype)      # engine works in float32
    assert np.all(result.denoised[~mask] == passthrough[~mask]), "outside the mask is untouched"
    assert (dwi - result.denoised)[mask].std() < SIGMA_TRUE * 1.2


def test_sigma_units_switch(dataset, model):
    dwi, _, mask = dataset
    data = denoise_with_noise_model(dwi, model, patch=(5, 5, 3), sigma_units="data")
    norm = denoise_with_noise_model(dwi, model, patch=(5, 5, 3), sigma_units="d")

    assert not np.allclose(data.sigma_hat[mask], norm.sigma_hat[mask])
    assert np.array_equal(data.denoised, norm.denoised), "only sigma_hat's units change"
    with pytest.raises(ValueError, match="sigma_units"):
        denoise_with_noise_model(dwi, model, patch=(5, 5, 3), sigma_units="nope")


def test_denoise_refuses_a_patch_the_model_cannot_serve(dataset, model):
    dwi, _, _ = dataset
    with pytest.raises(ValueError, match="needs lag"):
        denoise_with_noise_model(dwi, model, patch=(9, 9, 9))


def test_denoise_refuses_a_mismatched_spatial_shape(dataset, model):
    dwi, _, _ = dataset
    with pytest.raises(ValueError, match="model mask"):
        denoise_with_noise_model(dwi[:-1], model, patch=(5, 5, 3))
