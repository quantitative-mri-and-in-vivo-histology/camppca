"""End to end on synthetic data with a KNOWN noise covariance.

Injects spatially correlated noise with an analytically known rho and sigma, then checks
the whole chain: the estimator recovers both, whitening collapses the patch off-diagonals
toward the i.i.d. floor, and the MP fit lands on the true level -- where standard,
white-noise MPPCA does not.
"""

import warnings

import numpy as np
import pytest

from camppca import (
    denoise_camppca,
    denoise_mppca,
    denoise_with_noise_model,
    estimate_noise_model,
)
from camppca.patch import patch_offsets

from helpers import colored_noise, filter_rho

SIGMA_TRUE = 10.0
HX = np.array([0.5, 1.0, 0.5])          # in-plane correlation along x
HY = np.array([0.25, 1.0, 0.25])        # and along y; z left uncorrelated


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    shape, n_vols, rank = (40, 40, 12), 60, 4
    bvals = np.concatenate([np.zeros(6), np.full(n_vols - 6, 1000.0)])
    signal = rng.standard_normal(shape + (rank,)) @ rng.standard_normal((rank, n_vols))
    dwi = signal * 25 + 400 + colored_noise(rng, shape, n_vols, HX, HY) * SIGMA_TRUE
    mask = np.zeros(shape, bool)
    mask[4:36, 4:36, 2:10] = True
    return dwi, bvals, mask


@pytest.fixture(scope="module")
def model(dataset):
    dwi, bvals, mask = dataset
    return estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3))


def test_estimator_recovers_the_injected_correlation(model):
    lag = model.lag
    rho = model.rho
    want_x, want_y = filter_rho(HX, lag), filter_rho(HY, lag)

    assert rho[lag, lag, 0] == pytest.approx(1.0)
    for k in (1, 2):
        assert rho[lag + k, lag, 0] == pytest.approx(want_x[lag + k], abs=0.03)
        assert rho[lag, lag + k, 0] == pytest.approx(want_y[lag + k], abs=0.03)
    assert abs(rho[lag, lag, 1]) < 0.05, "z was never correlated"


def test_whitening_collapses_the_patch_off_diagonals(model, dataset):
    """The point of the whole method: after C^(-1/2) the patch looks i.i.d."""
    from camppca.denoiser.camppca import sqrtm_psd

    dwi, _, mask = dataset
    patch = (5, 5, 3)
    C = model.patch_correlation(patch)
    W = sqrtm_psd(C, -0.5)

    rng = np.random.default_rng(1)
    n_cols = 200
    noise = colored_noise(rng, dwi.shape[:3], n_cols, HX, HY) * SIGMA_TRUE
    offs = patch_offsets(patch)
    kk = np.asarray(patch) // 2
    centers = np.argwhere(mask)[rng.permutation(int(mask.sum()))[:60]]
    stack = np.stack([np.stack([noise[x - kk[0] + dx, y - kk[1] + dy, z - kk[2] + dz, :]
                                for dx, dy, dz in offs]) for x, y, z in centers])

    def patch_corr(patches):
        """Empirical n_vox x n_vox correlation, pooled over patches and columns."""
        flat = patches.transpose(1, 0, 2).reshape(patches.shape[1], -1)
        return np.corrcoef(flat)

    n_vox = stack.shape[1]
    off = ~np.eye(n_vox, dtype=bool)
    before = patch_corr(stack)
    after = patch_corr(np.einsum("pq,bqn->bpn", W, stack))
    floor = 1.0 / np.sqrt(stack.shape[0] * n_cols)      # i.i.d. sampling floor

    # Adjacent-in-x pairs carry the injected correlation; they are the sharp probe.
    adjacent = (np.abs(offs[:, 0][:, None] - offs[:, 0][None, :]) == 1) & \
               (offs[:, 1][:, None] == offs[:, 1][None, :]) & \
               (offs[:, 2][:, None] == offs[:, 2][None, :])
    want_x1 = filter_rho(HX, 1)[2]

    assert np.abs(before[adjacent]).mean() == pytest.approx(want_x1, abs=0.05)
    collapsed = np.abs(after[adjacent]).mean()
    assert collapsed < 0.10, f"adjacent correlation must collapse, got {collapsed}"
    assert collapsed < np.abs(before[adjacent]).mean() / 8
    assert np.abs(after[off]).max() < np.abs(before[off]).max() / 5
    assert np.abs(after[off]).mean() < 20 * floor


def test_camppca_lands_on_the_true_level_where_white_mppca_does_not(dataset):
    """The paper's core claim: color makes standard MPPCA under-estimate sigma."""
    dwi, bvals, mask = dataset
    patch = (5, 5, 3)

    ours = denoise_camppca(dwi, mask, bvals=bvals, patch=patch).sigma
    white = denoise_mppca(dwi, mask, patch=patch).sigma

    ours_med, white_med = np.median(ours[mask]), np.median(white[mask])
    # Measured: camppca ~10.0, white MPPCA ~5.3 against a true 10.0.
    assert ours_med == pytest.approx(SIGMA_TRUE, rel=0.15)
    assert white_med < 0.7 * SIGMA_TRUE, "white MPPCA under-estimates on colored noise"
    assert abs(ours_med - SIGMA_TRUE) < abs(white_med - SIGMA_TRUE) / 3


def test_denoise_reduces_variance_and_keeps_shape(dataset):
    dwi, bvals, mask = dataset
    result = denoise_camppca(dwi, mask, bvals=bvals, patch=(5, 5, 3))

    assert result.denoised.shape == dwi.shape
    assert result.sigma.shape == dwi.shape[:3]
    assert (dwi - result.denoised)[mask].std() < SIGMA_TRUE * 1.2


def test_every_voxel_is_denoised_unless_mask_only(dataset, model):
    """The mask says where the model holds. A brain voxel the mask missed is still denoised."""
    dwi, bvals, mask = dataset
    everywhere = denoise_with_noise_model(dwi, model)
    only = denoise_with_noise_model(dwi, model, mask_only=True)

    assert (everywhere.sigma > 0).all() and (everywhere.rank > 0).all()
    assert (dwi - everywhere.denoised)[~mask].std() > 0.5 * SIGMA_TRUE
    passthrough = dwi.astype(only.denoised.dtype)        # engine works in float32
    assert np.all(only.denoised[~mask] == passthrough[~mask]), "outside the mask is untouched"
    assert not only.sigma[~mask].any()
    np.testing.assert_array_equal(only.denoised[mask], everywhere.denoised[mask])
    assert everywhere.provenance["mask_only"] is False
    assert only.provenance["mask_only"] is True


def test_the_scale_of_d_cancels(dataset, model):
    """Rows are divided by d and the center multiplied back, so a constant drops out.

    A power of two scales every float exactly, so x0.25 must be bit-identical. A generic
    factor may flip a near-tied MP cut by float32 rounding, in a small share of voxels.
    """
    from dataclasses import replace

    dwi, _, mask = dataset
    ref = denoise_with_noise_model(dwi, model, patch=(5, 5, 3))
    quarter = denoise_with_noise_model(dwi, replace(model, d=model.d * 0.25), patch=(5, 5, 3))
    odd = denoise_with_noise_model(dwi, replace(model, d=model.d * 7.3), patch=(5, 5, 3))

    for name in ("denoised", "sigma", "rank"):
        assert np.array_equal(getattr(quarter, name), getattr(ref, name)), name
    assert (odd.rank[mask] != ref.rank[mask]).mean() < 0.005
    same = odd.rank == ref.rank
    # float32 eigenvalues next to the baseline component: measured up to 6e-4 relative
    np.testing.assert_allclose(odd.sigma[same & mask], ref.sigma[same & mask],
                               rtol=2e-3)


def test_denoise_refuses_a_patch_the_model_cannot_serve(dataset, model):
    dwi, _, _ = dataset
    with pytest.raises(ValueError, match="not fitted at"):
        denoise_with_noise_model(dwi, model, patch=(9, 9, 9))


def test_denoise_refuses_a_mismatched_spatial_shape(dataset, model):
    dwi, _, _ = dataset
    with pytest.raises(ValueError, match="does not match the mask"):
        denoise_with_noise_model(dwi[:-1], model, patch=(5, 5, 3))


def test_a_bad_algorithm_is_refused_before_any_work(dataset, model):
    dwi, bvals, mask = dataset
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # no fit-mismatch warning first
        with pytest.raises(ValueError, match="algorithm must be one of"):
            denoise_with_noise_model(dwi, model, algorithm="exp2")
    with pytest.raises(ValueError, match="algorithm must be one of"):
        denoise_mppca(dwi, mask, algorithm="exp2")
    with pytest.raises(ValueError, match="algorithm must be one of"):
        estimate_noise_model(dwi, mask, bvals=bvals, algorithm="exp2")


def test_float64_input_gives_the_float32_model(dataset, model):
    """nibabel's get_fdata() gives float64. The estimator converts it once."""
    dwi, bvals, mask = dataset
    assert dwi.dtype == np.float64
    narrow = estimate_noise_model(dwi.astype(np.float32), mask, bvals=bvals, patch=(5, 5, 3))
    np.testing.assert_array_equal(narrow.rho, model.rho)
    np.testing.assert_array_equal(narrow.d, model.d)


def test_the_fit_mismatch_warning_names_only_recorded_settings(dataset, model):
    """A model may leave fitted_to keys out. One with "algorithm" but no "noise_field"
    warned about noise_field=None; one with only "noise_field" was never checked."""
    import dataclasses

    dwi, _, mask = dataset
    partial = dataclasses.replace(model, fitted_to=dict(patch=(5, 5, 3),
                                                        algorithm="cordero-grande"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        denoise_with_noise_model(dwi, partial, mask_only=True, workers=1)
    only_field = dataclasses.replace(model, fitted_to=dict(patch=(5, 5, 3), noise_field=False))
    with pytest.warns(RuntimeWarning, match="noise_field=False") as caught:
        denoise_with_noise_model(dwi, only_field, mask_only=True, workers=1)
    assert "algorithm" not in str(caught[0].message)
