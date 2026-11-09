"""The noise model object, the derived patch correlation C, and the estimator."""

import numpy as np
import pytest

from camppca import brain_mask, denoise_with_noise_model
from camppca.denoiser import _flat_mode_factor
from camppca.denoiser.camppca import denoise_whitened, whitened_level
from camppca.noise_model import (
    NoiseModel,
    _covered,
    _effective_d,
    _footprint_average,
    _level_centers,
    _white_rho,
    estimate_noise_model,
)
from camppca.patch import patch_offsets, table_index

from helpers import separable_rho


def _synthetic(rng, shape=(20, 20, 10), n_dirs=27, sigma=8.0, rank=4):
    X, Y, Z = shape
    bvals = np.concatenate([np.zeros(6), np.full(n_dirs, 1000.0), np.full(n_dirs, 2000.0)])
    n_vols = bvals.size
    signal = rng.standard_normal((X, Y, Z, rank)) @ rng.standard_normal((rank, n_vols))
    dwi = signal * 20 + 300 + rng.standard_normal((X, Y, Z, n_vols)) * sigma
    mask = np.zeros(shape, bool)
    mask[3:17, 3:17, 2:8] = True
    return dwi, bvals, mask, sigma


def _model_with_rho(rho_table, lag, shape=(8, 8, 4)):
    return NoiseModel(rho=rho_table, d=np.ones(shape, np.float32),
                      mask=np.ones(shape, bool), lag=lag)


def test_patch_correlation_is_a_valid_correlation_matrix():
    lag, patch = 4, (5, 5, 3)
    model = _model_with_rho(separable_rho(lag, lag, 0.6, 0.35), lag)
    C = model.patch_correlation(patch)

    assert C.shape == (75, 75)
    assert np.allclose(C.diagonal(), 1.0), "unit diagonal"
    assert np.allclose(C, C.T), "symmetric"
    assert np.linalg.eigvalsh(C).min() > -1e-9, "positive semi-definite"


def test_patch_correlation_reads_rho_at_the_right_lags():
    lag, patch = 4, (5, 5, 3)
    rho = separable_rho(lag, lag, 0.6, 0.35)
    C = _model_with_rho(rho, lag).patch_correlation(patch)

    offs = patch_offsets(patch)
    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, len(offs), size=(20, 2)):
        dx, dy, dz = offs[i] - offs[j]
        assert C[i, j] == pytest.approx(rho[table_index(dx, dy, dz, lag)], abs=1e-9)


def test_patch_correlation_refuses_a_patch_wider_than_the_measured_lag():
    model = _model_with_rho(separable_rho(4, 4, 0.6, 0.35), lag=4)
    model.patch_correlation((5, 5, 5))                    # needs lag 4 -- fits exactly
    with pytest.raises(ValueError, match=r"needs lag 6.*covers lag 4"):
        model.patch_correlation((7, 7, 7))


def test_patch_correlation_refuses_an_even_patch():
    model = _model_with_rho(separable_rho(4, 4, 0.6, 0.35), lag=4)
    with pytest.raises(ValueError, match="odd"):
        model.patch_correlation((4, 5, 5))


def test_patch_correlation_refuses_a_patch_deeper_than_the_measured_dz():
    model = _model_with_rho(separable_rho(4, 4, 0.6, 0.35)[:, :, :1], lag=4)
    model.patch_correlation((5, 5, 1))
    with pytest.raises(ValueError, match="through-slice lag 2.*covers 0"):
        model.patch_correlation((5, 5, 3))


def test_patch_correlation_refuses_a_table_that_is_not_a_valid_correlation():
    """Lag-1 correlation 0.6 alone is indefinite on 5 voxels: 1 - 1.2 cos(pi/6) < 0."""
    lag = 4
    rho = _white_rho((2 * lag + 1, 2 * lag + 1, 1))
    rho[lag - 1, lag, 0] = rho[lag + 1, lag, 0] = 0.6
    model = _model_with_rho(rho, lag)
    with pytest.raises(ValueError, match="smallest eigenvalue of C"):
        model.patch_correlation((5, 1, 1))


def test_estimate_serves_a_one_slice_patch_on_a_thin_volume():
    """A one-slice patch on a thin volume used to crash: dzmax followed the in-plane reach.

    Reported for (11, 11, 1) on 9 slices; (5, 5, 1) on 3 slices is the same failure, smaller.
    """
    dwi, bvals, _, _ = _synthetic(np.random.default_rng(5), shape=(40, 40, 3))
    mask = np.zeros(dwi.shape[:3], bool)
    mask[4:36, 4:36, :] = True
    model = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 1))

    assert model.rho.shape == (9, 9, 1)
    assert np.isfinite(model.rho).all()
    assert model.patch_correlation((5, 5, 1)).shape == (25, 25)


def test_estimate_refuses_a_patch_larger_than_the_volume():
    dwi, bvals, _, _ = _synthetic(np.random.default_rng(6), shape=(20, 20, 3))
    mask = np.zeros(dwi.shape[:3], bool)
    mask[3:17, 3:17, :] = True
    with pytest.raises(ValueError, match="larger than the image"):
        estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 5))


def test_estimate_covers_the_lags_the_patch_needs():
    dwi, bvals, mask, sigma = _synthetic(np.random.default_rng(0))
    model = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3))

    assert model.rho.shape == (9, 9, 3), "through-slice reach follows the patch depth"
    assert model.lag == 4
    assert model.provenance["n_vols"] == bvals.size


def test_d_is_the_standard_mppca_level_averaged_over_patch_footprints():
    """Each grid center's MP level, squared, spread over its patch and averaged per voxel,
    written out the long way."""
    dwi, bvals, mask, sigma = _synthetic(np.random.default_rng(1))
    patch = (5, 5, 3)
    model = estimate_noise_model(dwi, mask, bvals=bvals, patch=patch)
    centers = _level_centers(mask, patch)
    level = whitened_level(dwi, centers, None, patch, workers=1)

    half = np.array(patch) // 2
    sums = np.zeros(mask.shape)
    counts = np.zeros(mask.shape)
    for c in np.argwhere(centers):
        box = tuple(slice(max(0, a - h), a + h + 1) for a, h in zip(c, half, strict=True))
        sums[box] += level[tuple(c)] ** 2
        counts[box] += 1
    want = np.sqrt(sums[mask] / counts[mask])

    # Not exact: the estimate runs in the worker layout, and a near-tied MP cut can flip.
    np.testing.assert_allclose(model.d[mask], want, rtol=5e-3)
    assert np.median(model.d[mask]) == pytest.approx(sigma, rel=0.1)
    assert np.all(model.d[~mask] == 0)
    assert model.provenance["level_stride"] == 2
    assert model.provenance["level_rounds_run"] == 0, "no level rounds by default"


def test_the_level_centers_lie_on_the_grid_and_their_patches_cover_the_mask():
    mask = np.zeros((20, 20, 12), bool)
    mask[3:17, 4:15, 2:10] = True
    mask[1, 1, 1] = True                       # an island no grid patch reaches
    centers = _level_centers(mask, (3, 3, 3))
    x, y, z = np.indices(mask.shape)
    on = (x % 2 == 0) & (y % 2 == 0) & (z % 2 == 0)
    assert np.array_equal(centers & ~on, np.isin(np.arange(mask.size), [
        np.ravel_multi_index((1, 1, 1), mask.shape)]).reshape(mask.shape))
    assert not (centers & ~mask).any()
    assert _covered(centers, (3, 3, 3))[mask].all()


@pytest.mark.parametrize("level_rounds", [0, 1])
def test_estimate_is_deterministic(level_rounds):
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(2))
    a = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3), level_rounds=level_rounds)
    b = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3), level_rounds=level_rounds)
    assert np.array_equal(a.rho, b.rho)
    assert np.array_equal(a.d, b.d)
    assert a.provenance == b.provenance


def test_bvals_do_not_enter_the_estimate_when_the_mask_is_given():
    """They only pick the b0s the default mask is drawn from."""
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(3))
    with_bvals = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3))
    without = estimate_noise_model(dwi, mask, bvals=None, patch=(5, 5, 3))
    assert np.array_equal(with_bvals.rho, without.rho)
    assert np.array_equal(with_bvals.d, without.d)


def test_estimate_rejects_mismatched_bvals():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(4))
    with pytest.raises(ValueError, match="bvals has"):
        estimate_noise_model(dwi, mask, bvals=bvals[:-1], patch=(5, 5, 3))


def test_fewer_volumes_than_the_mp_fit_trusts_warns():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(0), n_dirs=17)   # 40 volumes
    with pytest.warns(RuntimeWarning, match="fewer than 50"):
        estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3))


def test_a_negative_level_round_count_is_refused():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(0))
    with pytest.raises(ValueError, match="level_rounds must be"):
        estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3), level_rounds=-1)


def test_the_auto_mask_is_the_whole_brain_not_the_eroded_interior():
    """The stored mask is also the set of voxels the engine WRITES.

    brain_mask erodes 3 voxels by default. Storing that eroded mask left a thick rim
    un-denoised -- 57% of brain voxels on this phantom.
    """
    rng = np.random.default_rng(0)
    shape, n_vols = (40, 40, 20), 50
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in (shape[2], shape[1], shape[0])),
                             indexing="ij")
    brain = ((((xx - 20) / 16.) ** 2 + ((yy - 20) / 16.) ** 2
              + ((zz - 10) / 8.) ** 2) < 1.0).transpose(2, 1, 0)
    bvals = np.concatenate([np.zeros(4), np.full(23, 1000.0), np.full(23, 2000.0)])
    signal = rng.standard_normal(shape + (3,)) @ rng.standard_normal((3, n_vols))
    dwi = np.zeros(shape + (n_vols,), np.float32)
    dwi[brain] = (signal * 20 + 400)[brain]
    dwi += (rng.standard_normal(shape + (n_vols,)) * 8).astype(np.float32)

    model = estimate_noise_model(dwi, None, bvals=bvals, patch=(5, 5, 3))
    covered = model.mask.sum() / brain.sum()
    assert covered > 0.95, f"the mask must cover the brain, got {covered:.0%}"


def test_a_non_bool_mask_does_not_become_an_integer_index():
    """`... & mask` with a uint8 mask makes `d[good]` a fancy INDEX, not a selection."""
    model = _model_with_rho(separable_rho(4, 4, 0.6, 0.35), lag=4)
    uint8 = np.ones((8, 8, 4), np.uint8)               # what get_fdata().astype() gives
    assert NoiseModel(rho=model.rho, d=model.d, mask=uint8, lag=4).mask.dtype == bool


def test_brain_mask_with_erode_zero_does_not_erode_to_nothing():
    """⚠ scipy erodes to CONVERGENCE at iterations=0, not zero times."""
    b0 = np.zeros((20, 20, 12))
    b0[4:16, 4:16, 3:9] = 100.0
    assert brain_mask(b0, erode=0).sum() > 0
    assert brain_mask(b0, erode=0).sum() > brain_mask(b0, erode=2).sum()


def test_brain_mask_keeps_brain_that_touches_the_image_faces():
    """The closing eroded against a zero border and cut two slices off each face."""
    b0 = np.zeros((20, 20, 8))
    b0[4:16, 4:16, :] = 100.0
    np.testing.assert_array_equal(brain_mask(b0, erode=0), b0 > 0)
    thin = np.zeros((24, 24, 3))
    thin[6:18, 6:18, :] = 100.0
    np.testing.assert_array_equal(brain_mask(thin, erode=0), thin > 0)


def test_the_footprint_average_is_over_squares_and_skips_centers_below_the_floor():
    level = np.array([3.0, 0.0, 4.0, 0.0, 1e-9]).reshape(5, 1, 1)
    centers = np.array([True, False, True, False, True]).reshape(5, 1, 1)
    mask = np.ones((5, 1, 1), bool)

    d = _footprint_average(level, centers, mask, (3, 1, 1))

    assert d[0, 0, 0] == pytest.approx(3.0)
    assert d[1, 0, 0] == pytest.approx(np.sqrt((9 + 16) / 2)), "averaging levels biases low"
    assert d[3, 0, 0] == pytest.approx(4.0), "the center below the floor adds nothing"
    assert d[4, 0, 0] == 0.0, "and stays zero itself, for _effective_d to fill"


def test_the_footprint_average_next_to_levels_near_zero_gives_no_nan():
    """The running sums of uniform_filter round below zero there. ideas01 sub-103 hit it."""
    level = np.random.default_rng(0).uniform(50, 150, (40, 40, 20))
    level[:, :, 15:] = 1e-7
    mask = np.ones(level.shape, bool)
    d = _footprint_average(level, mask, mask, (3, 3, 3))
    assert np.isfinite(d).all() and (d >= 0).all()


def test_levels_near_zero_do_not_pull_down_their_neighbours():
    """Averaged in, a zero slice took its neighbours to about 0.82 of their level."""
    level = np.full((12, 12, 12), 10.0)
    level[:, :, 9:] = 1e-7
    mask = np.ones(level.shape, bool)
    d = _footprint_average(level, mask, mask, (3, 3, 3))
    np.testing.assert_allclose(d[:, :, :9], 10.0, rtol=1e-6)
    assert not d[:, :, 9:].any(), "left zero for _effective_d to fill"


def test_levels_near_zero_take_their_neighbours_level():
    d = np.full((10, 10, 10), 5.0, np.float32)
    d[:, :, 8:] = 1e-7
    d[:, :, :2] = 9.0
    eff = _effective_d(d, np.ones(d.shape, bool))
    np.testing.assert_array_equal(eff[:, :, 8:], 5.0)
    np.testing.assert_array_equal(eff[:, :, :2], 9.0)


def test_zero_filled_slices_inside_the_mask_do_not_break_the_estimate():
    """Whole patches of zeros have an MP level of zero. The loop skips them, and the
    denoiser whitens them with their neighbours' level."""
    rng = np.random.default_rng(0)
    shape, n_vols, sigma = (24, 24, 20), 60, 10.0
    dwi = (rng.standard_normal(shape + (3,)) @ rng.standard_normal((3, n_vols)) * 30 + 300
           + sigma * rng.standard_normal(shape + (n_vols,))).astype(np.float32)
    dwi[:, :, 13:] = 0
    model = estimate_noise_model(dwi, mask=np.ones(shape, bool), patch=(5, 5, 5), workers=1)
    result = denoise_with_noise_model(dwi, model, workers=1)

    assert np.isfinite(result.denoised).all() and np.isfinite(result.sigma).all()
    assert abs(model.rho[model.lag + 1, model.lag, 0]) < 0.02, "the noise is white"
    assert np.median(result.sigma[:, :, 3:10]) == pytest.approx(sigma, rel=0.05)


@pytest.mark.parametrize("aggregate", ["center", "mean"])
def test_the_flat_mode_factor(aggregate):
    assert _flat_mode_factor(np.eye(27), aggregate) == pytest.approx(1.0, abs=1e-12)
    # Equal correlation a between every pair: P C P = (1 - a) P, so the factor is
    # sqrt(1 - a) at every position.
    a = 0.3
    C = np.full((27, 27), a) + (1 - a) * np.eye(27)
    assert _flat_mode_factor(C, aggregate) == pytest.approx(np.sqrt(1 - a), rel=1e-12)


def test_sigma_leaves_the_flat_mode_out():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(3))
    model = estimate_noise_model(dwi, mask, bvals=bvals, patch=(5, 5, 3))
    out = denoise_with_noise_model(dwi, model, mask_only=True)
    C = model.patch_correlation((5, 5, 3))
    _, raw, _ = denoise_whitened(dwi, mask, C, patch=(5, 5, 3), d_field=model.effective_d())
    expected = raw * model.effective_d() * _flat_mode_factor(C)
    np.testing.assert_allclose(out.sigma[mask], expected[mask], rtol=1e-6)
    assert _flat_mode_factor(C) < 1, "positive correlation: below the voxel's own level"


def _white_model(lag, dzmax, fitted=None):
    rho = _white_rho((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    return NoiseModel(rho=rho, d=np.ones((4, 4, 4)), mask=np.ones((4, 4, 4)), lag=lag,
                      fitted_to={} if fitted is None else dict(patch=fitted))


def test_a_patch_wider_than_the_fitted_one_on_any_axis_is_refused():
    """A (3, 7, 5) fit has lag 6 but measured only |dx| <= 2: (7, 3, 5) passed the lag
    check and read the white start as zero correlation."""
    model = _white_model(6, 4, fitted=(3, 7, 5))
    with pytest.raises(ValueError, match="not fitted at"):
        model.patch_correlation((7, 3, 5))
    assert model.patch_correlation((3, 5, 5)).shape == (75, 75)
    with pytest.raises(ValueError, match="needs lag"):
        _white_model(6, 4).patch_correlation((9, 9, 5))      # no fitted patch: the table


def test_a_table_that_does_not_fit_the_lag_or_is_not_a_correlation_is_refused():
    rho = _white_rho((9, 9, 3))
    with pytest.raises(ValueError, match="shaped"):
        NoiseModel(rho=rho, d=np.ones((4, 4, 4)), mask=np.ones((4, 4, 4)), lag=2)
    with pytest.raises(ValueError, match="1 at zero lag"):
        NoiseModel(rho=2 * rho, d=np.ones((4, 4, 4)), mask=np.ones((4, 4, 4)), lag=4)
