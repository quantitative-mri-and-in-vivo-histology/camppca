"""The noise model object, the derived patch correlation C, and the estimator."""

import subprocess
import sys

import numpy as np
import pytest

from camppca.noise_model import NoiseModel, estimate_noise_model
from camppca.patch import patch_offsets


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
    return NoiseModel(rho=rho_table[None], group=np.zeros(4, int),
                      d=np.ones(shape, np.float32), sigma_p2s=np.ones(shape, np.float32),
                      sigma_shell=np.ones(1), mask=np.ones(shape, bool), lag=lag)


def _separable_rho(lag, ax=0.6, ay=0.35, az=0.0):
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, lag + 1))
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            for k in range(lag + 1):
                rho[i + lag, j + lag, k] = ax ** abs(i) * ay ** abs(j) * az ** k
    return rho


def test_patch_correlation_is_a_valid_correlation_matrix():
    lag, patch = 4, (5, 5, 3)
    model = _model_with_rho(_separable_rho(lag), lag)
    Cs = model.patch_correlation(patch)

    assert Cs.shape == (1, 75, 75)
    C = Cs[0]
    assert np.allclose(C.diagonal(), 1.0), "unit diagonal"
    assert np.allclose(C, C.T), "symmetric"
    assert np.linalg.eigvalsh(C).min() > -1e-9, "positive semi-definite"


def test_patch_correlation_reads_rho_at_the_right_lags():
    lag, patch = 4, (5, 5, 3)
    rho = _separable_rho(lag)
    C = _model_with_rho(rho, lag).patch_correlation(patch)[0]

    offs = patch_offsets(patch)
    rng = np.random.default_rng(0)
    for i, j in rng.integers(0, len(offs), size=(20, 2)):
        dx, dy, dz = offs[i] - offs[j]
        assert C[i, j] == pytest.approx(rho[dx + lag, dy + lag, abs(dz)], abs=1e-9)


def test_patch_correlation_refuses_a_patch_wider_than_the_measured_lag():
    model = _model_with_rho(_separable_rho(4), lag=4)
    model.patch_correlation((5, 5, 5))                    # needs lag 4 -- fits exactly
    with pytest.raises(ValueError, match=r"needs lag 6.*covers lag 4"):
        model.patch_correlation((7, 7, 7))


def test_patch_correlation_refuses_an_even_patch():
    model = _model_with_rho(_separable_rho(4), lag=4)
    with pytest.raises(ValueError, match="odd"):
        model.patch_correlation((4, 5, 5))


def test_estimate_groups_shells_and_fits_one_pattern_each():
    dwi, bvals, mask, sigma = _synthetic(np.random.default_rng(0))
    model = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))

    assert model.n_shells == 2, "b1000 and b2000 get their own pattern"
    assert model.rho.shape == (2, 9, 9, 5)
    assert model.group.shape == (bvals.size,)
    assert model.lag == 4
    assert model.provenance["per_shell"] is True
    assert model.provenance["shells"] == [1000.0, 2000.0]


def test_estimate_recovers_the_noise_level_and_a_unit_median_field():
    dwi, bvals, mask, sigma = _synthetic(np.random.default_rng(1))
    model = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))

    assert np.median(model.sigma_p2s[mask]) == pytest.approx(sigma, rel=0.1)
    assert np.median(model.d[mask]) == pytest.approx(1.0, abs=1e-5), "d is median-normalized"
    assert np.all(model.d[~mask] == 0)


def test_column_scale_is_per_shell_and_mean_one():
    dwi, bvals, mask, sigma = _synthetic(np.random.default_rng(5))
    model = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))

    assert model.sigma_shell.shape == (2,)
    assert np.allclose(model.sigma_shell, sigma, rtol=0.15)
    col = model.column_scale()
    assert col.shape == (bvals.size,)
    assert col.mean() == pytest.approx(1.0)
    # every volume in a shell shares that shell's level
    for s in range(model.n_shells):
        assert len(np.unique(col[model.group == s])) == 1


def test_estimate_is_deterministic():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(2))
    a = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))
    b = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))
    assert np.array_equal(a.rho, b.rho)
    assert np.array_equal(a.sigma_p2s, b.sigma_p2s)
    assert a.provenance == b.provenance


def test_estimate_without_bvals_treats_everything_as_one_shell():
    dwi, _, mask, _ = _synthetic(np.random.default_rng(3))
    model = estimate_noise_model(dwi, None, mask, patch=(5, 5, 3))
    assert model.n_shells == 1
    assert model.provenance["per_shell"] is False


def test_estimate_rejects_mismatched_bvals():
    dwi, bvals, mask, _ = _synthetic(np.random.default_rng(4))
    with pytest.raises(ValueError, match="bvals has"):
        estimate_noise_model(dwi, bvals[:-1], mask, patch=(5, 5, 3))


def test_the_noise_model_half_never_imports_jax():
    """The seam, mechanically: the estimator must stay importable without JAX."""
    code = ("import camppca, camppca.noise_model, camppca.patch, sys; "
            "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd="/workspace")


def test_a_shell_too_small_to_fit_is_refused_not_silently_nan():
    """A stray single-volume shell used to yield an all-NaN rho and a fully NaN image.

    p2s_ols on one column has no predictors, so every residual is the centred signal;
    stable_region then drops every voxel and rho_region divides 0/0. Only a RuntimeWarning
    was emitted, and the CLI wrote a NaN NIfTI and exited 0.
    """
    rng = np.random.default_rng(0)
    shape, n_dirs = (20, 20, 10), 22
    bvals = np.concatenate([np.zeros(4), [3000.0], np.full(n_dirs, 1000.0),
                            np.full(n_dirs, 2000.0)])
    n_vols = bvals.size
    signal = rng.standard_normal(shape + (3,)) @ rng.standard_normal((3, n_vols))
    dwi = signal * 20 + 300 + rng.standard_normal(shape + (n_vols,)) * 8
    mask = np.ones(shape, bool)

    with pytest.raises(ValueError, match=r"b=3000 has 1.*minimum is"):
        estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3))

    # Pooling across shells is the documented way out, and it works.
    model = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 3), per_shell=False)
    assert model.n_shells == 1
    assert np.isfinite(model.rho).all()


def test_the_auto_mask_is_the_whole_brain_not_the_eroded_interior():
    """The stored mask is also the set of voxels the engine WRITES.

    brain_mask erodes 3 voxels because rho_region needs a strict interior. Storing that
    eroded mask left a thick rim un-denoised -- 57% of brain voxels on this phantom.
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

    model = estimate_noise_model(dwi, bvals, None, patch=(5, 5, 3))
    covered = model.mask.sum() / brain.sum()
    assert covered > 0.95, f"the mask must cover the brain, got {covered:.0%}"
    # ...while estimation still uses the strict interior.
    assert model.provenance["erode"] == 3
    assert model.provenance["n_interior"] < model.provenance["n_mask"] / 2


def test_a_non_bool_mask_does_not_become_an_integer_index():
    """`... & mask` with a uint8 mask makes `d[good]` a fancy INDEX, not a selection."""
    model = _model_with_rho(_separable_rho(4), lag=4)
    model.mask = np.ones((8, 8, 4), np.uint8)          # what get_fdata().astype() gives
    assert NoiseModel(rho=model.rho, group=model.group, d=model.d,
                      sigma_p2s=model.sigma_p2s, sigma_shell=model.sigma_shell,
                      mask=np.ones((8, 8, 4), np.uint8), lag=4).mask.dtype == bool


def test_brain_mask_with_erode_zero_does_not_erode_to_nothing():
    """⚠ scipy erodes to CONVERGENCE at iterations=0, not zero times."""
    from camppca.noise_model.masks import brain_mask

    b0 = np.zeros((20, 20, 12))
    b0[4:16, 4:16, 3:9] = 100.0
    assert brain_mask(b0, erode=0).sum() > 0
    assert brain_mask(b0, erode=0).sum() > brain_mask(b0, erode=2).sum()
