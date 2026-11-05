"""Complex data, checked against data built from a known signal and a known level per channel.

After phase stabilization the signal sits on the real axis and the noise is circular, the
same in both channels. Each test states a property the complex path must have.
"""

import warnings

import numpy as np
import pytest

from camppca import denoise_camppca, denoise_mppca, estimate_noise_model
from camppca.denoiser.camppca import denoise_whitened, whitened_level

SHAPE = (10, 10, 10)
N_VOLS = 60
PATCH = (5, 5, 5)


def interior_mask(shape=SHAPE, margin=2):
    """Voxels whose whole patch fits inside the volume, so nothing wraps."""
    mask = np.zeros(shape, bool)
    mask[margin:-margin, margin:-margin, margin:-margin] = True
    return mask


def circular_noise(rng, sigma, shape=SHAPE, n_vols=N_VOLS):
    """Complex noise with ``sigma`` in each channel."""
    return (sigma * (rng.standard_normal(shape + (n_vols,))
                     + 1j * rng.standard_normal(shape + (n_vols,)))).astype(np.complex64)


def low_rank_volume(rng, rank, shape=SHAPE, n_vols=N_VOLS, scale=100.0):
    """A real volume whose every patch matrix has rank at most ``rank``."""
    coeffs = rng.standard_normal(shape + (rank,))
    basis = rng.standard_normal((rank, n_vols))
    return (scale * coeffs @ basis).astype(np.float32)


@pytest.mark.parametrize("sigma_true", [2.0, 12.0])
def test_sigma_hat_is_the_noise_level_of_one_channel(sigma_true):
    dwi = circular_noise(np.random.default_rng(0), sigma_true)
    mask = interior_mask()

    result = denoise_mppca(dwi, mask, patch=PATCH, workers=1)

    assert result.denoised.dtype == np.complex64
    assert result.sigma_hat.dtype == np.float32
    assert abs(np.median(result.sigma_hat[mask]) - sigma_true) < 0.05 * sigma_true


def test_a_zero_imaginary_part_gives_the_real_result():
    rng = np.random.default_rng(1)
    dwi = low_rank_volume(rng, rank=3) + 20 * rng.standard_normal(SHAPE + (N_VOLS,))
    mask = interior_mask()

    real = denoise_mppca(dwi, mask, patch=PATCH, workers=1)
    cplx = denoise_mppca(dwi.astype(np.complex64), mask, patch=PATCH, workers=1)

    np.testing.assert_array_equal(cplx.rank, real.rank)
    np.testing.assert_allclose(cplx.sigma_hat * np.sqrt(2), real.sigma_hat, rtol=1e-4)
    scale = np.abs(real.denoised[mask]).max()
    assert np.abs(cplx.denoised[mask] - real.denoised[mask]).max() < 1e-4 * scale


def test_a_global_phase_changes_nothing_but_the_phase():
    rng = np.random.default_rng(2)
    dwi = low_rank_volume(rng, rank=3) + circular_noise(rng, 20.0)
    mask = interior_mask()
    turn = np.complex64(np.exp(0.7j))

    a = denoise_mppca(dwi, mask, patch=PATCH, workers=1)
    b = denoise_mppca(dwi * turn, mask, patch=PATCH, workers=1)

    np.testing.assert_array_equal(a.rank, b.rank)
    np.testing.assert_allclose(a.sigma_hat[mask], b.sigma_hat[mask], rtol=1e-4)
    scale = np.abs(a.denoised[mask]).max()
    assert np.abs(b.denoised[mask] - a.denoised[mask] * turn).max() < 1e-4 * scale


@pytest.mark.parametrize("aggregate", ["center", "mean"])
def test_a_real_signal_in_circular_noise_comes_back_closer_to_the_truth(aggregate):
    rng = np.random.default_rng(3)
    signal = low_rank_volume(rng, rank=3)
    dwi = signal + circular_noise(rng, 20.0)
    mask = interior_mask()

    result = denoise_mppca(dwi, mask, patch=PATCH, workers=1, aggregate=aggregate)

    before = np.abs(dwi[mask] - signal[mask]).mean()
    after = np.abs(result.denoised[mask] - signal[mask]).mean()
    assert after < 0.5 * before
    assert round(float(np.median(result.rank[mask]))) == 3   # "mean" averages the rank


@pytest.mark.parametrize("aggregate, shrink", [("center", "hard"), ("center", "lr"),
                                               ("mean", "lr")])
def test_an_identity_whitener_reproduces_standard_mppca(aggregate, shrink):
    rng = np.random.default_rng(4)
    dwi = low_rank_volume(rng, rank=3) + circular_noise(rng, 20.0)
    mask = interior_mask()

    standard = denoise_mppca(dwi, mask, patch=PATCH, workers=1, aggregate=aggregate,
                             shrink=shrink)
    denoised, sigma_hat, rank = denoise_whitened(dwi, mask, np.eye(125), patch=PATCH,
                                                 workers=1, aggregate=aggregate,
                                                 shrink=shrink)

    np.testing.assert_array_equal(rank, standard.rank)
    np.testing.assert_allclose(sigma_hat[mask], standard.sigma_hat[mask], rtol=1e-4)
    scale = np.abs(standard.denoised[mask]).max()
    assert np.abs(denoised[mask] - standard.denoised[mask]).max() < 1e-4 * scale


def test_camppca_denoises_complex_data_with_a_model_from_the_real_part():
    rng = np.random.default_rng(5)
    shape = (16, 16, 12)
    signal = low_rank_volume(rng, rank=3, shape=shape) + 500
    dwi = signal + circular_noise(rng, 20.0, shape=shape)
    mask = interior_mask(shape)

    result = denoise_camppca(dwi, mask=mask, patch=PATCH, workers=1)

    assert result.denoised.dtype == np.complex64
    assert result.model.provenance["data"] == "complex, estimated on the real part"
    assert abs(np.median(result.sigma_hat[mask]) - 20.0) < 0.1 * 20.0
    before = np.abs(dwi[mask] - signal[mask]).mean()
    after = np.abs(result.denoised[mask] - signal[mask]).mean()
    assert after < 0.5 * before


def test_the_level_pass_reads_one_channel_of_complex_data():
    dwi = circular_noise(np.random.default_rng(6), 7.0)
    mask = interior_mask()

    level = whitened_level(dwi, mask, None, PATCH, workers=1)

    assert abs(np.median(level[mask]) - 7.0) < 0.05 * 7.0


def test_complex_data_without_a_stabilized_phase_is_flagged():
    rng = np.random.default_rng(7)
    shape = (16, 16, 12)
    stable = low_rank_volume(rng, rank=3, shape=shape) + 500 + circular_noise(rng, 20.0,
                                                                              shape=shape)
    mask = interior_mask(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate_noise_model(stable, mask=mask, patch=PATCH, workers=1)

    # Unstabilized: a phase that changes from volume to volume, as motion gives.
    turned = (stable * np.exp(2j * np.pi * rng.random(N_VOLS))).astype(np.complex64)
    with pytest.warns(RuntimeWarning, match="phase is stabilized"):
        estimate_noise_model(turned, mask=mask, patch=PATCH, workers=1)
