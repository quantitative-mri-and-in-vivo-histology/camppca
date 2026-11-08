"""NaN and inf in the input: they enter the fit as zero and stay in the output."""

import numpy as np
import pytest

from camppca import denoise_mppca, denoise_with_noise_model, estimate_noise_model

SHAPE, N_VOLS, PATCH = (16, 16, 12), 60, (5, 5, 5)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    clean = (rng.standard_normal(SHAPE + (3,)) @ rng.standard_normal((3, N_VOLS)) * 30
             + 300 + 10 * rng.standard_normal(SHAPE + (N_VOLS,))).astype(np.float32)
    clean[:3] = 0                                  # background
    dirty = clean.copy()
    dirty[:3] = np.nan                             # NaN background, as after resampling
    dirty[8, 8, 6, 5] = np.inf                     # one bad entry inside the brain
    zeroed = np.where(np.isfinite(dirty), dirty, 0).astype(np.float32)
    return dirty, zeroed


def _same_except_nonfinite(got, want_finite, dirty):
    bad = ~np.isfinite(dirty)
    np.testing.assert_array_equal(got[~bad], want_finite[~bad])
    np.testing.assert_array_equal(got[bad], dirty[bad])


@pytest.mark.parametrize("aggregate", ["center", "mean"])
def test_mppca_fits_nonfinite_as_zero_and_puts_it_back(data, aggregate):
    dirty, zeroed = data
    with pytest.warns(RuntimeWarning, match="NaN or inf"):
        got = denoise_mppca(dirty, patch=PATCH, workers=1, aggregate=aggregate)
    want = denoise_mppca(zeroed, patch=PATCH, workers=1, aggregate=aggregate)

    _same_except_nonfinite(got.denoised, want.denoised, dirty)
    np.testing.assert_array_equal(got.sigma, want.sigma)


@pytest.mark.parametrize("aggregate", ["center", "mean"])
def test_camppca_fits_nonfinite_as_zero_and_puts_it_back(data, aggregate):
    dirty, zeroed = data
    mask = np.zeros(SHAPE, bool)
    mask[4:14, 3:13, 2:10] = True
    with pytest.warns(RuntimeWarning, match="NaN or inf"):
        model = estimate_noise_model(dirty, mask=mask, patch=PATCH, workers=1)
        got = denoise_with_noise_model(dirty, model, workers=1, aggregate=aggregate)
    clean_model = estimate_noise_model(zeroed, mask=mask, patch=PATCH, workers=1)
    want = denoise_with_noise_model(zeroed, clean_model, workers=1, aggregate=aggregate)

    np.testing.assert_array_equal(model.rho, clean_model.rho)
    np.testing.assert_array_equal(model.d, clean_model.d)
    _same_except_nonfinite(got.denoised, want.denoised, dirty)
    np.testing.assert_array_equal(got.sigma, want.sigma)


def test_finite_input_is_not_copied_or_warned_about(data):
    from camppca.denoiser.gather import zero_nonfinite
    _, zeroed = data
    out, bad, values = zero_nonfinite(zeroed)
    assert out is zeroed and bad is None and values is None
