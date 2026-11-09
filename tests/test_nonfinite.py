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


def test_a_value_that_would_overflow_the_gram_stops_the_run(data):
    """One entry of 3e19 made a whole patch's neighbours NaN, with no error."""
    clean = data[1]
    huge = clean.copy()
    huge[8, 8, 6, 10] = 3e19
    with pytest.raises(ValueError, match="float32 Gram"):
        denoise_mppca(huge, patch=PATCH, workers=1)
    with pytest.raises(ValueError, match="float32 Gram"):
        estimate_noise_model(huge, patch=PATCH, workers=1)
    denoise_mppca(clean * 1e6, patch=PATCH, workers=1)  # large but real-looking data pass


def test_a_failed_eigendecomposition_raises():
    from camppca.denoiser.spectrum import eigh_descending, tridiagonal_eigvalsh

    gram = np.full((2, 6, 6), np.nan, np.float32)
    with pytest.raises(ValueError, match="not finite"):
        tridiagonal_eigvalsh(gram)
    with pytest.raises(ValueError):            # scipy's own LinAlgError, or the backstop
        eigh_descending(gram)


def test_the_warning_points_at_the_callers_line(data):
    """Not at the BLAS decorator every public function runs under, nor at the helper."""
    dirty = data[0]
    with pytest.warns(RuntimeWarning, match="NaN or inf") as caught:
        denoise_mppca(dirty, patch=PATCH)
    assert {w.filename for w in caught} == {__file__}
    with pytest.warns(RuntimeWarning, match="NaN or inf") as caught:
        estimate_noise_model(dirty, patch=PATCH)
    assert {w.filename for w in caught if "NaN" in str(w.message)} == {__file__}
