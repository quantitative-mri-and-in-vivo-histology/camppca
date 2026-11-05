"""The default patch: dwidenoise's rule, and who falls back to it."""

import numpy as np
import pytest

from camppca import NoiseModel, denoise_mppca, denoise_with_noise_model
from camppca.patch import default_patch, resolve_patch


@pytest.mark.parametrize("n_vols, side", [(1, 1), (2, 3), (27, 3), (28, 5), (125, 5),
                                          (126, 7), (199, 7), (297, 7), (343, 7), (344, 9)])
def test_the_smallest_odd_cube_with_at_least_as_many_voxels_as_volumes(n_vols, side):
    assert default_patch((200, 200, 200, n_vols)) == (side, side, side)


def test_a_side_is_clipped_to_the_largest_odd_size_that_fits():
    assert default_patch((14, 6, 3, 200)) == (7, 5, 3)
    assert default_patch((14, 12, 4, 200)) == (7, 7, 3)


def test_an_explicit_patch_wins():
    np.testing.assert_array_equal(resolve_patch((3, 3, 1), (20, 20, 20, 300)), (3, 3, 1))
    np.testing.assert_array_equal(resolve_patch(None, (20, 20, 20, 300)), (7, 7, 7))
    with pytest.raises(ValueError, match="odd"):
        resolve_patch((4, 5, 5), (20, 20, 20, 300))


def _noise(shape, n_vols):
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape + (n_vols,)).astype(np.float32) + 100


def test_mppca_defaults_to_the_rule():
    mask = np.zeros((10, 10, 8), bool)
    mask[4:6, 4:6, 3:5] = True
    for n_vols, patch in ((27, (3, 3, 3)), (28, (5, 5, 5))):
        got = denoise_mppca(_noise(mask.shape, n_vols), mask, workers=1)
        assert got.provenance["patch"] == patch


def test_denoising_with_a_model_defaults_to_the_model_patch():
    """The loop fitted rho to that patch's split, so the denoise uses it too. A model that
    names no patch in ``fitted_to`` gets the rule. Provenance is a record, never read."""
    mask = np.zeros((10, 10, 8), bool)
    mask[4:6, 4:6, 3:5] = True
    dwi = _noise(mask.shape, 130)                   # the rule would say 7 x 7 x 7
    rho = np.zeros((7, 7, 3))
    rho[3, 3, 0] = 1.0
    d = np.ones(mask.shape, np.float32)

    fitted = NoiseModel(rho=rho, d=d, mask=mask, lag=3, fitted_to={"patch": (3, 3, 3)})
    got = denoise_with_noise_model(dwi, fitted, workers=1)
    assert got.provenance["patch"] == (3, 3, 3)

    wide = np.zeros((13, 13, 7))
    wide[6, 6, 0] = 1.0
    bare = NoiseModel(rho=wide, d=d, mask=mask, lag=6, provenance={"patch": (3, 3, 3)})
    got = denoise_with_noise_model(dwi, bare, workers=1)
    assert got.provenance["patch"] == (7, 7, 7)


def test_a_patch_larger_than_the_image_is_refused_by_both_denoisers():
    """No 5-deep box fits in 3 slices. When the gather still wrapped, MP-PCA held slices
    twice and gave sigma_hat 1.3 against a true 5, with no error."""
    dwi = np.random.default_rng(0).standard_normal((16, 16, 3, 60)).astype(np.float32)
    with pytest.raises(ValueError, match="larger than the image"):
        denoise_mppca(dwi, patch=(5, 5, 5), workers=1)
    model = NoiseModel(rho=np.zeros((9, 9, 5)) + np.pad([[[1.0]]], ((4, 4), (4, 4), (0, 4))),
                       d=np.ones((16, 16, 3), np.float32), mask=np.ones((16, 16, 3), bool),
                       lag=4)
    with pytest.raises(ValueError, match="larger than the image"):
        denoise_with_noise_model(dwi, model, patch=(5, 5, 5), workers=1)
