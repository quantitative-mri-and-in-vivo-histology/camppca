"""Standard MPPCA, checked against data we build from a known signal and a known sigma.

Nothing here compares against another denoiser. Each test states a property the method
must have and constructs the data that makes it checkable: pure noise of a known level,
a signal of a known rank, or a single patch whose answer plain numpy can work out.
"""

import numpy as np
import pytest

from camppca import denoise_mppca
from camppca.denoiser.camppca import denoise_whitened

from helpers import interior_mask, low_rank_volume

SHAPE = (10, 10, 10)
N_VOLS = 36
PATCH = (5, 5, 5)
CENTER_ROW = 125 // 2          # center of a 5x5x5 box flattened in C order




def test_recovers_the_noise_level_of_a_pure_noise_volume():
    rng = np.random.default_rng(0)
    sigma_true = 12.0
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * sigma_true).astype(np.float32)
    mask = interior_mask(SHAPE)

    result = denoise_mppca(dwi, mask, patch=PATCH)

    assert abs(np.median(result.sigma[mask]) - sigma_true) < 0.05 * sigma_true


def test_a_noiseless_low_rank_volume_survives_intact():
    """With no noise there is no bulk to find, so the threshold keeps everything.

    That fallback is what makes this safe: nothing is thrown away when the rule has
    nothing to measure.
    """
    rng = np.random.default_rng(1)
    dwi = low_rank_volume(rng, 3, SHAPE, N_VOLS)
    mask = interior_mask(SHAPE)

    result = denoise_mppca(dwi, mask, patch=PATCH)

    scale = np.abs(dwi[mask]).max()
    assert np.abs(result.denoised[mask] - dwi[mask]).max() < 1e-5 * scale


@pytest.mark.parametrize("sigma_true", [2.0, 8.0, 20.0])
def test_the_retained_rank_and_the_noise_level_match_the_signal_they_came_from(sigma_true):
    rng = np.random.default_rng(2)
    signal = low_rank_volume(rng, 3, SHAPE, N_VOLS)
    noisy = (signal + rng.standard_normal(signal.shape) * sigma_true).astype(np.float32)
    mask = interior_mask(SHAPE)

    result = denoise_mppca(noisy, mask, patch=PATCH)

    assert np.median(result.rank[mask]) == 3
    assert (result.rank[mask] <= 4).all()
    assert abs(np.median(result.sigma[mask]) - sigma_true) < 0.05 * sigma_true


def test_denoising_moves_a_noisy_volume_closer_to_its_signal():
    rng = np.random.default_rng(2)
    signal = low_rank_volume(rng, 3, SHAPE, N_VOLS)
    noisy = (signal + rng.standard_normal(signal.shape) * 8.0).astype(np.float32)
    mask = interior_mask(SHAPE)

    result = denoise_mppca(noisy, mask, patch=PATCH)

    before = np.sqrt(np.mean((noisy[mask] - signal[mask]) ** 2))
    after = np.sqrt(np.mean((result.denoised[mask] - signal[mask]) ** 2))
    assert after < 0.5 * before, f"rmse {before:.2f} -> {after:.2f}"


def test_the_center_voxel_matches_a_full_svd_of_its_own_patch():
    """The core eigendecomposes the Gram instead of taking the SVD of the patch.

    For the center row those are the same thing. Here the patch is pulled out with plain
    slicing and reconstructed with numpy's SVD at the rank the denoiser reported.
    """
    rng = np.random.default_rng(3)
    signal = low_rank_volume(rng, 4, SHAPE, N_VOLS)
    dwi = (signal + rng.standard_normal(signal.shape) * 6.0).astype(np.float32)
    voxel = (5, 5, 5)
    mask = np.zeros(SHAPE, bool)
    mask[voxel] = True

    result = denoise_mppca(dwi, mask, patch=PATCH)

    box = dwi[3:8, 3:8, 3:8, :].reshape(125, N_VOLS).astype(np.float64)
    left, values, right = np.linalg.svd(box, full_matrices=False)
    rank = int(result.rank[voxel])
    expected = ((left[:, :rank] * values[:rank]) @ right[:rank])[CENTER_ROW]

    np.testing.assert_allclose(result.denoised[voxel], expected, rtol=1e-4,
                               atol=1e-4 * np.abs(expected).max())


def test_voxels_outside_the_mask_are_left_alone():
    rng = np.random.default_rng(4)
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * 10.0).astype(np.float32)
    mask = interior_mask(SHAPE)

    result = denoise_mppca(dwi, mask, patch=PATCH)

    outside = ~mask
    np.testing.assert_array_equal(result.denoised[outside], dwi[outside])
    assert (result.sigma[outside] == 0).all()
    assert (result.rank[outside] == 0).all()


def test_every_voxel_can_be_denoised_including_the_corners():
    """At the image edge the box moves inward, so a border voxel still gets a full box."""
    rng = np.random.default_rng(5)
    signal = low_rank_volume(rng, 3, SHAPE, N_VOLS)
    dwi = (signal + rng.standard_normal(signal.shape) * 10.0).astype(np.float32)

    result = denoise_mppca(dwi, patch=PATCH)

    assert np.isfinite(result.denoised).all()
    assert (result.sigma > 0).all()
    assert (result.rank > 0).all()


def test_tiling_does_not_change_the_answer():
    rng = np.random.default_rng(7)
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * 10.0).astype(np.float32)
    mask = interior_mask(SHAPE)

    one_tile = denoise_whitened(dwi, mask, None, patch=PATCH, tile=10_000, workers=1)
    many_tiles = denoise_whitened(dwi, mask, None, patch=PATCH, tile=7, workers=1)

    np.testing.assert_array_equal(one_tile[0], many_tiles[0])
    np.testing.assert_array_equal(one_tile[2], many_tiles[2])


def test_the_input_is_not_modified():
    rng = np.random.default_rng(8)
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * 10.0).astype(np.float32)
    untouched = dwi.copy()

    denoise_mppca(dwi, interior_mask(SHAPE), patch=PATCH)

    np.testing.assert_array_equal(dwi, untouched)


def test_the_result_describes_what_was_run():
    rng = np.random.default_rng(9)
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * 10.0).astype(np.float32)

    result = denoise_mppca(dwi, interior_mask(SHAPE), patch=PATCH, algorithm="veraart")

    assert result.denoised.shape == dwi.shape
    assert result.sigma.shape == SHAPE
    assert result.rank.shape == SHAPE
    assert result.model is None, "standard MPPCA does not use a noise model"
    assert result.provenance["method"] == "mppca"
    assert result.provenance["patch"] == PATCH
    assert result.provenance["algorithm"] == "veraart"


@pytest.mark.parametrize("patch, message", [
    ((4, 5, 5), "must be odd"),
    ((5, 5), "must be 3 ints"),
    ((5, 5, 0), "must be >= 1"),
])
def test_a_patch_without_a_center_voxel_is_rejected(patch, message):
    dwi = np.zeros((6, 6, 6, 8), np.float32)
    with pytest.raises(ValueError, match=message):
        denoise_mppca(dwi, patch=patch)


def test_a_volume_that_is_not_4d_is_rejected():
    with pytest.raises(ValueError, match="must be 4D"):
        denoise_mppca(np.zeros((6, 6, 6), np.float32))


def test_a_mask_of_the_wrong_shape_is_rejected():
    dwi = np.zeros((6, 6, 6, 8), np.float32)
    with pytest.raises(ValueError, match="does not match the mask"):
        denoise_mppca(dwi, np.ones((5, 5, 5), bool))


def test_a_mask_that_is_not_3d_or_empty_is_rejected():
    dwi = np.zeros((6, 6, 6, 8), np.float32)
    with pytest.raises(ValueError, match="mask must be 3D"):
        denoise_mppca(dwi, np.ones((6, 6, 6, 8), bool))
    with pytest.raises(ValueError, match="the mask is empty"):
        denoise_mppca(dwi, np.zeros((6, 6, 6), bool))
