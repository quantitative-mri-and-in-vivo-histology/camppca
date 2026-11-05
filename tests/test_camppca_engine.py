"""The covariance-aware engine, checked against paths we write out longhand.

The engine never forms the whitened patch's full reconstruction. It carries only the
centre row of ``C^(1/2)`` and projects that. These tests write the long way out in plain
numpy float64 and demand the same answer.
"""

import numpy as np
import pytest

from camppca import denoise_mppca
from camppca.denoiser.camppca import denoise_whitened, sqrtm_psd, whitened_level
from camppca.denoiser.mp_threshold import mp_threshold
from camppca.patch import patch_offsets, patch_size

SHAPE = (12, 12, 8)
N_VOLS = 40
PATCH = (5, 5, 3)


def separable_correlation(patch, ax=0.6, ay=0.35, az=0.2):
    """Unit-diagonal patch correlation from a separable AR(1) pattern, PSD by construction."""
    offsets = patch_offsets(patch)
    lags = [offsets[:, i][:, None] - offsets[:, i][None, :] for i in range(3)]
    return (ax ** np.abs(lags[0])) * (ay ** np.abs(lags[1])) * (az ** np.abs(lags[2]))


def patch_at(dwi, voxel, patch=PATCH):
    """The patch box around ``voxel``, flattened C order with z fastest."""
    half = np.asarray(patch) // 2
    box = tuple(slice(v - h, v + h + 1) for v, h in zip(voxel, half, strict=True))
    return dwi[box].reshape(patch_size(patch), dwi.shape[3]).astype(np.float64)


def literal_centre_voxel(Y, C, algorithm="cordero-grande"):
    """Whiten the patch, cut, unwhiten the WHOLE patch, then read the centre row out."""
    n_vox, n_vols = Y.shape
    whitened = sqrtm_psd(C, -0.5) @ Y
    evals, evecs = np.linalg.eigh(whitened.T @ whitened)
    evecs = evecs[:, ::-1]
    rank, _ = mp_threshold(np.maximum(evals[::-1], 0.0)[None],
                           min(n_vox, n_vols), max(n_vox, n_vols), algorithm)
    kept = evecs[:, :int(np.asarray(rank)[0])]
    return (sqrtm_psd(C, 0.5) @ (whitened @ kept @ kept.T))[n_vox // 2]


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    dwi = (rng.standard_normal(SHAPE + (N_VOLS,)) * 30 + 100).astype(np.float32)
    mask = np.zeros(SHAPE, bool)
    mask[3:-3, 3:-3, 2:-2] = True
    return dwi, mask


def test_an_identity_whitener_reproduces_standard_mppca(dataset):
    """With C = I there is nothing to whiten, so the two engines must agree."""
    dwi, mask = dataset
    identity = np.eye(patch_size(PATCH))

    denoised, sigma_hat, rank = denoise_whitened(dwi, mask, identity, patch=PATCH)
    plain = denoise_mppca(dwi, mask, patch=PATCH)

    scale = np.abs(plain.denoised[mask]).max()
    assert np.abs(denoised[mask] - plain.denoised[mask]).max() < 1e-5 * scale
    assert np.abs(sigma_hat[mask] - plain.sigma_hat[mask]).max() < 1e-4 * np.median(sigma_hat[mask])
    np.testing.assert_array_equal(rank[mask], plain.rank[mask])


def test_the_centre_row_shortcut_matches_unwhitening_the_whole_patch(dataset):
    """The engine's whole reason for being cheap, checked against the long way."""
    dwi, _ = dataset
    C = separable_correlation(PATCH)
    voxel = (6, 6, 4)
    mask = np.zeros(SHAPE, bool)
    mask[voxel] = True

    denoised, _, _ = denoise_whitened(dwi, mask, C, patch=PATCH)
    expected = literal_centre_voxel(patch_at(dwi, voxel), C)

    np.testing.assert_allclose(denoised[voxel], expected, rtol=1e-4,
                               atol=1e-4 * np.abs(expected).max())


def test_every_masked_voxel_matches_the_long_way(dataset):
    """Not just one voxel: the shortcut has to hold everywhere the patch fits."""
    dwi, _ = dataset
    C = separable_correlation(PATCH)
    mask = np.zeros(SHAPE, bool)
    mask[4:8, 4:8, 2:6] = True

    denoised, _, _ = denoise_whitened(dwi, mask, C, patch=PATCH)

    worst = 0.0
    for voxel in zip(*np.where(mask), strict=True):
        expected = literal_centre_voxel(patch_at(dwi, voxel), C)
        worst = max(worst, np.abs(denoised[voxel] - expected).max() / np.abs(expected).max())
    assert worst < 1e-4, f"worst relative error {worst:.2e}"


def test_the_level_pass_matches_the_full_denoise(dataset):
    """whitened_level stops after the fit, so its sigma_hat is the denoiser's, with and
    without a level field. With C None it is standard MP-PCA's."""
    dwi, mask = dataset
    C = separable_correlation(PATCH)
    level = np.random.default_rng(1).uniform(0.5, 2.0, SHAPE).astype(np.float32)

    for corr, d_field in ((np.eye(patch_size(PATCH)), None), (C, None), (C, level)):
        _, sigma_hat, _ = denoise_whitened(dwi, mask, corr, patch=PATCH, d_field=d_field)
        fast = whitened_level(dwi, mask, corr, patch=PATCH, d_field=d_field)
        np.testing.assert_allclose(fast[mask], sigma_hat[mask], rtol=1e-4)
        assert np.all(fast[~mask] == 0)
    plain = denoise_mppca(dwi, mask, patch=PATCH).sigma_hat
    np.testing.assert_allclose(whitened_level(dwi, mask, None, patch=PATCH)[mask],
                               plain[mask], rtol=1e-4)


def test_a_constant_level_field_changes_nothing(dataset):
    """d carries ratios only, so a flat field must be absorbed by the per-patch fit."""
    dwi, mask = dataset
    C = separable_correlation(PATCH)

    without = denoise_whitened(dwi, mask, C, patch=PATCH)
    with_flat = denoise_whitened(dwi, mask, C, patch=PATCH, d_field=np.full(SHAPE, 3.0, np.float32))

    scale = np.abs(without[0][mask]).max()
    assert np.abs(with_flat[0][mask] - without[0][mask]).max() < 1e-5 * scale


def test_a_varying_level_field_is_undone_at_the_centre_voxel(dataset):
    """Scaling one voxel's noise up and the data with it must give back the same signal."""
    dwi, _ = dataset
    C = separable_correlation(PATCH)
    voxel = (6, 6, 4)
    mask = np.zeros(SHAPE, bool)
    mask[voxel] = True

    plain, _, _ = denoise_whitened(dwi, mask, C, patch=PATCH)

    level = np.ones(SHAPE, np.float32)
    level[voxel] = 2.0
    scaled = dwi.copy()
    scaled[voxel] *= 2.0
    lifted, _, _ = denoise_whitened(scaled, mask, C, patch=PATCH, d_field=level)

    np.testing.assert_allclose(lifted[voxel] / 2.0, plain[voxel], rtol=2e-3)


def test_a_correlation_of_the_wrong_size_is_rejected(dataset):
    dwi, mask = dataset
    with pytest.raises(ValueError, match="C must be"):
        denoise_whitened(dwi, mask, np.eye(7), patch=PATCH)
