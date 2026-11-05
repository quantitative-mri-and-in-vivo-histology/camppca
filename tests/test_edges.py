"""At the image edge the patch box moves inward, as in dwidenoise. It does not wrap."""

import numpy as np
import pytest

from camppca import denoise_mppca
from camppca.denoiser.camppca import denoise_whitened, sqrtm_psd
from camppca.denoiser.mp_threshold import mp_threshold
from camppca.denoiser.spectrum import eigh_descending
from camppca.patch import patch_offsets

SHAPE, N_VOLS, PATCH = (11, 10, 9), 60, (5, 5, 3)


@pytest.fixture(scope="module")
def dwi():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(SHAPE + (3,)) @ rng.standard_normal((3, N_VOLS))
    return (signal * 30 + 200 + 5 * rng.standard_normal(SHAPE + (N_VOLS,))).astype(np.float32)


def _correlation(patch, ax=0.5, ay=0.3):
    off = patch_offsets(patch)
    return (ax ** np.abs(off[:, 0][:, None] - off[:, 0][None, :])
            * ay ** np.abs(off[:, 1][:, None] - off[:, 1][None, :]))


def _by_hand(dwi, voxel, C=None, d=None):
    """One voxel the long way: its inward box, whiten, cut, rebuild its own row."""
    half = np.array(PATCH) // 2
    start = np.clip(np.array(voxel) - half, 0, np.array(SHAPE) - np.array(PATCH))
    idx = tuple((start + patch_offsets(PATCH)).T)
    row = int(np.flatnonzero((np.array(idx).T == voxel).all(1))[0])
    levels = np.ones(len(idx[0]), np.float32) if d is None else d[idx]
    X = dwi[idx] / levels[:, None]
    Y = X if C is None else sqrtm_psd(C, -0.5).astype(np.float32) @ X
    evals, V = eigh_descending((Y.T @ Y)[None])
    n_vox = len(idx[0])
    r = int(mp_threshold(evals, min(n_vox, N_VOLS), max(n_vox, N_VOLS))[0][0])
    V = V[0, :, :r].astype(np.float64)
    # Unwhitened with C^(1/2) C^(-1/2) = I: the voxel's own row of X, projected. Through
    # C^(1/2) times the float32 whitener the hand calculation itself was off by 0.5.
    return (X[row].astype(np.float64) @ V @ V.T) * levels[row]


EDGE_VOXELS = [(0, 0, 0), (0, 5, 4), (10, 9, 8), (1, 4, 0), (5, 0, 8)]


@pytest.mark.parametrize("voxel", EDGE_VOXELS)
def test_an_edge_voxel_is_denoised_from_its_inward_box(dwi, voxel):
    C = _correlation(PATCH)
    d = np.linspace(1.0, 1.5, SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE,
                                                                                  np.float32)
    mask = np.zeros(SHAPE, bool)
    mask[voxel] = True

    white = denoise_mppca(dwi, mask, patch=PATCH, workers=1).denoised[voxel]
    whitened = denoise_whitened(dwi, mask, C, patch=PATCH, d_field=d, workers=1)[0][voxel]

    # The wrapped box was off by 0.1 to 0.7 of the scale.
    scale = np.abs(dwi).max()
    assert np.abs(white - _by_hand(dwi, voxel)).max() < 1e-4 * scale
    assert np.abs(whitened - _by_hand(dwi, voxel, C, d)).max() < 1e-4 * scale


def test_the_opposite_face_does_not_reach_an_edge_voxel(dwi):
    """With a wrap, the last slices sat in the first slice's patches."""
    far = dwi.copy()
    far[-2:] = 0
    mask = np.zeros(SHAPE, bool)
    mask[0] = True
    a = denoise_mppca(dwi, mask, patch=PATCH, workers=1)
    b = denoise_mppca(far, mask, patch=PATCH, workers=1)
    np.testing.assert_array_equal(a.denoised[0], b.denoised[0])
    np.testing.assert_array_equal(a.sigma_hat[0], b.sigma_hat[0])
