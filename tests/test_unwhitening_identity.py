"""Unwhitening is exact for ANY spectral profile, not just a hard threshold.

The engine reconstructs the centre voxel as

    x_hat_c = e_c^T C^(1/2) ( Yw V diag(f) V^T )
            = ( e_c^T C^(1/2) ) Yw V diag(f) V^T          <- what the engine computes

which is plain associativity. It never uses idempotency of the projector, so `f` being a
0/1 keep-mask is irrelevant to exactness. A continuous shrinkage profile, or an arbitrary
one, is equally exact.

Hard thresholding is therefore a METHOD choice, made for sigma_hat sensitivity and because
shrinkage would optimize in the whitened metric while the output lands in data units. It
is not a structural constraint. See CLAUDE.md.

Run with the row scale d in play, since that is the other half of the whitener.
"""

import numpy as np
import pytest

from camppca.denoiser.camppca import sqrtm_psd
from camppca.noise_model import NoiseModel
from camppca.patch import patch_size

PATCH = (5, 5, 3)
N_VOLS = 24


def _patch_correlation(ax, ay, lag):
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, lag + 1))
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            rho[i + lag, j + lag, 0] = ax ** abs(i) * ay ** abs(j)
    shape = (4, 4, 4)
    model = NoiseModel(rho=rho, d=np.ones(shape, np.float32),
                       sigma_p2s=np.ones(shape, np.float32),
                       mask=np.ones(shape, bool), lag=lag)
    return model.patch_correlation(PATCH)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    n_vox = patch_size(PATCH)
    C = _patch_correlation(0.6, 0.35, max(PATCH) - 1)
    d = rng.uniform(0.5, 2.0, n_vox)                   # per-voxel row scale
    Y = rng.standard_normal((n_vox, N_VOLS)) * 40 + 200
    return rng, C, d, Y, n_vox


def _profiles(rng, evals):
    """A projector, a continuous shrinker, and a deliberately meaningless profile."""
    sv = np.sqrt(np.maximum(evals, 0.0))
    gd_like = np.clip(1 - 0.3 / np.maximum(sv / 50, 1e-9), 0.0, 1.0)
    return [("hard threshold", (np.arange(N_VOLS) < 7).astype(float)),
            ("continuous shrinkage", gd_like),
            ("arbitrary profile", rng.uniform(-1, 2, N_VOLS))]


@pytest.mark.parametrize("profile_idx", [0, 1, 2])
def test_centre_row_shortcut_is_exact_for_any_profile(setup, profile_idx):
    rng, C, d, Y, n_vox = setup
    centre = n_vox // 2
    whitener, unwhitener = sqrtm_psd(C, -0.5), sqrtm_psd(C, 0.5)

    Yw = whitener @ (Y / d[:, None])
    evals, V = np.linalg.eigh(Yw.T @ Yw)
    V = V[:, ::-1]
    name, f = _profiles(rng, evals[::-1])[profile_idx]

    Xw = Yw @ (V * f) @ V.T

    # LITERAL: unwhiten the whole n_vox x N patch, then take the centre row.
    literal = (unwhitener @ Xw * d[:, None])[centre]

    # FUSED: carry only the centre ROW of C^(1/2) -- what the engine does.
    fused = ((unwhitener[centre] @ Yw) @ (V * f)) @ V.T * d[centre]

    rel = np.abs(literal - fused).max() / np.abs(literal).max()
    assert rel < 1e-12, f"{name}: shortcut must be exact, got {rel:.2e}"


def test_the_shortcut_carries_one_row_not_the_whole_matrix(setup):
    """The saving: one row of C^(1/2), not an n_vox x N reconstruction."""
    _, C, _, _, n_vox = setup
    assert C.shape == (n_vox, n_vox)
    assert sqrtm_psd(C, 0.5)[n_vox // 2].shape == (n_vox,)
