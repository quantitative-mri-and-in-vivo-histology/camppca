"""Unwhitening is exact for ANY spectral profile, not just a hard threshold.

The engine reconstructs the centre voxel as

    x_hat_c = e_c^T C^(1/2) ( Yw V diag(f) V^T )
            = ( e_c^T C^(1/2) ) Yw V diag(f) V^T          <- what the engine computes

which is plain associativity. It never uses idempotency of the projector, so `f` being a
0/1 keep-mask is irrelevant to exactness -- a continuous shrinkage profile, or an
arbitrary one, is equally exact.

This matters because it corrects a claim inherited from the porting brief: "shrinkage
breaks the exact-unwhitening identity". It does not. Hard thresholding is a METHOD choice
(sigma_hat sensitivity, and shrinkage would optimize in the whitened metric while the
output lands in data units), not a structural constraint -- see CLAUDE.md.

Run in the full model: per-shell C_s, per-shell sigma_s, per-voxel row scale d.
"""

import numpy as np
import pytest

from camppca.denoiser.camppca import sqrtm_psd
from camppca.noise_model import NoiseModel
from camppca.patch import patch_size

PATCH = (5, 5, 3)
N_VOLS, N_SHELLS = 24, 2


def _patch_correlation(ax, ay, lag, n_vox):
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, lag + 1))
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            rho[i + lag, j + lag, 0] = ax ** abs(i) * ay ** abs(j)
    shape = (4, 4, 4)
    model = NoiseModel(rho=rho[None], group=np.zeros(N_VOLS, int),
                       d=np.ones(shape, np.float32), sigma_p2s=np.ones(shape, np.float32),
                       sigma_shell=np.ones(1), mask=np.ones(shape, bool), lag=lag)
    return model.patch_correlation(PATCH)[0]


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    lag = max(PATCH) - 1
    P = patch_size(PATCH)
    Cs = np.stack([_patch_correlation(0.6, 0.35, lag, P),
                   _patch_correlation(0.25, 0.55, lag, P)])
    group = rng.integers(0, N_SHELLS, N_VOLS)
    sigma_shell = np.array([0.8, 1.3])
    d = rng.uniform(0.5, 2.0, P)                       # per-voxel row scale
    Y = rng.standard_normal((P, N_VOLS)) * 40 + 200
    return rng, Cs, group, sigma_shell, d, Y, P


def _profiles(rng, evals):
    """A projector, a continuous shrinker, and a deliberately meaningless profile."""
    sv = np.sqrt(np.maximum(evals, 0.0))
    gd_like = np.clip(1 - 0.3 / np.maximum(sv / 50, 1e-9), 0.0, 1.0)
    return [("hard threshold", (np.arange(N_VOLS) < 7).astype(float)),
            ("continuous shrinkage", gd_like),
            ("arbitrary profile", rng.uniform(-1, 2, N_VOLS))]


@pytest.mark.parametrize("profile_idx", [0, 1, 2])
def test_centre_row_shortcut_is_exact_for_any_profile(setup, profile_idx):
    rng, Cs, group, sigma_shell, d, Y, P = setup
    centre = P // 2
    W = [sqrtm_psd(C, -0.5) for C in Cs]
    H = [sqrtm_psd(C, 0.5) for C in Cs]

    Yw = np.stack([W[group[n]] @ (Y[:, n] / d) / sigma_shell[group[n]]
                   for n in range(N_VOLS)], 1)
    evals, V = np.linalg.eigh(Yw.T @ Yw)
    V = V[:, ::-1]
    name, f = _profiles(rng, evals[::-1])[profile_idx]

    Xw = Yw @ (V * f) @ V.T

    # LITERAL: unwhiten the whole P x N patch, then take the centre row.
    literal = np.stack([H[group[n]] @ Xw[:, n] * sigma_shell[group[n]] * d
                        for n in range(N_VOLS)], 1)[centre]

    # FUSED: carry only the centre ROW of C^(1/2), one per shell -- what the engine does.
    rows = np.stack([H[s][centre] @ Yw for s in range(N_SHELLS)])
    proj = (rows @ (V * f)) @ V.T
    fused = np.array([proj[group[n], n] * sigma_shell[group[n]] * d[centre]
                      for n in range(N_VOLS)])

    rel = np.abs(literal - fused).max() / np.abs(literal).max()
    assert rel < 1e-12, f"{name}: shortcut must be exact, got {rel:.2e}"


def test_the_shortcut_is_cheaper_than_it_looks(setup):
    """S candidate centre rows, not a P x N reconstruction -- S x the work, not P x."""
    _, Cs, _, _, _, _, P = setup
    assert Cs.shape == (N_SHELLS, P, P)
    assert N_SHELLS < P / 10, "the whole point: S << P"
