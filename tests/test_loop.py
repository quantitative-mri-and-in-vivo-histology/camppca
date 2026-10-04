"""The loop kernel, on synthetic data with a KNOWN noise correlation and level field.

P2S is already close to unbiased on data like this: the signal is low rank and the mask
is large. These tests check that the loop keeps the kernel right, converges, does not
depend on where it starts, and that the batched solve is the per-patch algebra.
"""

import jax
import numpy as np
import pytest

from camppca import estimate_noise_model
from camppca.denoiser import _effective_d
from camppca.denoiser.camppca import sqrtm_psd
from camppca.noise_model import loop

HX = np.array([0.3, 1.0, 0.3])          # in-plane correlation along x
HY = np.array([0.15, 1.0, 0.15])        # and along y; z left uncorrelated
PATCH = np.array((5, 5, 3))
LAG = 4


def _filter_rho(h, lag):
    full = np.correlate(h, h, mode="full")
    mid = len(h) - 1
    return np.array([full[mid + k] if abs(k) < len(h) else 0.0
                     for k in range(-lag, lag + 1)]) / full[mid]


def _true_rho():
    rho = np.zeros((2 * LAG + 1, 2 * LAG + 1, PATCH[2]))
    rho[:, :, 0] = np.outer(_filter_rho(HX, LAG), _filter_rho(HY, LAG))
    return rho


@pytest.fixture(scope="module")
def dataset():
    """Coloured noise at a level rising 1x to 2x along x, on a rank-4 signal."""
    rng = np.random.default_rng(0)
    shape, n_vols, rank = (40, 40, 12), 60, 4
    bvals = np.concatenate([np.zeros(6), np.full(n_vols - 6, 1000.0)])
    signal = rng.standard_normal(shape + (rank,)) @ rng.standard_normal((rank, n_vols))
    w = rng.standard_normal(shape + (n_vols,))
    noise = np.apply_along_axis(lambda v: np.convolve(v, HX, "same"), 0, w)
    noise = np.apply_along_axis(lambda v: np.convolve(v, HY, "same"), 1, noise)
    noise /= np.sqrt((HX ** 2).sum() * (HY ** 2).sum())
    level = np.linspace(1.0, 2.0, shape[0])[:, None, None, None]
    dwi = signal * 25 + 400 + 10.0 * level * noise
    mask = np.zeros(shape, bool)
    mask[4:36, 4:36, 2:10] = True
    return dwi, bvals, mask


@pytest.fixture(scope="module")
def model(dataset):
    dwi, bvals, mask = dataset
    return estimate_noise_model(dwi, bvals, mask, patch=PATCH)


def test_loop_keeps_the_injected_correlation_under_a_varying_level(model):
    rho, want = model.rho, _true_rho()

    assert model.provenance["kernel"] == "loop"
    assert model.provenance["loop_converged"]
    assert rho[LAG, LAG, 0] == pytest.approx(1.0)
    for k in (1, 2):
        assert rho[LAG + k, LAG, 0] == pytest.approx(want[LAG + k, LAG, 0], abs=0.03)
        assert rho[LAG, LAG + k, 0] == pytest.approx(want[LAG, LAG + k, 0], abs=0.03)
    assert abs(rho[LAG, LAG, 1]) < 0.05, "z was never correlated"


def test_the_loop_kernel_is_a_valid_correlation_for_its_patch(model):
    """The final least squares fills the patch-mean direction the Richardson solve
    leaves empty, so the table must pass the near-singular check."""
    C = model.patch_correlation(PATCH)
    assert model.provenance["u_kernel"] == pytest.approx(loop._sphericity_excess(C))


def test_the_loop_lands_on_the_same_kernel_from_a_white_start(dataset, model):
    dwi, _, mask = dataset
    white = np.zeros_like(model.rho)
    white[LAG, LAG, 0] = 1.0

    from_white, info = loop.loop_kernel(dwi, mask, model.d, white, PATCH, LAG, seed=0)

    assert info["converged"]
    assert np.abs(from_white - model.rho).max() < 0.01


def _reference_split(patches, C, ranks):
    """One patch at a time, in numpy, at the ranks given."""
    W, L = sqrtm_psd(C, -0.5), sqrtm_psd(C, 0.5)
    n_vox, n_vols = patches.shape[1:]
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    S, Q = np.zeros((n_vox, n_vox)), []
    for x, r in zip(patches, ranks, strict=True):
        xw = W @ x
        lam, V = np.linalg.eigh(xw.T @ xw)
        lam, V = lam[::-1], V[:, ::-1]
        v = lam[r:n_small].sum() / ((n_small - r) * (n_large - r))
        U = xw @ V[:, :r] / np.sqrt(lam[:r])
        resid = xw - xw @ V[:, :r] @ V[:, :r].T
        S += L @ (resid @ resid.T / (n_vols * v)) @ L
        Q.append((L @ U, W @ U))
    return S / len(patches), Q


def _reference_solve(S, Q):
    Qm = sum(a @ b.T for a, b in Q) / len(Q)
    sigma = S.copy()
    for _ in range(loop.N_INNER):
        cross = sum(a @ (b.T @ sigma @ b) @ a.T for a, b in Q) / len(Q)
        sigma = S + Qm @ sigma + sigma @ Qm.T - cross
        sigma = 0.5 * (sigma + sigma.T)
    return sigma


def test_the_batched_split_and_solve_are_the_per_patch_algebra(dataset, model):
    dwi, _, mask = dataset
    centres = loop._sample_centres(mask, PATCH, np.random.default_rng(1))
    patches = loop._gather(dwi, _effective_d(model.d, mask), centres, PATCH)[:40]
    C = model.patch_correlation(PATCH)

    with jax.enable_x64(True):
        S, A, Bm, rank = loop._split(patches, C)
        sigma = np.asarray(loop._solve(S, A, Bm))
        A, Bm = np.asarray(A), np.asarray(Bm)
    S_ref, Q_ref = _reference_split(patches, C, rank)

    assert rank.min() > 0, "every patch keeps its signal"
    np.testing.assert_allclose(S, S_ref, rtol=1e-8, atol=1e-10)
    for b, (a, bm) in enumerate(Q_ref):
        np.testing.assert_allclose(A[b] @ Bm[b].T, a @ bm.T, atol=1e-8)
    np.testing.assert_allclose(sigma, _reference_solve(S_ref, Q_ref), atol=1e-8)


def test_kernel_p2s_skips_the_loop(dataset, model):
    dwi, bvals, mask = dataset
    p2s = estimate_noise_model(dwi, bvals, mask, patch=PATCH, kernel="p2s")

    assert p2s.provenance["kernel"] == "p2s"
    assert "loop_rounds" not in p2s.provenance
    assert np.array_equal(p2s.d, model.d), "the loop changes rho only"
    assert not np.array_equal(p2s.rho, model.rho)


def test_an_unknown_kernel_is_refused(dataset):
    dwi, bvals, mask = dataset
    with pytest.raises(ValueError, match="kernel must be"):
        estimate_noise_model(dwi, bvals, mask, patch=PATCH, kernel="kernel")


def test_a_mask_too_thin_for_the_patch_is_refused():
    mask = np.zeros((20, 20, 6), bool)
    mask[3:17, 3:17, 2:4] = True               # two slices, the patch needs three
    with pytest.raises(ValueError, match="no mask voxel has its whole"):
        loop._sample_centres(mask, PATCH, np.random.default_rng(0))


def test_lags_the_patch_never_spans_keep_the_p2s_values(dataset):
    """A 3 x 5 patch has in-plane lag 4 but spans only |dx| <= 2."""
    dwi, bvals, mask = dataset
    patch = (3, 5, 3)
    p2s = estimate_noise_model(dwi, bvals, mask, patch=patch, kernel="p2s")
    looped = estimate_noise_model(dwi, bvals, mask, patch=patch)
    lag = looped.lag

    assert lag == 4
    for dx in (3, 4):
        np.testing.assert_array_equal(looped.rho[lag + dx], p2s.rho[lag + dx])
        np.testing.assert_array_equal(looped.rho[lag - dx], p2s.rho[lag - dx])
    assert not np.array_equal(looped.rho[lag + 1], p2s.rho[lag + 1]), "spanned lags move"


def test_the_convergence_warning_reports_the_last_change(dataset, model, monkeypatch):
    dwi, _, mask = dataset
    monkeypatch.setattr(loop, "MAX_ROUNDS", 1)
    monkeypatch.setattr(loop, "LOOP_TOL", 0.0)
    with pytest.warns(RuntimeWarning, match="did not converge") as caught:
        loop.loop_kernel(dwi, mask, model.d, model.rho, PATCH, LAG, seed=0)
    assert "by 0.0000" not in str(caught[0].message)


def test_an_unusable_loop_kernel_is_refused_at_estimation(dataset, monkeypatch):
    """Lag-1 correlation 0.7 alone is indefinite on 5 voxels."""
    dwi, bvals, mask = dataset

    def indefinite(S, A, Bm, patch, lag, dzmax):
        rho = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
        rho[lag, lag, 0] = 1.0
        rho[lag - 1, lag, 0] = rho[lag + 1, lag, 0] = 0.7
        return rho

    monkeypatch.setattr(loop, "_stationary_table", indefinite)
    with pytest.raises(ValueError, match="loop kernel is not usable"):
        estimate_noise_model(dwi, bvals, mask, patch=PATCH)
