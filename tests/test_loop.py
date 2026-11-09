"""The loop, on synthetic data with a KNOWN noise correlation and relative noise map.

These tests check that the loop finds the correlation pattern from a white start,
converges, does not depend on where it starts, that the batched split is the per-patch algebra, that the fit
is the pinned least squares, and that the level rounds recover the relative noise map.
"""

import warnings

import numpy as np
import pytest
from scipy.ndimage import distance_transform_cdt

from camppca import estimate_noise_model
from camppca.denoiser.camppca import sqrtm_psd
from camppca.noise_model import _effective_d, loop
from camppca.patch import table_index

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
    """Colored noise at a level rising 1x to 2x along x, on a rank-4 signal."""
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
    return estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH)


def test_loop_keeps_the_injected_correlation_under_a_varying_level(model):
    rho, want = model.rho, _true_rho()

    assert model.provenance["loop_converged"]
    assert rho[LAG, LAG, 0] == pytest.approx(1.0)
    for k in (1, 2):
        assert rho[LAG + k, LAG, 0] == pytest.approx(want[LAG + k, LAG, 0], abs=0.03)
        assert rho[LAG, LAG + k, 0] == pytest.approx(want[LAG, LAG + k, 0], abs=0.03)
    assert abs(rho[LAG, LAG, 1]) < 0.05, "z was never correlated"


def test_the_loop_tells_a_sheared_correlation_from_its_mirror():
    """Noise correlated between (x, z) and (x+1, z+1) only. A table read by |dz| alone
    would force the same value on (-1, 0, 1)."""
    rng = np.random.default_rng(4)
    shape, n_vols, a = (32, 32, 12), 60, 0.6
    signal = rng.standard_normal(shape + (4,)) @ rng.standard_normal((4, n_vols))
    w = rng.standard_normal(shape + (n_vols,))
    noise = (w + a * np.roll(w, (1, 1), axis=(0, 2))) / np.sqrt(1 + a ** 2)
    dwi = signal * 25 + 400 + 10.0 * noise
    mask = np.zeros(shape, bool)
    mask[3:29, 3:29, 2:10] = True
    model = estimate_noise_model(dwi, mask, patch=PATCH)

    rho = model.rho
    assert rho[table_index(1, 0, 1, LAG)] == pytest.approx(a / (1 + a ** 2), abs=0.05)
    assert abs(rho[table_index(-1, 0, 1, LAG)]) < 0.05


def test_the_loop_pattern_is_a_valid_correlation_for_its_patch(model):
    """The table must pass the near-singular check at its own patch."""
    C = model.patch_correlation(PATCH)
    assert model.provenance["sphericity"] == pytest.approx(loop._sphericity_excess(C))


def test_the_loop_lands_on_the_same_pattern_from_a_white_start(dataset, model):
    dwi, _, mask = dataset
    white = np.zeros_like(model.rho)
    white[LAG, LAG, 0] = 1.0

    from_white, info = loop.fit_rho(dwi, mask, model.d, white, PATCH, LAG)

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


def _bulk_map(sigma, Q):
    """mean_b (I - Q_b) sigma (I - Q_b)^T, one patch at a time."""
    Qm = sum(a @ b.T for a, b in Q) / len(Q)
    cross = sum(a @ (b.T @ sigma @ b) @ a.T for a, b in Q) / len(Q)
    return sigma - Qm @ sigma - sigma @ Qm.T + cross


def _patch_projectors(factors, C):
    """Each patch's kept projector (A_b, Bm_b), in the original patch order."""
    root, W = sqrtm_psd(C, 0.5), sqrtm_psd(C, -0.5)
    Q = [None] * factors.n_b
    for _, k, Ug in factors.groups:
        for i, b in enumerate(np.flatnonzero(factors.rank == k)):
            Q[b] = (root @ Ug[i], W @ Ug[i])
    return Q


def _spread(centers, n):
    """``n`` of the grid's centers spread over the whole mask. The grid lists them in C
    order, so the first ``n`` would all sit at low x, where the test level is lowest."""
    pick = np.linspace(0, centers[0].size - 1, min(n, centers[0].size)).astype(int)
    return tuple(c[pick] for c in centers)


def test_the_batched_split_is_the_per_patch_algebra(dataset, model):
    dwi, _, mask = dataset
    centers = _spread(loop._sample_centers(mask, PATCH)[0], 120)
    patches = loop._gather(dwi, _effective_d(model.d, mask), centers, PATCH)
    C = model.patch_correlation(PATCH)
    # Patches with a clear gap at the cut. Where the kept and dropped singular values are
    # within 10%, float32 and float64 disagree on the projector by up to 8e-3, which is
    # rounding, not algebra. A patch keeping none or all has no gap to test.
    rank = loop._split(patches, C, "cordero-grande")[1].rank
    s = np.linalg.svd(sqrtm_psd(C, -0.5) @ patches.astype(np.float64), compute_uv=False)
    rows, inner = np.arange(len(s)), (rank > 0) & (rank < s.shape[1])
    gap = np.where(inner, s[rows, np.clip(rank - 1, 0, None)]
                   / s[rows, np.clip(rank, None, s.shape[1] - 1)], 0.0)
    clear = np.flatnonzero(gap > 1.1)
    patches = patches[clear[np.linspace(0, clear.size - 1, min(40, clear.size)).astype(int)]]

    S, factors = loop._split(patches, C, "cordero-grande")
    S_ref, Q_ref = _reference_split(patches, C, factors.rank)

    assert factors.rank.min() > 0, "every patch keeps its signal"
    np.testing.assert_allclose(S, S_ref, rtol=1e-4, atol=1e-5)
    for (a, bm), (a_ref, bm_ref) in zip(_patch_projectors(factors, C), Q_ref, strict=True):
        np.testing.assert_allclose(a @ bm.T, a_ref @ bm_ref.T, atol=1e-4)


def test_a_mask_too_thin_for_the_patch_is_refused():
    mask = np.zeros((20, 20, 6), bool)
    mask[3:17, 3:17, 2:4] = True               # two slices, the patch needs three
    with pytest.raises(ValueError, match="no mask voxel has its whole"):
        loop._sample_centers(mask, PATCH)[0]


def _grid_mask():
    mask = np.zeros((60, 60, 30), bool)
    mask[3:57, 3:57, 2:28] = True              # 800 whole 5 x 5 x 3 tiles
    return mask


def test_the_grid_tiles_the_mask_with_whole_patches():
    mask = _grid_mask()
    centers, stride = loop._sample_centers(mask, PATCH)

    assert stride == tuple(PATCH)
    cover = np.zeros(mask.shape, int)
    for c in np.array(centers).T:
        cover[tuple(slice(v - h, v + h + 1) for v, h in zip(c, PATCH // 2, strict=True))] += 1
    assert centers[0].size == 800
    assert cover.max() == 1, "patches at stride = patch share no voxel"
    assert not cover[~mask].any(), "every patch lies inside the mask"
    np.testing.assert_array_equal(centers, loop._sample_centers(mask, PATCH)[0])


def test_the_grid_widens_above_the_cap_and_narrows_below_the_floor(monkeypatch):
    mask = _grid_mask()
    monkeypatch.setattr(loop, "MAX_CENTERS", 300)
    centers, stride = loop._sample_centers(mask, PATCH)
    assert centers[0].size <= 300 and all(s > p for s, p in zip(stride, PATCH, strict=True))

    monkeypatch.setattr(loop, "MAX_CENTERS", 5000)
    monkeypatch.setattr(loop, "MIN_CENTERS", 2000)
    centers, stride = loop._sample_centers(mask, PATCH)
    assert centers[0].size >= 2000 and all(s < p for s, p in zip(stride, PATCH, strict=True))

    tiny = np.zeros((20, 20, 10), bool)
    tiny[6:14, 6:14, 2:7] = True               # 48 eligible centers, fewer than the floor
    centers, stride = loop._sample_centers(tiny, PATCH)
    assert stride == (1, 1, 1) and centers[0].size == 48


def test_lags_the_patch_never_spans_keep_the_white_start(dataset):
    """A 3 x 5 patch has in-plane lag 4 but spans only |dx| <= 2."""
    dwi, bvals, mask = dataset
    looped = estimate_noise_model(dwi, mask, bvals=bvals, patch=(3, 5, 3))
    lag = looped.lag

    assert lag == 4
    for dx in (3, 4):
        assert not looped.rho[lag + dx].any()
        assert not looped.rho[lag - dx].any()
    assert looped.rho[lag + 1, lag, 0] > 0.1, "spanned lags move"


def test_the_convergence_warning_reports_the_last_change(dataset, model, monkeypatch):
    dwi, _, mask = dataset
    monkeypatch.setattr(loop, "MAX_ROUNDS", 1)
    monkeypatch.setattr(loop, "LOOP_TOL", 0.0)
    white = np.zeros_like(model.rho)
    white[LAG, LAG, 0] = 1.0
    with pytest.warns(RuntimeWarning, match="did not converge") as caught:
        loop.fit_rho(dwi, mask, model.d, white, PATCH, LAG)
    assert "by 0.0000" not in str(caught[0].message)


def test_an_unusable_loop_pattern_falls_back_to_white_noise(dataset, monkeypatch):
    """Lag-1 correlation 0.7 alone is indefinite on 5 voxels. Whitening by its projection
    overflowed float32; the loop now stops at the first such round."""
    dwi, bvals, mask = dataset

    def indefinite(S, factors, C, patch, lag, dzmax, start, threads=1):
        rho = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
        rho[lag, lag, 0] = 1.0
        rho[lag - 1, lag, 0] = rho[lag + 1, lag, 0] = 0.7
        return rho

    monkeypatch.setattr(loop, "_fit_table", indefinite)
    with pytest.warns(RuntimeWarning, match="not valid after round 1"):
        model = estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, level_rounds=2)

    white = np.zeros_like(model.rho)
    white[LAG, LAG, 0] = 1.0
    np.testing.assert_array_equal(model.rho, white)
    assert "unstabilized phase" in model.provenance["fallback"]
    assert model.provenance["level_rounds_run"] == 0
    assert model.provenance["sphericity"] == 0.0


def test_the_loop_fits_to_the_denoisers_settings(dataset, model):
    dwi, bvals, mask = dataset
    other = estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, algorithm="veraart",
                                 noise_field=False)

    assert (model.provenance["loop_algorithm"], model.provenance["loop_noise_field"]) == \
        ("cordero-grande", True)
    assert (other.provenance["loop_algorithm"], other.provenance["loop_noise_field"]) == \
        ("veraart", False)
    assert not np.array_equal(other.rho, model.rho)


def test_denoising_with_other_settings_than_the_loop_fitted_warns(dataset, model):
    from camppca import denoise_with_noise_model

    dwi, _, _ = dataset
    with pytest.warns(RuntimeWarning, match="re-estimate|Re-estimate"):
        denoise_with_noise_model(dwi, model, patch=PATCH, noise_field=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        denoise_with_noise_model(dwi, model, patch=PATCH)


def test_an_unknown_algo_is_refused_before_the_estimate(dataset):
    dwi, bvals, mask = dataset
    with pytest.raises(ValueError, match="algorithm must be"):
        estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, algorithm="exp3")


def test_the_fit_is_the_pinned_least_squares(dataset, model):
    """The float32 conjugate-gradient fit against a dense float64 least squares built one
    patch at a time, in whitened units, with the far-class mean held at zero."""
    dwi, _, mask = dataset
    centers = _spread(loop._sample_centers(mask, PATCH)[0], 300)
    patches = loop._gather(dwi, _effective_d(model.d, mask), centers, PATCH)
    C = model.patch_correlation(PATCH)
    S, factors = loop._split(patches, C, "cordero-grande")
    table = loop._fit_table(S, factors, C, PATCH, LAG, PATCH[2] - 1, model.rho)

    W = sqrtm_psd(C, -0.5)
    cls, lags = loop._lag_classes(PATCH)
    Q = _patch_projectors(factors, C)
    design = np.stack([(W @ _bulk_map((cls == c).astype(float), Q) @ W).ravel()
                       for c in range(len(lags))], axis=1)
    reach = np.abs(lags).max(0)
    far = np.any((np.abs(lags) == reach) & (reach > 0), axis=1).astype(float)
    pin = np.eye(len(lags)) - np.outer(np.ones(len(lags)), far) / far.sum()
    values = pin @ np.linalg.lstsq(design @ pin, (W @ S @ W).ravel(), rcond=None)[0]
    ref = np.zeros_like(table)
    for v, (dx, dy, dz) in zip(values, lags, strict=True):
        ref[dx + LAG, dy + LAG, dz] = v
        if dz == 0:
            ref[-dx + LAG, -dy + LAG, 0] = v
    ref /= ref[LAG, LAG, 0]

    assert abs(far @ values) < 1e-10
    assert np.abs(table - ref).max() < 1e-4


def _relative_error(d, mask):
    """|d / true level - 1| after normalising both to their median in the mask."""
    level = np.broadcast_to(np.linspace(1.0, 2.0, mask.shape[0])[:, None, None], mask.shape)
    return np.abs((d / np.median(d[mask])) / (level / np.median(level[mask])) - 1)


def test_every_level_round_fits_the_same_patches(dataset, monkeypatch):
    """The centers came from the current d, so a voxel crossing the level floor between
    rounds changed the set of patches."""
    dwi, bvals, mask = dataset
    drawn = []
    original = loop._sample_centers
    monkeypatch.setattr(loop, "_sample_centers",
                        lambda *a: drawn.append(original(*a)) or drawn[-1])
    estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, level_rounds=2, workers=1)
    assert len(drawn) >= 2
    for later in drawn[1:]:
        for a, b in zip(drawn[0], later, strict=True):
            np.testing.assert_array_equal(a, b)


def test_level_rounds_recover_the_injected_level_and_flatten_s(dataset, model):
    """Away from the mask edge. Edge patches reach outside the mask, where d is only the
    nearest mask voxel's level, so the rim is less exact."""
    dwi, bvals, mask = dataset
    rounds = estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, level_rounds=3)
    interior = mask & (distance_transform_cdt(mask, metric="chessboard") > PATCH.max() // 2)

    assert rounds.provenance["level_converged"]
    assert rounds.provenance["s_spread"] < 0.05, "s is flat at the fixed point"
    assert np.median(_relative_error(rounds.d, mask)[interior]) < 0.015
    assert (np.median(_relative_error(rounds.d, mask)[interior])
            < np.median(_relative_error(model.d, mask)[interior]))


def test_without_the_noise_field_d_is_one_and_no_level_round_runs(dataset):
    dwi, bvals, mask = dataset
    flat = estimate_noise_model(dwi, mask, bvals=bvals, patch=PATCH, noise_field=False,
                                level_rounds=2)

    assert np.all(flat.d[mask] == 1)
    assert flat.provenance["level_rounds_run"] == 0
    assert "s_spread" not in flat.provenance
