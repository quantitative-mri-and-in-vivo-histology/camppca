"""The check of one rho across regions and volume halves, on synthetic noise.

Stationary noise must give small changes everywhere. Noise whose correlation differs in
one half of the image must show up in that half's regions, beyond the random subsets.
"""

import warnings

import numpy as np
import pytest

from camppca import NoiseModel, check_noise_model, estimate_noise_model
from camppca.noise_model import _good_level, loop

PATCH = (5, 5, 3)
H_LEFT = np.array([0.3, 1.0, 0.3])      # correlation along x in the left half
H_RIGHT = np.array([0.7, 1.0, 0.7])     # and in the right half of the other dataset


def _dataset(n_vols, h_right):
    """Noise correlated along x by H_LEFT, and by ``h_right`` for x >= 24, on a rank-4
    signal."""
    rng = np.random.default_rng(1)
    shape, rank = (48, 48, 15), 4
    signal = rng.standard_normal(shape + (rank,)) @ rng.standard_normal((rank, n_vols))
    w = rng.standard_normal(shape + (n_vols,))
    noise = np.empty_like(w)
    for h, half in ((H_LEFT, slice(0, 24)), (h_right, slice(24, 48))):
        colored = np.apply_along_axis(lambda v, h=h: np.convolve(v, h, "same"), 0, w)
        noise[half] = colored[half] / np.sqrt((h ** 2).sum())
    dwi = (signal * 25 + 400 + 10.0 * noise).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[2:46, 2:46, 1:14] = True
    return dwi, mask


@pytest.fixture(scope="module")
def stationary():
    dwi, mask = _dataset(100, H_LEFT)
    model = estimate_noise_model(dwi, mask, patch=PATCH)
    return dwi, model, check_noise_model(dwi, model, random_splits=3, n_noise=256)


def test_stationary_noise_changes_the_level_by_little_anywhere(stationary):
    _, _, check = stationary
    assert len(check.regions) == 8 and len(check.reference) == 3
    assert all(abs(r["change"]) < 0.01 for r in check.regions + check.reference)
    assert check.flagged() == []
    assert len(check.volume_splits) == 3
    for split in check.volume_splits:
        assert split["n_vols"] == (50, 50)
        assert all(abs(c) < 0.01 for c in split["change"])
    assert check.provenance["volume_splits_skipped"] is None
    assert len(check.report()) == 1 + 8 + 1 + 3


def test_a_half_with_other_noise_stands_out_and_few_volumes_skip_the_halves():
    dwi, mask = _dataset(60, H_RIGHT)
    model = estimate_noise_model(dwi, mask, patch=PATCH)
    with pytest.warns(RuntimeWarning, match="volume halves skipped"):
        check = check_noise_model(dwi, model, grid=(2, 1, 1), random_splits=3,
                                  n_noise=256)
    changes = {r["region"]: r["change"] for r in check.regions}
    spread = check.reference_spread()
    # The left half is less correlated than the model, the right half more.
    assert changes["x0 y0 z0"] > 2 * spread and changes["x1 y0 z0"] < -2 * spread
    assert check.flagged() == ["x0 y0 z0", "x1 y0 z0"]
    assert check.volume_splits == []
    assert "fewer than 50" in check.report()[-1]


def test_given_centers_fit_as_the_sampled_ones(stationary):
    dwi, model, _ = stationary
    patch = np.asarray(PATCH)
    start = np.zeros_like(model.rho)
    start[model.lag, model.lag, 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sampled, info = loop.fit_rho(dwi, model.mask, model.d, start, patch, model.lag)
        centers, _ = loop._sample_centers(_good_level(model.d, model.mask), patch)
        given, given_info = loop.fit_rho(dwi, model.mask, model.d, start, patch,
                                         model.lag, centers=centers)
    np.testing.assert_array_equal(given, sampled)
    assert given_info["stride"] is None and info["stride"] is not None


def test_a_model_without_a_fitted_patch_is_refused(stationary):
    dwi, model, _ = stationary
    bare = NoiseModel(rho=model.rho, d=model.d, mask=model.mask, lag=model.lag)
    with pytest.raises(ValueError, match="no patch it was fitted to"):
        check_noise_model(dwi, bare)


def test_the_pooled_comparisons_are_the_serial_ones():
    from camppca.noise_model.qc import _sigma_changes
    idx = np.arange(27)
    pairs = [(0.3 ** np.abs(idx[:, None] - idx[None, :]), np.eye(27)),
             (np.eye(27), 0.2 ** np.abs(idx[:, None] - idx[None, :]))]
    serial = _sigma_changes(pairs, 60, "cordero-grande", 128, 0, workers=1)
    assert _sigma_changes(pairs, 60, "cordero-grande", 128, 0, workers=2) == serial
    assert serial[0] < 0 < serial[1]     # correlated noise read as white comes out low
