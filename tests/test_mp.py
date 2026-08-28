"""The Marchenko-Pastur threshold, checked against spectra we construct ourselves.

Every case here builds its own matrix from a known noise level and a known signal rank,
so what the threshold returns can be compared to the truth rather than to another
implementation.
"""

import numpy as np
import pytest

from camppca.mp import ALGORITHMS, mp_threshold

N_VOX, N_VOLS = 125, 40
N_SMALL, N_LARGE = min(N_VOX, N_VOLS), max(N_VOX, N_VOLS)


def spectrum(matrix):
    """Descending eigenvalues of ``matrix.T @ matrix``, batched to one."""
    evals = np.linalg.eigvalsh(matrix.T @ matrix)
    return np.maximum(evals[::-1], 0.0)[None, :].astype(np.float32)


def noise_matrix(rng, sigma, shape=(N_VOX, N_VOLS)):
    return (rng.standard_normal(shape) * sigma).astype(np.float32)


def test_recovers_the_noise_level_of_a_pure_noise_matrix():
    rng = np.random.default_rng(0)
    sigma_true = 7.5
    estimates = [float(mp_threshold(spectrum(noise_matrix(rng, sigma_true)),
                                    N_SMALL, N_LARGE)[1][0]) for _ in range(20)]
    assert abs(np.median(estimates) - sigma_true) < 0.05 * sigma_true


def test_pure_noise_retains_almost_nothing():
    rng = np.random.default_rng(1)
    ranks = [int(mp_threshold(spectrum(noise_matrix(rng, 1.0)), N_SMALL, N_LARGE)[0][0])
             for _ in range(20)]
    assert max(ranks) <= 3, f"pure noise kept up to {max(ranks)} components"


@pytest.mark.parametrize("true_rank", [1, 3, 6])
def test_finds_a_signal_that_stands_well_clear_of_the_noise(true_rank):
    rng = np.random.default_rng(2)
    left = rng.standard_normal((N_VOX, true_rank))
    right = rng.standard_normal((true_rank, N_VOLS))
    signal = 60.0 * left @ right
    data = (signal + noise_matrix(rng, 1.0)).astype(np.float32)
    rank, sigma_hat = mp_threshold(spectrum(data), N_SMALL, N_LARGE)
    assert int(rank[0]) == true_rank
    assert abs(float(sigma_hat[0]) - 1.0) < 0.1


def test_the_noise_level_is_recovered_alongside_a_strong_signal():
    rng = np.random.default_rng(3)
    sigma_true = 4.0
    signal = 200.0 * rng.standard_normal((N_VOX, 3)) @ rng.standard_normal((3, N_VOLS))
    data = (signal + noise_matrix(rng, sigma_true)).astype(np.float32)
    _, sigma_hat = mp_threshold(spectrum(data), N_SMALL, N_LARGE)
    assert abs(float(sigma_hat[0]) - sigma_true) < 0.1 * sigma_true


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_agrees_on_a_well_separated_case(algorithm):
    rng = np.random.default_rng(4)
    signal = 100.0 * rng.standard_normal((N_VOX, 2)) @ rng.standard_normal((2, N_VOLS))
    data = (signal + noise_matrix(rng, 1.0)).astype(np.float32)
    rank, sigma_hat = mp_threshold(spectrum(data), N_SMALL, N_LARGE, algorithm)
    assert int(rank[0]) == 2
    assert abs(float(sigma_hat[0]) - 1.0) < 0.15


def test_a_batch_is_thresholded_independently():
    """Each row of a batch must get the answer it would get on its own."""
    rng = np.random.default_rng(5)
    batch = []
    for r in (0, 2, 5):
        signal = (100.0 * rng.standard_normal((N_VOX, r)) @ rng.standard_normal((r, N_VOLS))
                  if r else 0.0)
        batch.append(spectrum((signal + noise_matrix(rng, 1.0)).astype(np.float32))[0])
    batch = np.stack(batch)

    together = mp_threshold(batch, N_SMALL, N_LARGE)
    alone = [mp_threshold(row[None], N_SMALL, N_LARGE) for row in batch]

    assert list(np.asarray(together[0])) == [int(r[0]) for r, _ in alone]
    np.testing.assert_allclose(together[1], [float(s[0]) for _, s in alone], rtol=1e-6)


def test_the_noise_level_stays_positive_and_finite_for_a_degenerate_spectrum():
    _, sigma_hat = mp_threshold(np.zeros((1, N_VOLS), np.float32), N_SMALL, N_LARGE)
    assert float(sigma_hat[0]) > 0
    assert np.isfinite(sigma_hat[0])


def test_an_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm must be one of"):
        mp_threshold(np.ones((1, N_VOLS), np.float32), N_SMALL, N_LARGE, "gavish-donoho")
