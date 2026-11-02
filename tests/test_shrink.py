"""Shrinkage of the kept components: "gd" and "lr" against the hard threshold."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from camppca import NoiseModel, denoise_camppca, denoise_mppca, denoise_with_noise_model
from camppca.denoiser.camppca import sqrtm_psd
from camppca.denoiser.mp_threshold import mp_threshold
from camppca.denoiser.shrink import shrink_weight, signal_alignment

PATCH = (5, 5, 3)


def _gavish_donoho(s, sigma, m, n):
    """Gavish & Donoho's Frobenius shrinker for an m x n matrix, m <= n, written out."""
    beta = m / n
    y = s / (sigma * np.sqrt(n))
    eta = np.where(y > 1 + np.sqrt(beta),
                   np.sqrt(np.maximum((y ** 2 - beta - 1) ** 2 - 4 * beta, 0.0))
                   / np.maximum(y, 1e-12), 0.0)
    return eta * sigma * np.sqrt(n)


def _spectra(rng, n_vox, n_vols, n_patches=16):
    """Whitened-like patches: a few strong components on unit white noise."""
    signal = rng.standard_normal((n_patches, n_vox, 3)) @ rng.standard_normal((3, n_vols))
    X = 3.0 * signal + rng.standard_normal((n_patches, n_vox, n_vols))
    evals, evecs = np.linalg.eigh(np.einsum("bpn,bpm->bnm", X, X))
    return X, np.maximum(evals[:, ::-1], 0.0), evecs[:, :, ::-1]


@pytest.mark.parametrize("n_vox, n_vols", [(75, 40), (27, 60)])
def test_gd_weight_is_gavish_donohos_shrinker(n_vox, n_vols):
    _, evals, _ = _spectra(np.random.default_rng(0), n_vox, n_vols)
    sigma = np.full(len(evals), 1.0)
    m, n = min(n_vox, n_vols), max(n_vox, n_vols)

    weight = np.asarray(shrink_weight(jnp.asarray(evals), jnp.asarray(sigma), n_vox, n_vols))
    s = np.sqrt(evals)
    want = _gavish_donoho(s, sigma[:, None], m, n) / np.maximum(s, 1e-12)

    np.testing.assert_allclose(weight, want, atol=1e-6)
    assert weight.max() <= 1.0 and weight.min() >= 0.0
    assert (weight[:, 0] > 0).all() and (weight[:, -1] == 0).all()


def test_lr_with_white_noise_is_gd():
    X, evals, evecs = _spectra(np.random.default_rng(1), 75, 40)
    sigma = jnp.ones(len(evals))
    with jax.default_matmul_precision("highest"):     # as the engine calls it; GPU: not TF32
        q = signal_alignment(jnp.asarray(X), jnp.asarray(evecs), jnp.asarray(evals),
                             jnp.eye(75))

    np.testing.assert_allclose(np.asarray(q[:, :3]), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(shrink_weight(jnp.asarray(evals), sigma, 75, 40, q)),
        np.asarray(shrink_weight(jnp.asarray(evals), sigma, 75, 40)), atol=1e-6)


def test_lr_keeps_more_where_the_signal_lies_in_loud_noise():
    """Smooth signal under smooth correlated noise: q > 1, so mu_tau < 1 and "lr" keeps
    more of the top component than "gd"."""
    rng = np.random.default_rng(2)
    g = np.arange(5) - 2
    off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    C = np.exp(-np.linalg.norm(off[:, None] - off[None], axis=-1) ** 2 / 2.0)
    n_vox, n_vols = len(off), 90
    smooth = np.exp(-np.linalg.norm(off, axis=1) ** 2 / 8.0)
    root = sqrtm_psd(C, 0.5)
    X = np.stack([np.outer(smooth, 6 * rng.standard_normal(n_vols))
                  + root @ rng.standard_normal((n_vox, n_vols)) for _ in range(16)])
    whitened = np.einsum("pq,bqn->bpn", sqrtm_psd(C, -0.5), X)
    evals, evecs = np.linalg.eigh(np.einsum("bpn,bpm->bnm", whitened, whitened))
    evals, evecs = jnp.asarray(np.maximum(evals[:, ::-1], 0.0)), jnp.asarray(evecs[:, :, ::-1])
    _, sigma = mp_threshold(evals, min(n_vox, n_vols), max(n_vox, n_vols))

    q = signal_alignment(jnp.asarray(whitened), evecs, evals, jnp.asarray(C))
    gd = shrink_weight(evals, sigma, n_vox, n_vols)
    lr = shrink_weight(evals, sigma, n_vox, n_vols, q)

    assert float(jnp.median(q[:, 0])) > 1.0
    assert (lr[:, 0] > gd[:, 0]).all()


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(3)
    shape, n_vols = (20, 20, 8), 40
    signal = rng.standard_normal(shape + (3,)) @ rng.standard_normal((3, n_vols))
    dwi = (signal * 20 + 300 + 8 * rng.standard_normal(shape + (n_vols,))).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[3:17, 3:17, 2:6] = True
    return dwi, mask


def _white_model(mask):
    lag = 4
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, PATCH[2]))
    rho[lag, lag, 0] = 1.0
    ones = np.ones(mask.shape, np.float32)
    return NoiseModel(rho=rho, d=ones, mask=mask, lag=lag)


def test_shrinkage_changes_the_weights_only(dataset):
    dwi, mask = dataset
    model = _white_model(mask)
    out = {k: denoise_with_noise_model(dwi, model, patch=PATCH, shrink=k)
           for k in ("hard", "gd", "lr")}
    std = {k: denoise_mppca(dwi, mask, patch=PATCH, shrink=k) for k in ("hard", "gd", "lr")}

    for results in (out, std):
        for k in ("gd", "lr"):
            assert np.array_equal(results[k].rank, results["hard"].rank)
            assert np.array_equal(results[k].sigma_hat, results["hard"].sigma_hat)
            assert not np.array_equal(results[k].denoised[mask], results["hard"].denoised[mask])
        assert results["gd"].provenance["shrink"] == "gd"
    np.testing.assert_allclose(out["lr"].denoised, out["gd"].denoised, rtol=1e-5)
    assert np.array_equal(std["lr"].denoised, std["gd"].denoised)


def test_an_unknown_shrinker_is_refused(dataset):
    dwi, mask = dataset
    with pytest.raises(ValueError, match="shrink must be"):
        denoise_mppca(dwi, mask, patch=PATCH, shrink="soft")
    with pytest.raises(ValueError, match="shrink must be"):
        denoise_with_noise_model(dwi, _white_model(mask), patch=PATCH, shrink="soft")
    with pytest.raises(ValueError, match="shrink must be"):
        denoise_camppca(dwi, mask=mask, patch=PATCH, shrink="soft")

