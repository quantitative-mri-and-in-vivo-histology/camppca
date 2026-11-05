"""Patch averaging (aggregate="mean"), checked against the long way written out in float64.

The reference rebuilds every patch centred in the mask, wraps it into the volume,
accumulates only into mask voxels, and divides. For camppca it unwhitens with an explicit
``D C^(1/2)``, which also checks that the engine may skip that product.

Each patch's rank, sigma_hat and eigenvectors come from the float32 split of that one
patch, as in the engine: in float64 about 6% of these patches cut one component lower, a
near-tie at the MP edge, and eigenvectors near the cut, whose eigenvalues nearly
coincide, span a slightly different subspace. The split is tested elsewhere; here the
rebuild, the unwhitening and the averaging are, in float64.
"""

import numpy as np
import pytest

from camppca import denoise_mppca
from camppca.denoiser import camppca as engine
from camppca.denoiser.aggregate import check_aggregate, patch_centres
from camppca.denoiser.camppca import denoise_whitened, sqrtm_psd
from camppca.denoiser.spectrum import eigh_descending
from camppca.patch import patch_offsets

SHAPE = (14, 12, 8)
N_VOLS = 40
PATCH = (3, 3, 3)


@pytest.fixture(scope="module")
def dataset():
    """A clear rank-3 signal, so float32 and float64 agree on every patch's rank."""
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(SHAPE + (3,)) @ rng.standard_normal((3, N_VOLS))
    dwi = (signal * 30 + 200 + rng.standard_normal(SHAPE + (N_VOLS,))).astype(np.float32)
    mask = np.zeros(SHAPE, bool)
    mask[2:12, 1:11, 1:7] = True
    mask[0, 0, 0] = True                      # a lone voxel at the corner: wraps, no neighbours
    d = np.linspace(1.0, 1.6, SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(
        SHAPE, np.float32)
    return dwi, mask, d


def correlation(patch, ax=0.5, ay=0.3, az=0.15):
    off = patch_offsets(patch)
    lags = [off[:, i][:, None] - off[:, i][None, :] for i in range(3)]
    return (ax ** np.abs(lags[0])) * (ay ** np.abs(lags[1])) * (az ** np.abs(lags[2]))


def reference_mean(dwi, mask, centres, C=None, d=None):
    """Brute force in float64. Returns (denoised, sigma_hat, rank) like the engine."""
    shape = np.array(SHAPE)
    off = patch_offsets(PATCH) - np.array(PATCH) // 2
    n_vox = len(off)
    n_small, n_large = min(n_vox, N_VOLS), max(n_vox, N_VOLS)
    W = None if C is None else sqrtm_psd(C, -0.5)
    root = None if C is None else sqrtm_psd(C, 0.5)
    W32 = None if C is None else W.astype(np.float32)
    crow = None if C is None else root[n_vox // 2].astype(np.float32)
    total = np.zeros(SHAPE + (N_VOLS,))
    count = np.zeros(SHAPE)
    var = np.zeros(SHAPE)
    ranks = np.zeros(SHAPE)
    for c in np.array(np.unravel_index(centres, SHAPE)).T:
        idx = tuple(((c + off) % shape).T)
        X = dwi[idx].astype(np.float64)
        levels = np.ones(n_vox) if d is None else d[idx].astype(np.float64)
        Y = X / levels[:, None]
        if W is not None:
            Y = W @ Y
        patch32 = (X / levels[:, None]).astype(np.float32)[None]
        Y32 = patch32[0] if W32 is None else W32 @ patch32[0]
        V = eigh_descending((Y32.T @ Y32)[None])[1][0].astype(np.float64)
        C32 = None if C is None else C.astype(np.float32)
        _, s, r = engine._denoise_tile(patch32, W32, crow, C32, n_small, n_large,
                                       "cordero-grande", "hard")
        r, s = int(r[0]), float(s[0])
        rebuilt = Y @ V[:, :r] @ V[:, :r].T
        if root is not None:
            rebuilt = root @ rebuilt
        rebuilt = rebuilt * levels[:, None]
        inside = mask[idx]
        for k in np.flatnonzero(inside):
            v = tuple(np.array(idx)[:, k])
            total[v] += rebuilt[k]
            count[v] += 1
            var[v] += s ** 2
            ranks[v] += r
    m = count > 0
    denoised = dwi.astype(np.float64).copy()
    denoised[m] = total[m] / count[m][:, None]
    sigma = np.where(m, np.sqrt(var / np.maximum(count, 1)), 0)
    return denoised, sigma, np.where(m, ranks / np.maximum(count, 1), 0)


@pytest.mark.parametrize("stride", [1, 2])
def test_mppca_mean_is_the_long_way(dataset, stride):
    dwi, mask, _ = dataset
    got = denoise_mppca(dwi, mask, patch=PATCH, aggregate="mean", stride=stride, workers=1)
    want = reference_mean(dwi, mask, patch_centres(mask, PATCH, stride))

    np.testing.assert_allclose(got.rank, want[2], atol=1e-6)
    np.testing.assert_allclose(got.denoised, want[0], rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(got.sigma_hat, want[1], rtol=1e-4, atol=1e-5)
    assert got.provenance["aggregate"] == "mean" and got.provenance["stride"] == stride


@pytest.mark.parametrize("stride", [1, 2])
def test_camppca_mean_is_the_long_way_with_explicit_unwhitening(dataset, stride):
    dwi, mask, d = dataset
    C = correlation(PATCH)
    got = denoise_whitened(dwi, mask, C, patch=PATCH, d_field=d, aggregate="mean",
                           stride=stride, workers=1)
    want = reference_mean(dwi, mask, patch_centres(mask, PATCH, stride), C=C, d=d)

    np.testing.assert_allclose(got[2], want[2], atol=1e-6)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-4, atol=1e-5)


def test_pooled_mean_is_the_serial_mean(dataset):
    dwi, mask, d = dataset
    C = correlation(PATCH)
    for stride in (1, 2):
        serial = denoise_mppca(dwi, mask, patch=PATCH, aggregate="mean", stride=stride,
                               workers=1)
        pooled = denoise_mppca(dwi, mask, patch=PATCH, aggregate="mean", stride=stride,
                               workers=3)
        for a, b in zip((serial.denoised, serial.sigma_hat, serial.rank),
                        (pooled.denoised, pooled.sigma_hat, pooled.rank), strict=True):
            np.testing.assert_array_equal(a, b)
    serial = denoise_whitened(dwi, mask, C, patch=PATCH, d_field=d, aggregate="mean",
                              stride=2, workers=1)
    pooled = denoise_whitened(dwi, mask, C, patch=PATCH, d_field=d, aggregate="mean",
                              stride=2, workers=3)
    for a, b in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_stride_still_covers_every_mask_voxel_and_leaves_the_outside(dataset):
    dwi, mask, _ = dataset
    got = denoise_mppca(dwi, mask, patch=PATCH, aggregate="mean", stride=3, workers=1)

    assert np.all(got.sigma_hat[mask] > 0), "every mask voxel got at least one patch"
    assert np.all(got.rank[mask] > 0)
    np.testing.assert_array_equal(got.denoised[~mask], dwi[~mask])
    assert not got.sigma_hat[~mask].any() and not got.rank[~mask].any()
    assert patch_centres(mask, PATCH, 3).size < mask.sum(), "the stride thins the centres"


def test_the_centre_default_is_untouched(dataset):
    dwi, mask, _ = dataset
    default = denoise_mppca(dwi, mask, patch=PATCH, workers=1)
    explicit = denoise_mppca(dwi, mask, patch=PATCH, aggregate="centre", workers=1)
    np.testing.assert_array_equal(default.denoised, explicit.denoised)
    assert default.rank.dtype == np.int32
    assert default.provenance["rule"].endswith("centre-voxel only")


def test_bad_aggregate_settings_are_refused(dataset):
    dwi, mask, _ = dataset
    assert check_aggregate("mean", 2) == ("mean", 2)
    with pytest.raises(ValueError, match="aggregate must be"):
        check_aggregate("median", 1)
    with pytest.raises(ValueError, match="stride must be"):
        check_aggregate("mean", 0)
    with pytest.raises(ValueError, match="needs aggregate='mean'"):
        denoise_mppca(dwi, mask, patch=PATCH, stride=2, workers=1)


def test_mean_lr_weighs_the_whole_patch_and_pools_bit_for_bit(dataset):
    """In mean mode "lr" runs with each patch's levels. It differs from "gd", and the
    pool still changes nothing."""
    dwi, mask, d = dataset
    C = correlation(PATCH)
    args = dict(patch=PATCH, d_field=d, aggregate="mean", stride=2)
    serial = denoise_whitened(dwi, mask, C, shrink="lr", workers=1, **args)
    pooled = denoise_whitened(dwi, mask, C, shrink="lr", workers=3, **args)
    for a, b in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(a, b)
    gd = denoise_whitened(dwi, mask, C, shrink="gd", workers=1, **args)
    assert not np.array_equal(serial[0], gd[0])
