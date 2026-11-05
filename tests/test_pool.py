"""The worker pool: chunks cover the voxels once, and pooling changes nothing.

A pooled pass is the serial pass, bit for bit: chunks hold whole tiles, and every patch
is gathered from the whole volume, its box moved inward at the image edge.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from camppca import NoiseModel, denoise_mppca, denoise_with_noise_model
from camppca.denoiser import _effective_d
from camppca.denoiser.camppca import whitened_level
from camppca.noise_model import _build_C, loop
from camppca.pool import default_workers, resolve_workers, voxel_chunks

SHAPE = (24, 20, 12)
N_VOLS = 60
PATCH = (5, 5, 3)
LAG = 4


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(SHAPE + (3,)) @ rng.standard_normal((3, N_VOLS))
    dwi = (signal * 20 + 300 + rng.standard_normal(SHAPE + (N_VOLS,)) * 8).astype(np.float32)
    mask = np.zeros(SHAPE, bool)
    mask[3:21, 3:17, 2:10] = True
    return dwi, mask


@pytest.fixture(scope="module")
def model(dataset):
    """A correlated kernel and a level field that varies, so both whitening steps act."""
    _, mask = dataset
    rho = np.zeros((2 * LAG + 1, 2 * LAG + 1, PATCH[2]))
    for i in range(-LAG, LAG + 1):
        for j in range(-LAG, LAG + 1):
            rho[i + LAG, j + LAG, 0] = 0.4 ** abs(i) * 0.2 ** abs(j)
    d = np.where(mask, np.linspace(1.0, 2.0, SHAPE[0])[:, None, None], 0).astype(np.float32)
    return NoiseModel(rho=rho, d=d, mask=mask, lag=LAG)


def test_chunks_cover_every_mask_voxel_once_in_whole_tiles():
    mask = np.zeros((30, 6, 6), bool)
    mask[2:12] = True
    mask[16:28, :3] = True

    voxels = np.flatnonzero(mask)

    chunks = voxel_chunks(voxels, 4, multiple=64)

    assert np.array_equal(np.concatenate(chunks), voxels), "once, in order"
    assert all(c.size % 64 == 0 for c in chunks[:-1]), "full tiles but the last"
    assert 2 <= len(chunks) <= 4, "rounding up to whole tiles only ever merges chunks"
    assert voxel_chunks(np.array([], int), 4, 64) == []


def test_worker_count_is_checked():
    assert default_workers() >= 1
    assert resolve_workers(None) == default_workers()
    with pytest.raises(ValueError, match="workers must be"):
        resolve_workers(0)


def test_pooled_mppca_is_the_serial_pass(dataset):
    dwi, mask = dataset
    serial = denoise_mppca(dwi, mask, patch=PATCH, workers=1)
    for workers in (2, 3):
        pooled = denoise_mppca(dwi, mask, patch=PATCH, workers=workers)
        np.testing.assert_array_equal(pooled.denoised, serial.denoised)
        np.testing.assert_array_equal(pooled.sigma_hat, serial.sigma_hat)
        np.testing.assert_array_equal(pooled.rank, serial.rank)


def test_pooled_camppca_is_the_serial_pass(dataset, model):
    dwi, _ = dataset
    serial = denoise_with_noise_model(dwi, model, patch=PATCH, workers=1)
    pooled = denoise_with_noise_model(dwi, model, patch=PATCH, workers=3)
    np.testing.assert_array_equal(pooled.denoised, serial.denoised)
    np.testing.assert_array_equal(pooled.sigma_hat, serial.sigma_hat)
    np.testing.assert_array_equal(pooled.rank, serial.rank)


@pytest.mark.parametrize("white", [True, False], ids=["standard", "whitened"])
def test_pooled_level_pass_is_the_serial_pass(dataset, model, white):
    dwi, mask = dataset
    C = None if white else model.patch_correlation(PATCH)
    d_eff = None if white else _effective_d(model.d, mask)
    serial = whitened_level(dwi, mask, C, PATCH, d_field=d_eff, workers=1)
    pooled = whitened_level(dwi, mask, C, PATCH, d_field=d_eff, workers=3)
    np.testing.assert_array_equal(pooled, serial)


def test_the_pooled_loop_is_the_serial_loop(dataset, model):
    """Bit for bit: the sampled patches go in fixed chunks, pooled in chunk order. When
    the chunks followed the worker count, a near-tied rank moved the kernel by 1e-4."""
    dwi, mask = dataset
    white = np.zeros_like(model.rho)
    white[LAG, LAG, 0] = 1.0
    serial, _ = loop.loop_kernel(dwi, mask, model.d, white, np.array(PATCH), LAG, 0,
                                 workers=1)
    for workers in (2, 3):
        pooled, info = loop.loop_kernel(dwi, mask, model.d, white, np.array(PATCH), LAG, 0,
                                        workers=workers)
        np.testing.assert_array_equal(pooled, serial)
    assert info["converged"]
    assert np.linalg.eigvalsh(_build_C(pooled, np.array(PATCH), LAG))[0] > 1e-3


def test_a_script_without_a_main_guard_runs(tmp_path):
    """The workers do not re-run the caller's script."""
    script = tmp_path / "unguarded.py"
    script.write_text(textwrap.dedent("""
        import numpy as np
        from camppca import denoise_mppca
        dwi = np.random.default_rng(0).standard_normal((12, 10, 8, 40)).astype(np.float32)
        result = denoise_mppca(dwi, patch=(3, 3, 3), workers=2)
        print("done", int(np.isfinite(result.denoised).all()))
    """))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["done", "1"]


def test_the_default_worker_count_respects_a_cpu_quota(monkeypatch):
    from camppca import pool
    monkeypatch.setattr(pool.joblib, "cpu_count", lambda: 3)
    assert pool.default_workers() == min(3, pool._physical_cores())
