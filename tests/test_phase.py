"""Phase stabilization, against complex data built from a known magnitude and phase."""

import nibabel as nib
import numpy as np
import pytest

from camppca import stabilize_phase as public
from camppca.cli import main
from camppca.phase import _dk_weights, _tukey, phase_scale, stabilize_phase

SHAPE = (48, 40, 3, 8)


def blob(shape=SHAPE[:3]):
    """A smooth positive magnitude that falls to zero well inside the image."""
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]),
                       indexing="ij")
    return (100 * np.exp(-(x ** 2 + y ** 2) / 0.15))[:, :, None] * np.ones(shape)


def phases(rng, shape=SHAPE):
    """A rough fixed phase, the same in every volume, plus a smooth one per volume."""
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    fixed = rng.uniform(-np.pi, np.pi, shape[:3])
    smooth = np.stack([(a * x + b * y)[:, :, None] * np.ones(shape[2])
                       for a, b in rng.uniform(-0.05, 0.05, (shape[3], 2))], axis=-1)
    return fixed[..., None] + smooth


def test_phase_scale_takes_the_range_as_one_turn():
    assert phase_scale(np.array([-4096.0, 0.0, 4094.0])) == pytest.approx(np.pi / 4096,
                                                                           rel=1e-3)
    assert phase_scale(np.array([0.0, 4095.0])) == pytest.approx(2 * np.pi / 4096, rel=1e-3)
    assert phase_scale(np.array([-180.0, 179.0])) == pytest.approx(np.pi / 180, rel=1e-2)
    assert phase_scale(np.array([-3142.0, 3141.0])) == pytest.approx(1e-3, rel=1e-3)
    assert phase_scale(np.array([-4096.0, np.nan, 4094.0])) == pytest.approx(np.pi / 4096,
                                                                             rel=1e-3)


def test_phase_scale_refuses_a_constant_or_empty_phase():
    with pytest.raises(ValueError, match="phase unit"):
        phase_scale(np.full(4, 4000.0))
    with pytest.raises(ValueError, match="phase unit"), pytest.warns(RuntimeWarning):
        phase_scale(np.full(4, np.nan))


def test_phase_scale_keeps_radians():
    assert phase_scale(np.array([-3.1, 3.14])) == 1.0
    assert phase_scale(np.array([0.0, 6.28])) == 1.0
    assert phase_scale(np.array([-0.5, 0.5])) == 1.0, "less than a turn is not stretched"


def test_stabilize_phase_reads_degrees_like_radians():
    rng = np.random.default_rng(4)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    phase = np.angle(np.exp(1j * phases(rng)))
    phase[0, 0, 0, 0], phase[0, 0, 0, 1] = -np.pi, np.pi      # the full turn, as noise gives
    radians = stabilize_phase(magnitude, phase)
    degrees = stabilize_phase(magnitude, np.degrees(phase))
    np.testing.assert_allclose(degrees, radians, atol=1e-3 * magnitude.max())


def test_the_tukey_window_has_its_half_maximum_at_fwhm():
    n, fwhm = 200, 0.58
    r, w = 2 * np.abs(np.fft.fftfreq(n)), _tukey(n, fwhm)
    assert w[0] == 1.0 and w[r == 1].item() == pytest.approx(0, abs=1e-12)
    order = np.argsort(r)
    assert np.interp(fwhm, r[order], w[order]) == pytest.approx(0.5, abs=1e-3)


def test_the_dk_filter_is_symmetric_and_avoids_the_center():
    k = _dk_weights()
    assert k.sum() == pytest.approx(1)
    np.testing.assert_array_equal(k, k.T)
    np.testing.assert_array_equal(k, k[::-1, ::-1])
    assert k[2, 2] < 0.01 and k[1, 2] < 0.01 and k[1, 1] == k.max()


@pytest.mark.parametrize("filt", ["dk", "tukey"])
def test_without_noise_the_real_part_is_the_magnitude(filt):
    rng = np.random.default_rng(0)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    real, imag = stabilize_phase(magnitude, phases(rng), filter=filt, mean_phase=True, imag=True)
    inside = magnitude > 20
    np.testing.assert_allclose(real[inside], magnitude[inside], rtol=2e-3)
    assert np.abs(imag[inside] / magnitude[inside]).max() < 0.1   # under 0.1 rad off


def test_the_rough_fixed_phase_needs_the_mean_phase_step():
    rng = np.random.default_rng(0)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    real = stabilize_phase(magnitude, phases(rng), mean_phase=False)
    inside = magnitude > 20
    assert np.median(np.abs(real[inside] - magnitude[inside]) / magnitude[inside]) > 0.1


def test_the_real_part_carries_no_noise_floor():
    """Pure complex noise: the magnitude sits at 1.25 sigma, the real part near 0 with
    the decorrelated kernel."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((64, 64, 4, 12)) + 1j * rng.standard_normal((64, 64, 4, 12))
    real = stabilize_phase(np.abs(z), np.angle(z))
    assert np.abs(z).mean() == pytest.approx(1.25, abs=0.02)
    assert real.mean() < 0.35


def test_the_slice_axis_picks_the_plane():
    rng = np.random.default_rng(2)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    phase = phases(rng)
    want = stabilize_phase(magnitude, phase)
    got = stabilize_phase(magnitude.transpose(2, 0, 1, 3), phase.transpose(2, 0, 1, 3),
                          slice_axis=0)
    np.testing.assert_allclose(got.transpose(1, 2, 0, 3), want, rtol=1e-5, atol=1e-3)


def test_bad_settings_are_refused():
    assert public is stabilize_phase
    m = np.ones(SHAPE)
    with pytest.raises(ValueError, match="filter must be"):
        stabilize_phase(m, m, filter="hann")
    with pytest.raises(ValueError, match="fwhm must be"):
        stabilize_phase(m, m, filter="tukey", fwhm=0.4)
    with pytest.raises(ValueError, match="alike"):
        stabilize_phase(m, m[..., :2])


def test_cli_writes_real_and_imaginary_parts(tmp_path):
    """Phase stored as Siemens does, uint16 scaled to -4096..4094."""
    rng = np.random.default_rng(3)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    phase = np.angle(np.exp(1j * phases(rng)))
    stored = np.round((phase / np.pi * 4096 + 4096) / 2).astype(np.uint16)
    img = nib.Nifti1Image(stored, np.eye(4))
    img.header.set_slope_inter(2.0, -4096.0)
    nib.save(img, tmp_path / "phase.nii.gz")
    nib.save(nib.Nifti1Image(magnitude.astype(np.float32), np.eye(4)), tmp_path / "mag.nii.gz")

    rc = main(["phase", str(tmp_path / "mag.nii.gz"), str(tmp_path / "phase.nii.gz"),
               str(tmp_path / "real.nii.gz"), "--imag", str(tmp_path / "imag.nii.gz"), "--mean-phase", "-q"])
    assert rc == 0
    real = np.asanyarray(nib.load(tmp_path / "real.nii.gz").dataobj)
    assert real.shape == SHAPE and real.dtype == np.float32
    inside = magnitude > 20
    np.testing.assert_allclose(real[inside], magnitude[inside], rtol=5e-3)
    assert (tmp_path / "imag.nii.gz").exists()
