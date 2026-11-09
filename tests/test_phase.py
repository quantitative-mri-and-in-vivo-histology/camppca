"""Phase stabilization, against complex data built from a known magnitude and phase."""

import nibabel as nib
import numpy as np
import pytest

from camppca import stabilize_phase as public
from camppca.cli import main
from camppca.phase import _dk_weights, phase_scale, reference_volume, stabilize_phase

SHAPE = (48, 40, 3, 8)


def blob(shape=SHAPE[:3]):
    """A smooth positive magnitude that falls to zero well inside the image."""
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]),
                       indexing="ij")
    return (100 * np.exp(-(x ** 2 + y ** 2) / 0.15))[:, :, None] * np.ones(shape)


def phases(rng, shape=SHAPE):
    """A rough fixed phase, the same in every volume, plus a smooth one per volume, wrapped
    to radians."""
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    fixed = rng.uniform(-np.pi, np.pi, shape[:3])
    smooth = np.stack([(a * x + b * y)[:, :, None] * np.ones(shape[2])
                       for a, b in rng.uniform(-0.05, 0.05, (shape[3], 2))], axis=-1)
    return np.angle(np.exp(1j * (fixed[..., None] + smooth)))


def test_phase_scale_takes_the_declared_turn():
    siemens = np.array([-4096.0, 0.0, np.nan, 4094.0])
    assert phase_scale(siemens, (-4096, 4096)) == pytest.approx(np.pi / 4096)
    assert phase_scale(np.array([0.0, 4095.0]), (0, 4096)) == pytest.approx(np.pi / 2048)
    assert phase_scale(np.array([-180.0, 179.0]), (-180, 180)) == pytest.approx(np.pi / 180)


def test_phase_scale_takes_radians_by_default():
    assert phase_scale(np.array([-3.1, 3.14])) == 1.0
    assert phase_scale(np.array([0.0, 6.28])) == 1.0, "only the width of the turn matters"


def test_phase_scale_refuses_a_phase_wider_than_the_turn():
    """Siemens integers read as radians, and radians that were unwrapped, were scaled by
    their range until 2026-11-09: [-4, 4] rad came out at 0.785 of the truth."""
    with pytest.raises(ValueError, match="--phase-range"):
        phase_scale(np.array([-4096.0, 4094.0]))
    with pytest.raises(ValueError, match="wider than the declared turn"):
        phase_scale(np.array([-4.0, 4.0]))
    with pytest.raises(ValueError, match="no finite"), pytest.warns(RuntimeWarning):
        phase_scale(np.full(4, np.nan))
    with pytest.raises(ValueError, match="low to high"):
        phase_scale(np.array([0.0, 1.0]), (180, -180))


def test_phase_scale_warns_on_less_than_half_a_turn():
    with pytest.warns(RuntimeWarning, match="less than half"):
        assert phase_scale(np.array([-0.5, 0.5])) == 1.0
    with pytest.warns(RuntimeWarning, match="less than half"):
        phase_scale(np.array([-3.1, 3.1]), (-4096, 4096))


def test_stabilize_phase_reads_degrees_like_radians():
    rng = np.random.default_rng(4)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    phase = np.angle(np.exp(1j * phases(rng)))
    phase[0, 0, 0, 0], phase[0, 0, 0, 1] = -np.pi, np.pi      # the full turn, as noise gives
    radians = stabilize_phase(magnitude, phase)
    degrees = stabilize_phase(magnitude, np.degrees(phase), phase_range=(-180, 180))
    np.testing.assert_allclose(degrees, radians, atol=1e-3 * magnitude.max())


def test_the_dk_filter_is_symmetric_and_avoids_the_center():
    k = _dk_weights()
    assert k.sum() == pytest.approx(1)
    np.testing.assert_array_equal(k, k.T)
    np.testing.assert_array_equal(k, k[::-1, ::-1])
    assert k[2, 2] < 0.01 and k[1, 2] < 0.01 and k[1, 1] == k.max()


def test_without_noise_the_real_part_is_the_magnitude():
    """The rough fixed phase goes with the reference volume's phase."""
    rng = np.random.default_rng(0)
    magnitude = blob()[..., None] * np.ones(SHAPE[3])
    real, imag = stabilize_phase(magnitude, phases(rng), imag=True)
    inside = magnitude > 20
    np.testing.assert_allclose(real[inside], magnitude[inside], rtol=2e-3)
    assert np.abs(imag[inside] / magnitude[inside]).max() < 0.1   # under 0.1 rad off


def test_the_reference_is_the_first_bright_volume():
    magnitude = np.ones((4, 4, 2, 5)) * np.array([50, 60, 100, 97, 100.0])
    assert reference_volume(magnitude) == 2
    magnitude[0, 0, 0, 1] = np.nan
    assert reference_volume(magnitude) == 2


def test_a_nan_leaves_its_neighbours_turned():
    """One NaN spread through the smoothing and left its 5 x 5 neighbours unturned."""
    magnitude = np.full((16, 16, 3, 4), 100.0)
    phase = np.zeros(magnitude.shape)
    phase[..., 1:] = 0.8                       # a constant phase on every volume but one
    phase[3, 3, 1, 0] = np.nan                 # in the reference volume
    phase[0, 0, 0, 1:3] = -np.pi, np.pi        # the full turn, as noise gives
    magnitude[8, 8, 1, 2] = np.nan
    real = stabilize_phase(magnitude, phase)
    # The reference phase's NaN makes that voxel NaN in every volume, and nothing else.
    assert np.isnan(real[8, 8, 1, 2]) and np.isnan(real[3, 3, 1]).all()
    finite = np.isfinite(real)
    assert finite.sum() == real.size - 1 - magnitude.shape[3]
    finite[:3, :3, 0] = False                  # the corner holding the full turn
    np.testing.assert_allclose(real[finite], 100.0, rtol=1e-5)


def test_the_real_part_carries_no_noise_floor():
    """Pure complex noise: the magnitude sits at 1.25 sigma, the real part near 0. The
    reference volume keeps its magnitude."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((64, 64, 4, 12)) + 1j * rng.standard_normal((64, 64, 4, 12))
    real = stabilize_phase(np.abs(z), np.angle(z))
    ref = reference_volume(np.abs(z))
    assert np.abs(z).mean() == pytest.approx(1.25, abs=0.02)
    np.testing.assert_allclose(real[..., ref], np.abs(z[..., ref]), rtol=1e-4, atol=1e-5)
    assert abs(np.delete(real, ref, axis=3).mean()) < 0.03


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

    args = ["phase", str(tmp_path / "mag.nii.gz"), str(tmp_path / "phase.nii.gz"),
            str(tmp_path / "real.nii.gz"), "--imag", str(tmp_path / "imag.nii.gz"), "-q"]
    with pytest.raises(ValueError, match="--phase-range"):
        main(args)
    assert not (tmp_path / "real.nii.gz").exists(), "refused before any work"
    rc = main(args + ["--phase-range", "-4096", "4096"])
    assert rc == 0
    real = np.asanyarray(nib.load(tmp_path / "real.nii.gz").dataobj)
    assert real.shape == SHAPE and real.dtype == np.float32
    inside = magnitude > 20
    np.testing.assert_allclose(real[inside], magnitude[inside], rtol=5e-3)
    assert (tmp_path / "imag.nii.gz").exists()


def test_the_pooled_result_is_the_serial_one():
    rng = np.random.default_rng(5)
    magnitude = blob()[..., None] * np.ones(SHAPE[3]) + rng.random(SHAPE)
    phase = phases(rng)
    serial = stabilize_phase(magnitude, phase, imag=True, workers=1)
    pooled = stabilize_phase(magnitude, phase, imag=True, workers=3)
    for a, b in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(a, b)
