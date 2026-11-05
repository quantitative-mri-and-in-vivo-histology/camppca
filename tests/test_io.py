"""The volume-by-volume NIfTI reader gives exactly what nibabel gives."""

import nibabel as nib
import numpy as np
import pytest

from camppca import io


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
@pytest.mark.parametrize("scaled", [False, True])
def test_load_dwi_matches_nibabel(tmp_path, suffix, scaled):
    rng = np.random.default_rng(0)
    data = rng.integers(-2000, 30000, size=(6, 5, 4, 7)).astype(np.int16)
    img = nib.Nifti1Image(data, np.diag([1.5, 1.5, 2.0, 1.0]))
    if scaled:
        img.header.set_slope_inter(0.37, 12.5)
    path = tmp_path / f"dwi{suffix}"
    nib.save(img, path)

    loaded, affine, _ = io.load_dwi(path)
    want = np.asanyarray(nib.load(path).dataobj, dtype=np.float32)

    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, want)
    np.testing.assert_array_equal(affine, nib.load(path).affine)


@pytest.mark.parametrize("scaled", [False, True])
def test_a_complex_image_keeps_its_imaginary_part(tmp_path, scaled):
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((6, 5, 4, 7))
            + 1j * rng.standard_normal((6, 5, 4, 7))).astype(np.complex64)
    img = nib.Nifti1Image(data, np.eye(4))
    if scaled:
        img.header.set_slope_inter(2.0, 0.0)
    path = tmp_path / "dwi.nii"
    nib.save(img, path)

    loaded, _, _ = io.load_dwi(path)

    assert loaded.dtype == np.complex64
    np.testing.assert_allclose(loaded, np.asanyarray(nib.load(path).dataobj), rtol=1e-6)
    assert np.abs(loaded.imag).max() > 1


def test_a_3d_image_is_refused(tmp_path):
    path = tmp_path / "vol.nii"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), path)
    with pytest.raises(ValueError, match="expected a 4D image"):
        io.load_dwi(path)


def test_save_like_writes_floats_under_an_integer_header(tmp_path):
    """A uint16 input header must not pack the float output back into 16 bits."""
    header = nib.Nifti1Header()
    header.set_data_dtype(np.uint16)
    data = np.linspace(-3.3, 7.7, 24, dtype=np.float32).reshape(2, 2, 2, 3)
    io.save_like(tmp_path / "out.nii", data, np.eye(4), header)

    saved = nib.load(tmp_path / "out.nii")
    assert saved.get_data_dtype() == np.float32
    np.testing.assert_array_equal(np.asanyarray(saved.dataobj), data)


def test_a_3d_map_keeps_the_input_space_codes_and_units(tmp_path):
    img = nib.Nifti1Image(np.zeros((4, 5, 3, 2), np.float32), np.diag([2.0, 2.0, 2.5, 1.0]))
    img.header.set_qform(img.affine, 1)
    img.header.set_sform(img.affine, 1)
    img.header.set_xyzt_units("mm", "sec")
    io.save_like(tmp_path / "map.nii", np.ones((4, 5, 3)), img.affine, img.header)

    back = nib.load(tmp_path / "map.nii").header
    assert (int(back["qform_code"]), int(back["sform_code"])) == (1, 1)
    assert back.get_xyzt_units()[0] == "mm"
    assert back.get_data_shape() == (4, 5, 3)


def test_a_mask_with_a_trailing_dimension_of_one_loads_as_3d(tmp_path):
    mask = np.zeros((4, 5, 3, 1), np.uint8)
    mask[1, 2, 1, 0] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii")
    loaded = io.load_mask(tmp_path / "mask.nii")
    assert loaded.shape == (4, 5, 3) and loaded.sum() == 1
