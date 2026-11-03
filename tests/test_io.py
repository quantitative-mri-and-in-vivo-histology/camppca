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


def test_a_3d_image_is_refused(tmp_path):
    path = tmp_path / "vol.nii"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), path)
    with pytest.raises(ValueError, match="expected a 4D image"):
        io.load_dwi(path)
