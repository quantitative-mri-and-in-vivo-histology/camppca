"""The nibabel boundary. Every core function takes and returns arrays; this is the only
module that knows about files.

Kept deliberately thin: load a 4D image and its gradient table, write NIfTIs that inherit
the input's affine and header.
"""

import json

import nibabel as nib
import numpy as np

__all__ = ["load_dwi", "load_bvals", "load_mask", "check_same_space", "save_like",
           "save_json"]

#: How far two affines may differ, in mm, and still count as one voxel grid.
AFFINE_TOL = 1e-3


def load_dwi(path, dtype=np.float32):
    """Load a 4D image. Returns ``(data, affine, header)``.

    A complex image is read as complex64 whatever ``dtype`` says.

    A NIfTI is read one volume at a time into the result, so only the result and one
    volume are ever in memory. Reading it whole through nibabel held about three copies.
    A NIfTI whose volumes are not stored one after another is read whole.

    Only NIfTI is read: the outputs are NIfTI and reuse the input's header, which another
    format would break at the first write.
    """
    img = _load_nifti(path)
    if img.ndim != 4:
        raise ValueError(f"{path}: expected a 4D image, got shape {img.shape}")
    if np.iscomplexobj(np.empty(0, img.get_data_dtype())):
        dtype = np.complex64
    data = _read_volumes(img, dtype)
    if data is None:
        data = np.asanyarray(img.dataobj, dtype=dtype)
    return data, img.affine, img.header


def _read_volumes(img, dtype):
    """The 4D NIfTI's data, read volume by volume and scaled as nibabel scales it.

    None when the image is not a NIfTI whose volumes follow each other in the file.
    """
    proxy = img.dataobj
    if not (isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)) and nib.is_proxy(proxy)
            and getattr(proxy, "order", "F") == "F"):
        return None
    shape = img.shape
    nbytes = int(np.prod(shape[:3])) * proxy.dtype.itemsize
    scaled = proxy.slope != 1 or proxy.inter != 0
    out = np.empty(shape, dtype)
    with nib.openers.ImageOpener(proxy.file_like) as f:
        f.seek(proxy.offset)
        for i in range(shape[3]):
            buf = f.read(nbytes)
            if len(buf) != nbytes:
                return None                        # truncated: let nibabel raise
            vol = np.frombuffer(buf, proxy.dtype).reshape(shape[:3], order="F")
            # result_type keeps complex data complex: a float64 cast would drop the
            # imaginary part.
            wide = np.result_type(vol.dtype, np.float64)
            out[..., i] = vol.astype(wide) * proxy.slope + proxy.inter if scaled else vol
    return out


def load_bvals(path):
    """Load a whitespace-separated FSL bval file as a flat float array."""
    bvals = np.loadtxt(str(path)).ravel().astype(float)
    if bvals.size == 0:
        raise ValueError(f"{path}: expected a 1-D list of b-values")
    return bvals


def load_mask(path, affine=None):
    """Load a mask image as a boolean array. A trailing dimension of 1 is dropped, as
    some tools write masks with shape (X, Y, Z, 1).

    With ``affine``, the data's, the mask must lie on the same voxel grid; see
    :func:`check_same_space`.
    """
    img = _load_nifti(path)
    if affine is not None:
        check_same_space(img.affine, affine, path, "the input")
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    return data > 0.5


def check_same_space(affine, reference, path, reference_name):
    """Raise unless two images lie on the same voxel grid.

    Comparing shapes is not enough: a mask of the right shape with a flipped axis would
    place the noise model on a mirrored brain without an error.
    """
    diff = float(np.abs(np.asarray(affine) - np.asarray(reference)).max())
    if diff > AFFINE_TOL:
        raise ValueError(f"{path}: its affine differs from {reference_name}'s by up to "
                         f"{diff:.3g}, so its voxels do not line up. Resample it to the "
                         f"same grid first.")


def _load_nifti(path):
    img = nib.load(str(path))
    if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(f"{path}: expected a NIfTI file (.nii or .nii.gz), got "
                         f"{type(img).__name__}")
    return img


def save_like(path, data, affine, header=None, dtype=np.float32):
    """Write ``data`` as a NIfTI carrying the input's geometry.

    The header is only reused when the array still has the input's shape -- a 3-D map
    written with a 4-D header would carry a stale dim[4]. A 3-D map still takes the
    input's qform and sform codes and units, so tools read it in the same space. The file
    is written as ``dtype``. A reused header otherwise keeps the input's type, and nibabel
    packed a float image from a uint16 input back into scaled 16-bit integers.
    """
    arr = np.asarray(data, dtype)
    use_header = header if (header is not None and arr.ndim == 4) else None
    img = nib.Nifti1Image(arr, affine, use_header)
    if header is not None and use_header is None:
        img.set_qform(affine, int(header["qform_code"]))
        img.set_sform(affine, int(header["sform_code"]))
        img.header.set_xyzt_units(xyz=header.get_xyzt_units()[0])
    img.set_data_dtype(arr.dtype)
    nib.save(img, str(path))


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_jsonable)
        fh.write("\n")


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    return str(o)
