"""The nibabel boundary. Every core function takes and returns arrays; this is the only
module that knows about files.

Kept deliberately thin: load a 4D image and its gradient table, write NIfTIs that inherit
the input's affine and header.
"""

import json

import nibabel as nib
import numpy as np

__all__ = ["load_dwi", "load_bvals", "load_mask", "save_like", "save_json"]


def load_dwi(path, dtype=np.float32):
    """Load a 4D image. Returns ``(data, affine, header)``."""
    img = nib.load(str(path))
    if img.ndim != 4:
        raise ValueError(f"{path}: expected a 4D image, got shape {img.shape}")
    return np.asanyarray(img.dataobj, dtype=dtype), img.affine, img.header


def load_bvals(path):
    """Load a whitespace-separated FSL bval file as a flat float array."""
    bvals = np.loadtxt(str(path)).ravel().astype(float)
    if bvals.ndim != 1 or bvals.size == 0:
        raise ValueError(f"{path}: expected a 1-D list of b-values")
    return bvals


def load_mask(path):
    """Load a mask image as a boolean array."""
    return np.asanyarray(nib.load(str(path)).dataobj) > 0.5


def save_like(path, data, affine, header=None, dtype=np.float32):
    """Write ``data`` as a NIfTI carrying the input's geometry.

    The header is only reused when the array still has the input's shape -- a 3-D map
    written with a 4-D header would carry a stale dim[4].
    """
    arr = np.asarray(data, dtype)
    use_header = header if (header is not None and arr.ndim == 4) else None
    nib.save(nib.Nifti1Image(arr, affine, use_header), str(path))


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
