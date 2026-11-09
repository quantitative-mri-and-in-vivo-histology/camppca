"""Eigendecomposition of a batch of Gram matrices, eigenvalues descending.

Each matrix goes to LAPACK on its own through ``scipy.linalg.eigh``, which asks LAPACK
for its best workspace first. The raw ``scipy.linalg.lapack`` wrappers default to the
smallest workspace, which forces the unblocked reduction: 1.55 ms against 1.2 ms for the
eigenvalues of one 297 x 297 matrix on one core. ``np.linalg.eigh`` took 3.3 ms for the
same matrix with vectors, against 1.55 ms here.

Matrices are small and many, and one LAPACK call per matrix keeps every core of a pool
worker busy on its own matrix. OpenBLAS cannot be threaded inside one process for this:
its calls from several threads wait on a shared lock (see :mod:`camppca.pool`).
"""

import ctypes
from functools import partial

import numpy as np
import scipy.linalg
import scipy.linalg.cython_lapack
from scipy.linalg import lapack

__all__ = ["eigh_descending", "eigvalsh_descending", "tridiagonal_eigvalsh", "top_eigvecs"]

_eigh = partial(scipy.linalg.eigh, driver="evd", check_finite=False)
_eigvalsh = partial(scipy.linalg.eigh, eigvals_only=True, driver="evd", check_finite=False)


def eigh_descending(gram):
    """(B, n, n) symmetric -> ``(evals, evecs)``: (B, n) descending, floored at 0, and
    (B, n, n) with eigenvectors as columns in the same order."""
    pairs = [_eigh(g) for g in gram]
    evals = np.stack([w for w, _ in pairs])[:, ::-1]
    evecs = np.stack([v for _, v in pairs])[:, :, ::-1]
    return _checked(np.maximum(evals, 0.0)), evecs


def eigvalsh_descending(gram):
    """(B, n, n) symmetric -> (B, n) eigenvalues, descending, floored at 0."""
    return _checked(np.maximum(np.stack([_eigvalsh(g) for g in gram])[:, ::-1], 0.0))


def _checked(evals, failed=False):
    """``evals``, unless LAPACK failed or one is not finite. The input guard
    (:func:`~camppca.denoiser.gather.check_range`) should make this unreachable; it is
    the backstop, since ``eigh`` runs with ``check_finite=False``."""
    if failed or not np.isfinite(evals).all():
        raise ValueError("a patch's eigendecomposition failed or is not finite: the "
                         "data hold values too large for float32")
    return evals


# The kept route. The MP cut needs every eigenvalue but the rebuild only the top P
# vectors, P often near 10 of 199. So each Gram is reduced to tridiagonal form once
# (sytrd), its eigenvalues come from that (sterf), and after the cut its top P vectors
# come by inverse iteration (stein), mapped back to the Gram by ormtr. scipy has no
# wrapper for ormtr, so it is called through ctypes from scipy's own LAPACK. EDDEN 0.9 mm
# at 7^3 and 199 volumes: 0.60 ms per voxel against 0.98 ms, the ranks equal and the
# center row within 7e-4 relative (2026-11-08). Real data only.

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]

_ORMTR = {}


def _ormtr(prefix):
    """``?ormtr`` from scipy's LAPACK, by its Cython capsule."""
    if prefix not in _ORMTR:
        capsule = scipy.linalg.cython_lapack.__pyx_capi__[prefix + "ormtr"]
        pointer = ctypes.pythonapi.PyCapsule_GetPointer(
            capsule, ctypes.pythonapi.PyCapsule_GetName(capsule))
        real = ctypes.POINTER(ctypes.c_float if prefix == "s" else ctypes.c_double)
        integer, char = ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        _ORMTR[prefix] = ctypes.CFUNCTYPE(
            None, char, char, char, integer, integer, real, integer, real, real, integer,
            real, integer, integer)(pointer), real
    return _ORMTR[prefix]


def _back_transform(prefix, a, tau, z):
    """``Q z``, with ``Q`` from sytrd's upper reflectors in ``a``. ``z`` is overwritten."""
    fn, real = _ormtr(prefix)
    n, m = z.shape
    ni, mi, info = ctypes.c_int(n), ctypes.c_int(m), ctypes.c_int()
    work = np.empty(1, z.dtype)
    lwork = ctypes.c_int(-1)
    args = (b"L", b"U", b"N", ctypes.byref(ni), ctypes.byref(mi), a.ctypes.data_as(real),
            ctypes.byref(ni), tau.ctypes.data_as(real), z.ctypes.data_as(real),
            ctypes.byref(ni))
    fn(*args, work.ctypes.data_as(real), ctypes.byref(lwork), ctypes.byref(info))
    work = np.empty(int(work[0]), z.dtype)
    lwork = ctypes.c_int(work.size)
    fn(*args, work.ctypes.data_as(real), ctypes.byref(lwork), ctypes.byref(info))
    return info.value


def tridiagonal_eigvalsh(gram):
    """(B, n, n) real symmetric -> ``(evals, reduced)``: (B, n) eigenvalues, descending and
    floored at 0, and each matrix's tridiagonal reduction for :func:`top_eigvecs`."""
    prefix = "s" if gram.dtype == np.float32 else "d"
    sytrd = getattr(lapack, prefix + "sytrd")
    sterf = getattr(lapack, prefix + "sterf")
    lwork = int(getattr(lapack, prefix + "sytrd_lwork")(gram.shape[1])[0])
    out = [sytrd(g, lwork=lwork) for g in gram]
    values = [sterf(d, e) for _, d, e, _, _ in out]
    failed = any(r[4] for r in out) or any(info for _, info in values)
    evals = _checked(np.maximum(np.stack([w for w, _ in values])[:, ::-1], 0.0), failed)
    return evals, [r[:4] for r in out]


def top_eigvecs(gram, evals, reduced, counts):
    """The top ``counts[b]`` eigenvectors of each Gram, from :func:`tridiagonal_eigvalsh`.

    Returns:
        (B, n, n), columns descending as ``evals``, zero beyond each matrix's count. A
        matrix that keeps more than a quarter of its components, whose tridiagonal
        splits, or where inverse iteration does not converge, gets all its vectors from
        :func:`eigh_descending` instead. Inverse iteration loses accuracy in a large
        cluster of equal eigenvalues, such as the zeros of noiseless data, which keeps
        everything. On EDDEN 0.9 mm about 1% of voxels keep more than a quarter.
    """
    prefix = "s" if gram.dtype == np.float32 else "d"
    stein = getattr(lapack, prefix + "stein")
    n = gram.shape[1]
    evecs = np.zeros(gram.shape, gram.dtype)
    block, split = np.ones(n, np.int32), np.full(n, n, np.int32)
    for b, (a, d, e, tau) in enumerate(reduced):
        p = int(counts[b])
        if p == 0:
            continue
        if p <= n // 4 and e.all():
            z, info = stein(d, e, np.ascontiguousarray(evals[b, :p][::-1]), block, split)
        else:
            info = 1
        if info == 0:
            z = np.asfortranarray(z, gram.dtype)
            info = _back_transform(prefix, a, tau, z)
        if info == 0:
            evecs[b, :, :p] = z[:, ::-1]
        else:
            evecs[b] = eigh_descending(gram[b:b + 1])[1][0]
    return evecs
