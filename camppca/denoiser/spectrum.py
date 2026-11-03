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

from functools import partial

import numpy as np
import scipy.linalg

__all__ = ["eigh_descending", "eigvalsh_descending"]

_eigh = partial(scipy.linalg.eigh, driver="evd", check_finite=False)
_eigvalsh = partial(scipy.linalg.eigh, eigvals_only=True, driver="evd", check_finite=False)


def eigh_descending(gram):
    """(B, n, n) symmetric -> ``(evals, evecs)``: (B, n) descending, floored at 0, and
    (B, n, n) with eigenvectors as columns in the same order."""
    pairs = [_eigh(g) for g in gram]
    evals = np.stack([w for w, _ in pairs])[:, ::-1]
    evecs = np.stack([v for _, v in pairs])[:, :, ::-1]
    return np.maximum(evals, 0.0), evecs


def eigvalsh_descending(gram):
    """(B, n, n) symmetric -> (B, n) eigenvalues, descending, floored at 0."""
    return np.maximum(np.stack([_eigvalsh(g) for g in gram])[:, ::-1], 0.0)
