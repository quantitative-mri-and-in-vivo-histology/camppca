"""One worker and one BLAS thread per test, so tests run side by side on the cores.

Each test used to start a pool on every core, and serial tests let OpenBLAS start a
thread per hardware thread for small matrices. With both at one, ``pytest -n 16`` ran
the suite in 22 s against 85 s. The pool itself is tested with explicit worker counts
in ``test_pool.py``. Set before numpy is imported, which is when OpenBLAS reads it.
"""

import os

os.environ.setdefault("CAMPPCA_WORKERS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
