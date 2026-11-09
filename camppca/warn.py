"""Warnings that point at the caller's own line.

A fixed ``stacklevel`` breaks whenever the call depth changes. Every public entry point
runs under :func:`camppca.pool.limit_blas`, whose wrapper adds a frame, and the same
helper is reached from several depths. So the level is counted here: the first frame
outside the package.
"""

import sys
import warnings
from pathlib import Path

__all__ = ["warn"]

_PACKAGE = str(Path(__file__).resolve().parent)


def warn(message, category=RuntimeWarning):
    """``warnings.warn`` at the first frame outside camppca."""
    frame, level = sys._getframe(0), 1
    while frame is not None and str(Path(frame.f_code.co_filename).resolve()).startswith(
            _PACKAGE):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)
