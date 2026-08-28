"""camppca -- Covariance-Aware Marchenko-Pastur PCA for diffusion MRI.

Estimates the spatial noise covariance of reconstructed dMRI from the data itself and
runs MP-PCA denoising in the whitened domain, where the Marchenko-Pastur i.i.d.-noise
assumption holds.
"""

import os

# MUST precede any `import jax` anywhere in the package: JAX reads this at first import
# and otherwise preallocates ~75% of the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from .noise_model import NoiseModel, brain_mask, estimate_noise_model  # noqa: E402

__version__ = "0.1.0"

__all__ = ["NoiseModel", "DenoiseResult", "brain_mask", "denoise_camppca",
           "denoise_mppca", "denoise_with_noise_model", "estimate_noise_model",
           "__version__"]

# Lazy: importing a denoiser pulls in JAX. `estimate_noise_model` and `brain_mask` stay
# JAX-free, and staying that way is asserted in the test suite.
_LAZY = {"denoise_camppca": "denoiser", "denoise_mppca": "denoiser",
         "denoise_with_noise_model": "denoiser", "DenoiseResult": "denoiser"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
