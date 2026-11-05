"""camppca -- covariance-aware MP-PCA for diffusion MRI.

Estimates the spatial noise covariance of reconstructed dMRI from the data itself and
runs MP-PCA denoising in the whitened domain, where the Marchenko-Pastur i.i.d.-noise
assumption holds.
"""

from .noise_model import NoiseModel, brain_mask, estimate_noise_model

__version__ = "0.1.0"

__all__ = ["NoiseModel", "DenoiseResult", "brain_mask", "denoise_camppca",
           "denoise_mppca", "denoise_with_noise_model", "estimate_noise_model",
           "stabilize_phase", "__version__"]

# Lazy, so `import camppca` stays light.
_LAZY = {"denoise_camppca": "denoiser", "denoise_mppca": "denoiser",
         "denoise_with_noise_model": "denoiser", "DenoiseResult": "denoiser",
         "stabilize_phase": "phase"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
