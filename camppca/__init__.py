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

__all__ = ["NoiseModel", "estimate_noise_model", "brain_mask", "__version__"]
