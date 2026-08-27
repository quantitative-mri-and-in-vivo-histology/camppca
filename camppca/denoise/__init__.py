"""Denoising: the engine half of the seam.

Source-agnostic by design -- it consumes a noise model and never asks where the model
came from, so a future alignment-based estimator feeds the same code.

``camppca.py`` is the method; ``mppca.py`` is the plain white-noise baseline arm plus the
Marchenko-Pastur threshold both share.

The public entries (``denoise``, ``denoise_with_noise_model``, ``DenoiseResult``) land
here next.
"""
