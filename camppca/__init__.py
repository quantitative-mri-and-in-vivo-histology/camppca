"""camppca — Covariance-Aware Marchenko-Pastur PCA for diffusion MRI.

Estimates the spatial noise covariance of reconstructed dMRI from the data itself
and runs MP-PCA denoising in the whitened domain, where the Marchenko-Pastur
i.i.d.-noise assumption holds.
"""

__version__ = "0.1.0"
