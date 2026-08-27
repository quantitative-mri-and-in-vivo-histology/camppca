# camppca

**Covariance-Aware Marchenko–Pastur PCA** — MP-PCA denoising for diffusion MRI that
accounts for spatially correlated noise.

Standard MP-PCA assumes white noise. Real EPI reconstruction (GRAPPA/SENSE, partial
Fourier, apodisation) correlates noise across neighbouring voxels, so MP-PCA
under-estimates the noise level and under-denoises. camppca estimates the spatial
noise covariance from the data itself — no noise-only scan needed — and denoises in
the whitened domain, where the Marchenko–Pastur assumption holds.

> **Not yet implemented.** This repository is currently scaffolding; the API below
> is the intended surface.

## Install

```bash
pip install -e .
```

Python ≥ 3.10. The engine is JAX and runs on CPU out of the box; for GPU, install
`jax[cuda12]`.

## Usage

```python
from camppca import denoise

den, sigma, rank = denoise(dwi, mask, bvals)
```

`dwi` is a 4D array `(X, Y, Z, N)`, `mask` a 3D boolean brain mask, `bvals` the
b-value of each volume. Returns the denoised 4D array, a per-voxel noise σ map, and
the per-voxel retained rank.

Covariance estimation and denoising are separate, so you can inspect or supply the
covariance yourself:

```python
from camppca import p2s_estimate_kernel, denoise_with_R

Rs, group = p2s_estimate_kernel(dwi, mask, bvals, kernel=(5, 5, 5))
den, sigma, rank = denoise_with_R(dwi, mask, Rs, group)
```

Arrays in, arrays out — `nibabel` is only used at the file I/O boundary.

## Development

Open in the devcontainer (**Dev Containers: Reopen in Container**), picking the GPU
config where a GPU is available. See `.devcontainer/README.md`.

## License

MIT
