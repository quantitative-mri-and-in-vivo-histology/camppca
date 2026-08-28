# camppca

**Covariance-Aware Marchenko–Pastur PCA** — MP-PCA denoising for diffusion MRI that
accounts for spatially correlated noise.

Standard MP-PCA assumes white noise. Real EPI reconstruction (GRAPPA/SENSE, partial
Fourier, apodization) correlates noise across neighboring voxels, so MP-PCA
under-estimates the noise level and under-denoises. camppca estimates the spatial noise
covariance from the data itself — no noise-only scan needed — and denoises in the
whitened domain, where the Marchenko–Pastur assumption holds.

## Install

```bash
pip install -e .
```

Python ≥ 3.10. The engine is JAX and runs on CPU out of the box; for GPU, install
`jax[cuda12]`.

## Command line

```bash
camppca denoise  in.nii.gz out.nii.gz --noise sigma.nii.gz --qc qc/ -fslgrad in.bvec in.bval
camppca estimate in.nii.gz --qc qc/                                 -fslgrad in.bvec in.bval
```

`estimate` stops after the noise model — seconds, no MP-PCA — for inspection or QC.

| flag | writes |
|---|---|
| *(positional out)* | the denoised 4D image |
| `--noise PATH` | the σ̂ map, in the input's units — a first-class product, as `dwidenoise -noise` is |
| `--qc DIR` | things to look at: the correlation pattern as NIfTI + `rho.png`, the level field, σ, rank, `provenance.json` |

Without `-fslgrad` no volume can be recognised as a b0, so every volume enters the fit.
`--patch X Y Z` sets the patch (default `5 5 5`, odd in every dimension).

## Python

```python
from camppca import denoise_camppca

result = denoise_camppca(dwi, bvals, mask)  # mask is optional
result.denoised     # (X, Y, Z, N)
result.sigma_hat    # (X, Y, Z) MP-PCA's noise estimate, in data units
result.rank         # (X, Y, Z) retained rank
result.model        # the noise model it estimated along the way
```

Standard white-noise MP-PCA is `camppca.denoise_mppca(dwi, mask)`, which returns the same
result object with `model` unset.

Estimation and denoising are separate, so the noise model can be inspected, reused, or
supplied from elsewhere:

```python
from camppca import estimate_noise_model, denoise_with_noise_model

model = estimate_noise_model(dwi, bvals, mask, patch=(5, 5, 5))
model.rho           # the correlation lag table — the pattern, no level
model.d             # relative noise level per voxel
model.sigma_p2s     # absolute noise level per voxel (drives whitening; NOT sigma_hat)
model.patch_correlation((5, 5, 5))          # the P×P patch correlation, derived on demand

result = denoise_with_noise_model(dwi, model, patch=(5, 5, 5))
```

That split is deliberate: `denoise_with_noise_model` never asks where the model came from,
so a differently-estimated covariance feeds the same engine.

Arrays in, arrays out — `nibabel` is only used at the file I/O boundary.

## Notes

- **Output is centre-voxel only**, with no overlap averaging. Intentional — it matches the
  MP theory the noise estimate rests on — but it differs from what `dwidenoise` users expect.
- **Keep ≳ 50 volumes in the pooled patch.** The noise-level estimator degenerates as the
  retained rank approaches `min(P, N)`.
- **P2S needs many more mask voxels than volumes.** It fits N−1 predictors across the mask,
  so a small mask deflates both the correlation estimate and σ.

`CLAUDE.md` carries the notation, the method invariants, and the reasoning behind them.

## Development

Open in the devcontainer (**Dev Containers: Reopen in Container**), picking the GPU config
where a GPU is available. See `.devcontainer/README.md`.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
