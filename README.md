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

Python ≥ 3.10. numpy and scipy only, on the CPU.

## Command line

```bash
camppca denoise  in.nii.gz out.nii.gz --mask brain.nii.gz --noise sigma.nii.gz --qc qc/ -fslgrad in.bvec in.bval
camppca estimate in.nii.gz --mask brain.nii.gz --qc qc/                                 -fslgrad in.bvec in.bval
```

`estimate` stops after the noise model, without denoising, for inspection or QC.

| flag | writes |
|---|---|
| *(positional out)* | the denoised 4D image |
| *(no `--mask`)* | `out_mask.nii.gz` next to the output: the brain mask estimated in its place |
| `--noise PATH` | the σ̂ map, in the input's units — a first-class product, as `dwidenoise -noise` is |
| `--qc DIR` | things to look at: the correlation pattern as NIfTI + `rho.png`, the level field, σ̂, rank, `provenance.json` |

**Supply a brain mask with `--mask`.** The noise model is estimated inside it, and only
voxels inside it are denoised. Without one, a mask is estimated from the mean b0 as a
fallback and written next to the output, so it can be checked and reused.
`-fslgrad` only picks the b0s that fallback mask is drawn from. Without it the mask comes
from the mean of every volume. Every volume enters the estimate either way.
`--patch X Y Z` sets the patch (default `5 5 5`, odd in every dimension).
`--level-rounds N` re-estimates the level field from the whitened patches up to N times
(default 0).

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
model.patch_correlation((5, 5, 5))          # the P×P patch correlation, derived on demand

result = denoise_with_noise_model(dwi, model, patch=(5, 5, 5))
```

That split is deliberate: `denoise_with_noise_model` never asks where the model came from,
so a differently-estimated covariance feeds the same engine.

Arrays in, arrays out — `nibabel` is only used at the file I/O boundary.

## Notes

- **The work runs in a pool of CPU processes**, one per physical core, through joblib.
  On a 16-core machine that was faster than an RTX 4060 Ti. `workers=1` (`--workers 1`)
  runs everything in one process. The image reaches the workers as a memory-mapped file in
  the temp directory; set `TMPDIR` or `JOBLIB_TEMP_FOLDER` to move it, to local scratch on
  a cluster for example.
- **Output is centre-voxel by default**, with no overlap averaging. Intentional — it
  matches the MP theory the noise estimate rests on. `aggregate="mean"`
  (`--aggregate mean`) averages the overlapping patch reconstructions instead, and
  `stride` (`--stride N`) keeps a patch every N voxels per axis.
- **Voxels outside the mask pass through unchanged**, and `sigma_hat` is zero there. The
  noise model is only valid inside the mask. Apply the mask yourself if you want zeros.
- **Keep ≳ 50 volumes in the pooled patch.** The noise-level estimator degenerates as the
  retained rank approaches `min(P, N)`.

`CLAUDE.md` carries the notation, the method invariants, and the reasoning behind them.

## Development

Open in the devcontainer (**Dev Containers: Reopen in Container**). See
`.devcontainer/README.md`.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
