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

**Supply a brain mask with `--mask`.** The noise model is estimated inside it. Every voxel
is denoised, so brain the mask missed at its edge is denoised too. `--mask-only` denoises
only inside the mask, which is faster. Without a mask, one is estimated from the mean b0 as a
fallback and written next to the output, so it can be checked and reused.
`-fslgrad` only picks the b0s that fallback mask is drawn from. Without it the mask comes
from the mean of every volume. Every volume enters the estimate either way.
`--patch X Y Z` sets the patch, odd in every dimension. The default follows dwidenoise.
It is the smallest odd cube with at least as many voxels as there are volumes: `5 5 5` up
to 125 volumes, `7 7 7` up to 343.
`--level-rounds N` re-estimates the level field from the whitened patches up to N times
(default 0).

### Complex data: phase stabilization

With magnitude and phase images, rotate the data onto the real axis first, then denoise
the real part:

```bash
camppca phase   mag.nii.gz phase.nii.gz real.nii.gz --imag imag.nii.gz
camppca denoise real.nii.gz out.nii.gz --mask brain.nii.gz -fslgrad in.bvec in.bval
```

The real part keeps Gaussian noise, so it has no Rician noise floor. In each slice and
volume the phase of the mean over all volumes is removed first, then a smooth estimate of
the phase that is left. `--filter dk`, the default, estimates it with the decorrelated
kernel of Sprenger et al. (2017). `--filter tukey --fwhm 0.58` uses a k-space low-pass, as
in the EDDEN preprocessing (Manzano-Patrón et al. 2024). The phase is read in radians or
as the scanner stores it, -4096 to 4094 on Siemens. `--imag` writes the imaginary part,
which should hold noise only.

On complex white noise the real part's mean over pure noise is 0.28σ with `dk` and 0.91σ
with `tukey` at 58%, against 1.25σ for the magnitude. In the image corners of EDDEN's 2 mm
data it was 0.07σ with `dk`, against 1.35σ for the magnitude.

### Complex data: denoising both channels

`camppca denoise` also takes a complex image and writes a complex one. Both channels are
denoised together. The noise model is estimated on the real part. σ̂ is the noise level of
one channel. After phase stabilization the imaginary part holds mostly noise, so denoising
the real part alone usually gives the lower error. On EDDEN's 2 mm data it did.

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

model = estimate_noise_model(dwi, bvals, mask)   # patch: dwidenoise's rule
model.rho           # the correlation lag table — the pattern, no level
model.d             # relative noise level per voxel
model.patch_correlation(model.provenance["patch"])   # the P×P patch correlation, on demand

result = denoise_with_noise_model(dwi, model)    # the patch the model was estimated for
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
- **With `--mask-only`, voxels outside the mask pass through unchanged**, and `sigma_hat`
  is zero there. Apply the mask yourself if you want zeros.
- **NaN or inf in the input enter the fit as zero** and stay NaN or inf in the output,
  with a warning. Resampled images often carry NaN in the background.
- **Keep ≳ 50 volumes in the pooled patch.** The noise-level estimator degenerates as the
  retained rank approaches `min(P, N)`.

Phase stabilization follows Eichner et al., NeuroImage 2015;122:373-384, and Sprenger et
al., Magn Reson Med 2017;77:559-570. The EDDEN data and its preprocessing are from
Manzano-Patrón et al., Imaging Neuroscience 2024, doi:10.1162/imag_a_00060.

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
