# camppca

**Covariance-aware MPPCA** — MPPCA denoising for diffusion MRI that
accounts for spatially correlated noise.

Standard MPPCA assumes white noise. Real EPI reconstruction (GRAPPA/SENSE, partial
Fourier, apodization) correlates noise across neighboring voxels, so MPPCA
under-estimates the noise level and under-denoises. camppca estimates the spatial noise
covariance from the data itself — no noise-only scan needed — and denoises in the
whitened domain, where the Marchenko–Pastur assumption holds.

## Install

```bash
pip install -e .
```

Python ≥ 3.10. numpy, scipy, joblib, threadpoolctl and nibabel, on the CPU.

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
| `--noise PATH` | the σ map, in the input's units — a first-class product, as `dwidenoise -noise` is |
| `--qc DIR` | things to look at: the correlation pattern as NIfTI + `rho.png`, the relative noise map, σ, rank, `provenance.json`, and the check of one pattern across regions and volume halves, `noise_model_check.txt` and `.json`. The check refits the noise model 23 times. |

**Supply a brain mask with `--mask`.** The noise model is estimated inside it. Every voxel
is denoised, so brain the mask missed at its edge is denoised too. `--mask-only` denoises
only inside the mask, which is faster. Without a mask, one is estimated from the mean b0 as a
fallback and written next to the output, so it can be checked and reused.
`-fslgrad` only picks the b0s that fallback mask is drawn from. Without it the mask comes
from the mean of every volume. Every volume enters the estimate either way.
`--patch X Y Z` sets the patch, odd in every dimension. The default follows dwidenoise.
It is the smallest odd cube with at least as many voxels as there are volumes: `5 5 5` up
to 125 volumes, `7 7 7` up to 343.
`--level-rounds N` re-estimates the relative noise map from the whitened patches up to N times
(default 0). `--shrink gd` or `lr` shrinks the kept components instead of keeping them
whole. `--algorithm` picks the Marchenko-Pastur rule, `cordero-grande` by default.

### Complex data: phase stabilization

With magnitude and phase images, rotate the data onto the real axis first, then denoise
the real part:

```bash
camppca phase   mag.nii.gz phase.nii.gz real.nii.gz --imag imag.nii.gz
camppca denoise real.nii.gz out.nii.gz --mask brain.nii.gz -fslgrad in.bvec in.bval
```

The real part keeps Gaussian noise, so it has no Rician noise floor. The recipe is the
default phase step of NORDIC's NIFTI_NORDIC2. In each slice and volume a smooth estimate
of the phase is removed. It comes from the decorrelated filter of Sprenger et al. (2017). The phase is read
in radians. For any other format, `--phase-range LOW HIGH` gives the values that span one
full turn, such as `-4096 4096` for Siemens integers or `-180 180` for degrees. A phase
wider than the declared turn stops the run before any work is done.
`--imag` writes the imaginary part, which should hold noise only.

### Complex data: denoising both channels

`camppca denoise` also takes a complex image and writes a complex one. Both channels are
denoised together. The noise model is estimated on the real part. σ is the noise level of
one channel. After phase stabilization the imaginary part holds mostly noise, so denoising
the real part alone usually gives the lower error. On EDDEN's 2 mm data it did.

## Python

```python
from camppca import denoise_camppca

result = denoise_camppca(dwi, mask, bvals=bvals)  # mask is optional
result.denoised     # (X, Y, Z, N)
result.sigma        # (X, Y, Z) the noise level, in data units, without the flat mode
result.rank         # (X, Y, Z) retained rank
result.model        # the noise model it estimated along the way
```

Standard white-noise MPPCA is `camppca.denoise_mppca(dwi, mask)`, which returns the same
result object with `model` unset.

Estimation and denoising are separate, so the noise model can be inspected, reused, or
supplied from elsewhere:

```python
from camppca import estimate_noise_model, denoise_with_noise_model

model = estimate_noise_model(dwi, mask, bvals=bvals)   # patch: dwidenoise's rule
model.rho           # the correlation pattern, no level
model.d             # relative noise level per voxel
model.patch_correlation(model.fitted_to["patch"])   # the n_vox × n_vox patch correlation, on demand

result = denoise_with_noise_model(dwi, model)    # the patch the model was estimated for
```

That split is deliberate: `denoise_with_noise_model` never asks where the model came from,
so a differently-estimated covariance feeds the same engine.

The model assumes one correlation pattern for the whole brain and every volume.
`check_noise_model` refits it on 8 regions and on 5 random halves of the volumes, and
reports how much each difference would change the noise level. `--qc DIR` runs it and
writes `noise_model_check.txt`.

```python
from camppca import check_noise_model

check = check_noise_model(dwi, model)     # 23 refits of the loop
print("\n".join(check.report()))
```

Arrays in, arrays out — `nibabel` is only used at the file I/O boundary.

## Notes

- **The work runs in a pool of CPU processes**, one per physical core, through joblib.
  `workers=1` (`--workers 1`)
  runs everything in one process. `CAMPPCA_WORKERS` sets the default, and a container's
  CPU quota caps it. The image reaches the workers as a memory-mapped file in
  the temp directory; set `TMPDIR` or `JOBLIB_TEMP_FOLDER` to move it, to local scratch on
  a cluster for example.
- **Output is center-voxel by default**, with no overlap averaging. Intentional — it
  matches the MP theory the noise estimate rests on. `aggregate="mean"`
  (`--aggregate mean`) averages the overlapping patch reconstructions instead, and
  `stride` (`--stride N`) keeps a patch every N voxels per axis.
- **σ leaves out the noise the whole patch shares.** That part depends on an assumption
  the estimate cannot check, so it is projected out. Under positive correlation σ is a
  little below the voxel's full noise level, about 1.6% on EDDEN's 2 mm data. For white
  noise the two are equal.
- **With `--mask-only`, voxels outside the mask pass through unchanged**, and `sigma`
  is zero there. Apply the mask yourself if you want zeros.
- **NaN or inf in the input enter the fit as zero** and stay NaN or inf in the output,
  with a warning. Resampled images often carry NaN in the background.
- **Keep ≳ 50 volumes.** The noise-level estimator degenerates as the retained rank
  approaches the smaller of the patch's voxel count and the volume count.

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
