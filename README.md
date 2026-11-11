# camppca

Covariance-aware MPPCA for diffusion MRI.

Image reconstruction reshapes the noise and breaks MPPCA's assumption of i.i.d. noise,
so part of the noise is kept as signal. camppca estimates the noise covariance from the
reconstructed images and restores that assumption before MPPCA.

## Install

```bash
pip install -e .
```

## Command line

```bash
camppca denoise in.nii.gz out.nii.gz --mask brain.nii.gz --noise sigma.nii.gz --qc qc/
```

Give a brain mask with `--mask`.

| option | does |
|---|---|
| *(positional out)* | the denoised 4D image |
| `--mask PATH` | the brain mask the noise model is estimated in. Every voxel is still denoised. Without it, a mask is drawn from the mean b0, or from the mean of all volumes without `-fslgrad BVEC BVAL`, and written next to the output as `out_mask.nii.gz`. `--lowb B` sets the b-value up to which a volume counts as a b0, default 100. |
| `--mask-only` | denoise only inside the mask, which is faster. Voxels outside pass through, with σ zero. |
| `--noise PATH` | the σ map, in the input's units, as `dwidenoise -noise` writes it |
| `--qc DIR` | the correlation pattern as NIfTI and `rho.png`, the relative noise map, σ, rank, `provenance.json`, and the check of one pattern across regions and volume halves. The check refits the noise model 23 times. |
| `--patch X Y Z` | the patch size, odd on every axis. By default the smallest odd cube with at least as many voxels as volumes, as in dwidenoise. |
| `--aggregate center\|mean` | keep each patch's center voxel (default), or average the overlapping patch reconstructions. With `mean`, `--stride N` keeps a patch every N voxels, default 1. |
| `--shrink hard\|gd\|lr` | keep the kept components whole (default), or shrink them. `gd` is Gavish and Donoho's optimal shrinkage for the whitened patch. `lr` is Leeb and Romanov's version of it for the error in data units. |
| `--level-rounds N` | re-estimate the relative noise map from the whitened patches N times, default 0 |
| `--algorithm veraart\|cordero-grande\|jespersen` | the Marchenko–Pastur rule, default `cordero-grande` |
| `--workers N` | CPU processes to split the work over. Default one per physical core, or `CAMPPCA_WORKERS`, capped by a container's CPU quota. `1` runs in one process. |

### Complex denoising

Complex data is currently handled by phase stabilization. The data is rotated onto the
real axis, and the real part is denoised:

```bash
camppca phase   mag.nii.gz phase.nii.gz real.nii.gz --imag imag.nii.gz
camppca denoise real.nii.gz out.nii.gz --mask brain.nii.gz
```

The real part keeps Gaussian noise, so it has no Rician noise floor. The recipe is the
default phase step of NORDIC's NIFTI_NORDIC2. In each slice and volume a smooth estimate
of the phase is removed. It comes from the decorrelated filter of Sprenger et al. (2017).
The phase is read in radians. For any other format, `--phase-range LOW HIGH` gives the
values that span one full turn, such as `-4096 4096` for Siemens integers or `-180 180`
for degrees. A phase wider than the declared turn stops the run before any work is done.
`--imag` writes the imaginary part, which should hold noise only.

## Python API

```python
from camppca import denoise_camppca

result = denoise_camppca(dwi, mask, bvals=bvals)
result.denoised, result.sigma, result.rank
```

## Notes

- **The image reaches the worker processes as a memory-mapped file** in the temp
  directory. Set `TMPDIR` or `JOBLIB_TEMP_FOLDER` to move it, for example to local
  scratch on a cluster.
- **Keep ≳ 50 volumes.** The noise-level estimator degenerates as the retained rank
  approaches the smaller of the patch's voxel count and the volume count.

## Development

Open in the devcontainer (**Dev Containers: Reopen in Container**). See
`.devcontainer/README.md`.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
