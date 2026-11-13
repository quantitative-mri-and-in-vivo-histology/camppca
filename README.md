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
camppca denoise in.nii.gz out.nii.gz --noise sigma.nii.gz --mask brain.nii.gz
```

Status lines go to stderr, each with the time since the start, and the long passes show a
progress bar.

| option | does |
|---|---|
| *(positional out)* | the denoised 4D image |
| `--noise PATH` | the output noise level map σ, in the input's units |
| `--mask PATH` | a rough brain mask, within which the noise correlation model is estimated. The whole image is still denoised, unless `--mask-only` is set. Without a mask, one is generated from the mean of all volumes and written as `out_mask.nii.gz`. For diffusion data, `-fslgrad` can be given to generate a mask from the mean b0 instead. |
| `--mask-only` | restrict denoising to within the mask. |
| `-fslgrad BVEC BVAL` | FSL gradient table files (`.bvec` and `.bval`) for diffusion data, used to generate a mask from the mean b0. `--lowb B` sets the largest b-value that counts as a b0 (default 100). |
| `--patch X Y Z` | the patch size, odd on every axis. Default: the smallest odd cube with at least as many voxels as volumes. |
| `--aggregate center\|mean` | keep each patch's center voxel (default), or average the overlapping patch reconstructions. With `mean`, `--stride N` keeps a patch every N voxels (default 1). |
| `--shrink hard\|gd\|lr` | how the retained components are weighted. `hard` keeps them unchanged (default). `gd` is Gavish and Donoho's optimal shrinkage for the whitened patch. `lr` is Leeb and Romanov's version of it for the error in data units. |
| `--algorithm veraart\|cordero-grande\|jespersen` | the Marchenko–Pastur threshold rule used by every MPPCA step, in the noise model and the denoising. Default: `cordero-grande`. |
| `--level-rounds N` | re-estimate the relative noise map from the whitened patches N times. Default: 0. |
| `--workers N` | CPU processes to split the work over, or set the environment variable `CAMPPCA_WORKERS`. Default: one per physical core. |
| `--qc DIR` | writes the noise covariance model (the relative noise map d, and the correlation pattern ρ as NIfTI and `rho.png`), the final noise map σ and the rank map to DIR, and runs additional quality control analyses. These check that one correlation pattern holds across regions and volume halves. |
| `-q` | show only warnings and errors |
| `-v` | verbose: also show the progress of each round of the noise correlation fit, and the full Python traceback when an error stops the run |

### Complex denoising

For complex data, camppca denoises real-valued data after phase stabilization, which
rotates the data onto the real axis. This takes two steps:

```bash
camppca phase   mag.nii.gz phase.nii.gz real.nii.gz --imag imag.nii.gz
camppca denoise real.nii.gz out.nii.gz --mask brain.nii.gz
```

The phase is read in radians. For any other format, `--phase-range LOW HIGH` gives the
values that span one full turn, such as `-4096 4096` for Siemens integers or `-180 180`
for degrees. `--imag` writes the imaginary part, which should hold noise only.

### Estimate noise model only

`camppca estimate` writes the noise covariance model and its check to the `--qc` folder:

```bash
camppca estimate in.nii.gz --mask brain.nii.gz --qc qc/
```

## Python API

```python
from camppca import denoise_camppca, denoise_mppca

result = denoise_camppca(dwi, mask, bvals=bvals)
result.denoised, result.sigma, result.rank

plain = denoise_mppca(dwi)        # standard MPPCA, for comparison
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

## References

- Cordero-Grande L et al. (2019). Complex diffusion-weighted image estimation via matrix recovery under general noise models. NeuroImage. [doi:10.1016/j.neuroimage.2019.06.039](https://doi.org/10.1016/j.neuroimage.2019.06.039)
- Eichner C et al. (2015). Real diffusion-weighted MRI enabling true signal averaging and increased diffusion contrast. NeuroImage. [doi:10.1016/j.neuroimage.2015.07.074](https://doi.org/10.1016/j.neuroimage.2015.07.074)
- Gavish M, Donoho DL (2017). Optimal shrinkage of singular values. IEEE Trans Inf Theory. [doi:10.1109/TIT.2017.2653801](https://doi.org/10.1109/TIT.2017.2653801)
- Leeb W, Romanov E (2021). Optimal spectral shrinkage and PCA with heteroscedastic noise. IEEE Trans Inf Theory. [doi:10.1109/TIT.2021.3055075](https://doi.org/10.1109/TIT.2021.3055075)
- Manjón JV et al. (2013). Diffusion weighted image denoising using overcomplete local PCA. PLoS One. [doi:10.1371/journal.pone.0073021](https://doi.org/10.1371/journal.pone.0073021)
- Marčenko VA, Pastur LA (1967). Distribution of eigenvalues for some sets of random matrices. Math USSR-Sbornik. [doi:10.1070/SM1967v001n04ABEH001994](https://doi.org/10.1070/SM1967v001n04ABEH001994)
- Moeller S. NORDIC_Raw. [github.com/SteenMoeller/NORDIC_Raw](https://github.com/SteenMoeller/NORDIC_Raw)
- Moeller S et al. (2021). NOise reduction with DIstribution Corrected (NORDIC) PCA in dMRI with complex-valued parameter-free locally low-rank processing. NeuroImage. [doi:10.1016/j.neuroimage.2020.117539](https://doi.org/10.1016/j.neuroimage.2020.117539)
- Olesen JL, Jespersen SN (2020). About the slightly modified MP-PCA. [github.com/Neurophysics-CFIN/MP-PCA-Denoising](https://github.com/Neurophysics-CFIN/MP-PCA-Denoising)
- Sprenger T et al. (2017). Real valued diffusion-weighted imaging using decorrelated phase filtering. Magn Reson Med. [doi:10.1002/mrm.26138](https://doi.org/10.1002/mrm.26138)
- Tournier JD et al. (2019). MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. NeuroImage. [doi:10.1016/j.neuroimage.2019.116137](https://doi.org/10.1016/j.neuroimage.2019.116137)
- Veraart J, Fieremans E, Novikov DS (2016). Diffusion MRI noise mapping using random matrix theory. Magn Reson Med. [doi:10.1002/mrm.26059](https://doi.org/10.1002/mrm.26059)
- Veraart J et al. (2016). Denoising of diffusion MRI using random matrix theory. NeuroImage. [doi:10.1016/j.neuroimage.2016.08.016](https://doi.org/10.1016/j.neuroimage.2016.08.016)

## License

MIT
