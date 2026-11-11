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

## References

- Cordero-Grande L et al. (2019). Complex diffusion-weighted image estimation via matrix recovery under general noise models. NeuroImage. [doi:10.1016/j.neuroimage.2019.06.039](https://doi.org/10.1016/j.neuroimage.2019.06.039)
- Eichner C et al. (2015). Real diffusion-weighted MRI enabling true signal averaging and increased diffusion contrast. NeuroImage. [doi:10.1016/j.neuroimage.2015.07.074](https://doi.org/10.1016/j.neuroimage.2015.07.074)
- Gavish M, Donoho DL (2017). Optimal shrinkage of singular values. IEEE Trans Inf Theory. [doi:10.1109/TIT.2017.2653801](https://doi.org/10.1109/TIT.2017.2653801)
- Leeb W, Romanov E (2021). Optimal spectral shrinkage and PCA with heteroscedastic noise. IEEE Trans Inf Theory. [doi:10.1109/TIT.2021.3055075](https://doi.org/10.1109/TIT.2021.3055075)
- Manjón JV et al. (2013). Diffusion weighted image denoising using overcomplete local PCA. PLoS One. [doi:10.1371/journal.pone.0073021](https://doi.org/10.1371/journal.pone.0073021)
- Manzano Patron JP et al. (2024). Denoising diffusion MRI: considerations and implications for analysis. Imaging Neuroscience. [doi:10.1162/imag_a_00060](https://doi.org/10.1162/imag_a_00060)
- Marčenko VA, Pastur LA (1967). Distribution of eigenvalues for some sets of random matrices. Math USSR-Sbornik. [doi:10.1070/SM1967v001n04ABEH001994](https://doi.org/10.1070/SM1967v001n04ABEH001994)
- Moeller S. NORDIC_Raw. [github.com/SteenMoeller/NORDIC_Raw](https://github.com/SteenMoeller/NORDIC_Raw)
- Moeller S et al. (2021). NOise reduction with DIstribution Corrected (NORDIC) PCA in dMRI with complex-valued parameter-free locally low-rank processing. NeuroImage. [doi:10.1016/j.neuroimage.2020.117539](https://doi.org/10.1016/j.neuroimage.2020.117539)
- Olesen JL, Jespersen SN (2020). About the slightly modified MP-PCA. [github.com/Neurophysics-CFIN/MP-PCA-Denoising](https://github.com/Neurophysics-CFIN/MP-PCA-Denoising)
- Sprenger T et al. (2017). Real valued diffusion-weighted imaging using decorrelated phase filtering. Magn Reson Med. [doi:10.1002/mrm.26138](https://doi.org/10.1002/mrm.26138)
- Tournier JD et al. (2019). MRtrix3: A fast, flexible and open software framework for medical image processing and visualisation. NeuroImage. [doi:10.1016/j.neuroimage.2019.116137](https://doi.org/10.1016/j.neuroimage.2019.116137)
- Veraart J, Fieremans E, Novikov DS (2016). Diffusion MRI noise mapping using random matrix theory. Magn Reson Med. [doi:10.1002/mrm.26059](https://doi.org/10.1002/mrm.26059)
- Veraart J et al. (2016). Denoising of diffusion MRI using random matrix theory. NeuroImage. [doi:10.1016/j.neuroimage.2016.08.016](https://doi.org/10.1016/j.neuroimage.2016.08.016)

## Development

Open in the devcontainer (**Dev Containers: Reopen in Container**). See
`.devcontainer/README.md`.

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
