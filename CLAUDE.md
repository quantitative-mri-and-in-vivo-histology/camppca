# camppca — scope and notation

**`camppca` — covariance-aware MPPCA.** The implementation of the denoising algorithm
described in the manuscript at
`~/gdrive/UKE/manuscripts/covar_denoising/initial/`, which cites this repository in its Data
and Code Availability section. MPPCA denoising for diffusion MRI that accounts for
spatially correlated noise.

⚠ **The name is `camppca`** — lower case, no hyphen, settled 2026-08-28 to match the package
and the paper. Not CAMPPCA, not CAMPPCA. Expanded as *covariance-aware MPPCA*.

Standard MPPCA assumes white noise. Real EPI reconstruction (GRAPPA/SENSE, partial
Fourier, apodization) correlates noise across neighboring voxels, so MPPCA
under-estimates the noise level and under-denoises. camppca estimates the spatial noise
covariance **from the data itself** (no noise-only scan), then denoises in the whitened
domain where the Marchenko–Pastur assumption actually holds.

Arrays in, arrays out. `nibabel` only at the I/O boundary.

Conventions for user-facing text: **MPPCA**, no hyphen, as in the paper (settled
2026-11-08). American spelling.

## Scope

**In:** noise covariance estimation from MPPCA itself; the relative noise map; whitened
MPPCA (whiten → MPPCA → unwhiten); phase stabilization of complex data, as a step
before denoising; complex denoising; the CLI around them.

**Out:** the warp/alignment lane (reusing FSL eddy displacement fields to denoise in a
common anatomical frame) — a separate project. The research repo this is ported from is
called `warp_denoise` for historical reasons; nothing warp-related comes with it.

Being ported from `~/workspace/cdmri_26_adaptive_denoising` @ `ec08de0`. Verbatim source
copies live in `reference/` (gitignored scratch). Brief: `docs/local/port_core_brief.md`.
File:line citations below refer to those **reference** files, not to ported modules.

## The noise model

The patch noise covariance factorizes into a **level** that varies in space and a
**correlation pattern** that does not:

    Sigma = sigma^2 * D * C * D          D = diag(d), d the relative noise map

`rho` (the correlation pattern) and `d` are the noise model. `C` is derived from `rho` plus a
patch shape, so it is computed on demand, never stored.

⚠ `D` holds **levels, not variances** — hence `D C D`, not `D^(1/2) C D^(1/2)`.

## Notation

| object | paper | code | notes |
|---|---|---|---|
| patch matrix, rows = voxels | X (M×N) | — | |
| unit-diagonal patch correlation | C | `C` | **not** `R` |
| lag correlation table | ρ(h) | `rho` | `rho[table_index(dx, dy, dz, lag)]`, the half dz ≥ 0, 1.0 at zero lag |
| patch noise covariance | Σ = σ²DCD | — | **paper only** — never materialized in code |
| whitener | W = C^(-1/2) D^(-1) | `whitener` + row division by `d` | **σ-free by design** |
| relative noise map | d(x) | `d` | **not** `g` — that is the g-factor |
| diagonal level matrix | D = diag(d) | `D` | |
| true noise level | σ | — | |
| MPPCA's per-patch level | σ_MPPCA | `sigma_mppca` | in units of `d` inside the engine |
| reported noise level | σ(x), Eq. 6 | `sigma` | no hats: this repo has no true σ to set apart |
| retained rank | P | `rank` | ⚠ see the P collision below |
| aspect ratio | γ = N/M | — | |
| box shape, e.g. (5,5,5) | — | `patch` | a **denoiser** knob, not a noise property |
| max lag covered by `rho` | — | `lag` / `LAG` | |

⚠ **The whitener is σ-free on purpose.** `W = C^(-1/2) D^(-1)`, *not* `Σ^(-1/2)`. Whitening
by `Σ^(-1/2)` would divide the noise level out, MPPCA would estimate σ = 1, and the
"σ is an output, not an input" claim collapses. `W` normalizes shape only; the level
stays in the data for the per-patch MP fit to find.

**Banned words.** Both are silent-corruption hazards:
- **`kernel`** — upstream it means the patch shape *and* the correlation pattern. Use
  `patch` and `rho`. The DK phase smoother is a filter.
- **`R`** — replaced by `C`; `Σ` carries the level, `C` does not.

### Dimensions

| paper | code | is |
|---|---|---|
| M | `P` = prod(patch) | voxels in a patch |
| N | `N` | volumes |
| — | `V` | mask voxels |

In code these are spelled `n_vox`, `n_vols`, `n_mask` — the bare letters were
renamed once the self-test was green. `Mp`/`Np` became **`n_small`/`n_large`**: the
reference had `Mp, Np = min(P, N), max(P, N)`, the *sorted* pair, so under this notation
the old names read as patch-M and patch-N, which they never meant.

⚠ **`P` is the retained rank, as in the paper.** The reference code used `P` for the
voxels in a patch, now `n_vox`. No other `P` in code or docs: the flat-mode projector is
written out as `(I − uuᵀ)`. `U` is the left singular vectors, as in the paper. The
sphericity is named in words.

Worked example: the story doc's "M = 125, N = 34" (paper convention: 125 voxels,
34 volumes) lands in code as `Mp = 34`, `Np = 125`. "Rank → Mp" means rank approaching
the 34 volumes.

## The seam

    estimator (source-specific)          engine (source-agnostic)
    ---------------------------          ------------------------
    MPPCA level + loop -> rho, d  ---->  C -> whiten -> MPPCA -> unwhiten

The two halves meet at **the noise model** (`rho` + `d`). A follow-up paper estimates
`rho` from the alignment/interpolation operator and feeds the same engine, so the engine
must never assume how `rho` was estimated.

There is one engine. `denoise_mppca` calls `denoise_whitened` with `C=None` and no level
field, which skips the whitening multiply and takes the center row from the patch. Its
output was bit for bit the same as the old separate copy in 228 cases (2026-11-05). Every
split goes through `_whiten_split`: both denoisers, the level pass (eigenvalues only) and
the loop.

Both halves are numpy and scipy, CPU only. JAX was removed 2026-11-03: per-patch
eigendecompositions over a pool of CPU processes beat the GPU on a 16-core host. The
estimator reuses the engine's own split.

Patch2Self was removed from the estimator 2026-11-02 (spec:
`docs/local/2026-11-02_no_p2s_estimator_spec.md`). Starting the loop from white noise,
with `d` = standard MPPCA's σ pooled 3³, gave the same pattern and median σ within
0.3% on four datasets. The P2S code moves to the paper repo, which still scores whiteness with it.

### The loop

`estimate_noise_model` starts `rho` from white noise (1 at lag 0) and refines it until it
explains the MPPCA bulk of its patches at once (`noise_model/loop.py`). It fits
to the denoiser's own split: `algorithm` and `noise_field` are passed to
`estimate_noise_model`, stored with the patch in `NoiseModel.fitted_to`, and a mismatch
at denoising warns. The denoiser reads `fitted_to`, never `provenance`, which is only a
record: a second estimator sets `fitted_to` or leaves keys out.
Every volume enters, b0s included, because the denoiser whitens them all. `bvals` only
picks the b0s the default mask is drawn from.

The patches lie on a grid (`_sample_centers`), each wholly inside the mask, with stride
equal to the patch, so they tile the mask without overlap. The grid's phase is the one
with the most centers. Above 5000 the stride widens, below 500 it narrows to overlap,
down to every voxel. No seed. EDDEN 0.9 mm gets 4579 patches, 2 mm 1563. Measured
2026-11-09 (`docs/local/2026-11-09_loop_sampling.md`): σ moved by at most 0.1% above a
few hundred patches, ρ stopped improving at about 700 patches of 7³, and at 2 mm
overlapping centers up to 7060 came no closer than the tiling. A grid was no more
precise than random centers at the same count. Patches reaching outside the mask lowered
the sphericity a little and gained nothing. Until then the loop drew 3000 random centers.

`d` is standard MPPCA's σ pooled 3³ (`whitened_level` with `C=None`, eigenvalues only).
⚠ The pooling is needed: unpooled, the rank grain under strong correlation made the final
`C` near singular on ideas02 (smallest eigenvalue 1e-9).

The level pass runs on a stride-2 grid of patch centers (`_strided_level`), filled in
between by a tent interpolation in variance, then pooled. MPPCA's level is a patch mean
and the pooling blurs at that scale anyway. Fixed, no option. EDDEN 0.9 mm: 26 s against
92 s, table within 5e-5, median d 0.3%, σ 0.1%, no voxel's d over 10%. ⚠ The two outer
mask layers are computed in full. Filled, the edge saw only inner grid points, and edge
voxels' d came out up to 79% too high where the level climbs steeply inward. It needs no flat-mode correction: no
`C` is involved, and only ratios of `d` matter.

`level_rounds` (default 0) re-estimates `d` as `pool(d * s)`, with `s` the whitened σ
of every mask voxel's patch, then refits the loop. The fixed point is a flat `s`. It
changed the sphericity by at most 0.006 and σ by at most 0.3% at three to four times the cost, hence
the default. Edge patches include outside-mask voxels, where `d` takes the nearest mask
voxel's level. On synthetic data the rim of half a patch at the mask edge then stays
within a median 0.8% of the true level, up to 5%. The old fill, the median over the mask,
pushed it off by a median 2.9%, up to 32%.

Each round fits one value per lag class (364 at 5³, 1098 at 7³) to the pooled bulk, in whitened
units, by conjugate gradients. ⚠ **The hole.** Every patch keeps its patch-mean pattern
as signal, so the bulk cannot see one direction: the same value added to every lag. It
is fixed by holding the mean over the far lag classes, those on the edge of the patch's
lag box, at zero. Do not go back to a free solve of `Sigma` with lag averaging: there the
hole is a near-null eigenvalue (1.1e-4 on the crop), a converged solve divides by it, and
the pattern ran away (sphericity 0.16 to 9). The old 30 Richardson steps only hid that by stopping
early.

⚠ **The split is biased, and the loop inherits it.** Signal components near the MP edge
are kept with patterns tilted toward the noise, and weaker ones stay in the bulk. Both
are smooth and read as extra correlation. Paper phantom (Hann f 0.5, 80 volumes, 5³):
the true z correlation is 0, the loop reports 0.05 at lag 1, and Patch2Self overshot
too. With a signal MPPCA separates perfectly the loop is within 0.005. This is the
split's limit, not the fit's. Estimating from the volume side (project every voxel off
a global signal subspace) was tried and fails on real data: the signal subspace is local.

The split runs in float32, its bulk summed in float64. It needed float64 only while the
free solve left `C` near singular. Against float64 the table moved by at most 2.4e-5.
The fit runs in float32 too, within 1e-4 of a float64 least squares.

### One whitener

The paper and the code both whiten with `W = C^(-1/2) D^(-1)`: divide rows by `d`, then
apply `C^(-1/2)` (`whitened.py:166`, `:200-206` in the reference). `WᵀW = (DCD)^(-1)`, so
the symmetric `(DCD)^(-1/2)` would give the same Gram, spectrum, MP threshold, retained
subspace and reconstruction. Older paper drafts used it.

## Layout

    camppca/
      noise_model/
        __init__.py        NoiseModel, estimate_noise_model() -> NoiseModel
        masks.py           brain mask
        loop.py            the loop, refines rho from white
      denoiser/
        __init__.py        denoise_camppca(), denoise_with_noise_model()
        camppca.py         covariance-aware MPPCA — the method, and the one engine
        mppca.py           standard white-noise MPPCA: the engine with C = None
        mp_threshold.py    mp_threshold() — the Marchenko-Pastur cutoff rule
        gather.py          patch extraction, a tile at a time
        aggregate.py       patch averaging, the opt-in aggregate="mean"
        spectrum.py        batched eigh through scipy's LAPACK, descending
        result.py          DenoiseResult, what both denoisers return
      patch.py             patch geometry — both halves of the seam need it
      phase.py             phase stabilization: complex data onto the real axis, before
                           denoising
      pool.py              CPU worker processes through joblib — both halves use it
      io.py                nibabel edge
      cli.py               camppca denoise | estimate
      __init__.py          public names

One subpackage per half of the seam. The package name carries the noun, so files inside
do not repeat it, and each `__init__.py` is that half's public face.

**The public API is flat.** Everything a user calls is `camppca.<name>`:
`denoise_camppca`, `denoise_mppca`, `denoise_with_noise_model`, `estimate_noise_model`,
`brain_mask`, `NoiseModel`, `DenoiseResult`, `stabilize_phase`. Nothing public needs a `camppca.denoiser.`
or `camppca.noise_model.` prefix.

- **Never `estimate` or `model` bare** — they do not say *what* is estimated or modeled.
  It is always the **noise model**: `estimate_noise_model`, `denoise_with_noise_model`.
  This deviates from the port brief's frozen `estimate_noise` / `denoise_with_model`,
  deliberately.
- `mp_threshold.py`, `gather.py` and `result.py` sit in `denoiser/`. `patch.py` sits
  above it because the loop needs it too.
- `loop.py` imports denoiser internals, and `denoiser/__init__.py` imports
  `noise_model`. So `estimate_noise_model` imports `loop` and `whitened_level` inside the
  function, or the import is circular.
- `NoiseModel` lives in `noise_model/__init__.py`, not with the estimation math: a future
  alignment-based estimator builds the same object and must not import the loop to get it.
  The level guard lives there too, as `NoiseModel.effective_d()`, since it belongs to the
  model. The estimator uses the same `_effective_d` and `_good_level`.
- Flat, with `noise_model_loop.py`-style prefixes, is the same structure with the noun
  repeated in every filename. Do not "simplify" back to it.

## Method invariants

Breaking any of these corrupts results without an error:

- `LAG = max(patch[0], patch[1]) - 1` in-plane, and `dzmax = patch[2] - 1`. Smaller, and
  lag lookup wraps around. `dzmax` used to equal `LAG`, which crashed a one-slice patch
  on a thin volume.
- **The loop falls back to white noise** when a round's fitted table is not a valid
  correlation, smallest eigenvalue of its `C` below `MIN_EIGENVALUE`, checked before
  projecting. It warns, and provenance records `fallback`. Cause seen: in the real part
  of data whose phase was never stabilized, the signal is higher rank, part of it falls
  below the MP edge, and pooled over the loop's patches that smooth signal reads as strong
  correlation no table explains. `_build_C` then projected to a C with eigenvalue 1e-9,
  the next round whitened by up to 3e4, and float32 overflowed (EDDEN 2 mm raw real
  part, 2026-11-05). camppca with a white model is MPPCA with the relative noise map.
- `C` must have smallest eigenvalue ≥ 1e-3 (`MIN_EIGENVALUE`). A table valid for one
  patch can be indefinite for a larger one, and whitening by a near-singular `C` sends
  σ to ~1e-7. `patch_correlation` raises. `_build_C` still projects silently, because
  the loop needs that each round.
- Patch flattening is **C-order, z fastest**; `rho[table_index(dx, dy, dz, lag)]`.
  Paired — the two orders must agree.
- **The table's only symmetry is ρ(h) = ρ(−h).** It stores the half dz ≥ 0, and a lag
  with dz < 0 is read as (−dx, −dy, −dz) (`patch.table_index`). Until 2026-11-08 it was
  read as (dx, dy, |dz|), which added a mirror in z: 204 classes at 5³ instead of 364.
  Do not bring back a mirror in any axis. It fails for a noise pattern tilted off the
  axes, by GRAPPA kernels, oblique reslicing or any resampling.
- `rho` is a correlation (unit at zero lag): **pattern only**. The absolute level comes
  from the per-patch MP fit. Only *ratios* of `d` matter.
- **`d` is defined up to scale.** Rows are divided by it and the center voxel multiplied
  back, so a constant cancels in the denoised image, the rank and σ in data units (×0.25
  is bit-identical; a generic factor flips near-tied MP cuts by float32 rounding).
  Outside the mask `d` is the level of the nearest mask voxel.
- **At the image edge the box moves inward** until it fits, as in dwidenoise
  (`gather.box_centers`), and the voxel's own row is written back. `C` depends only on
  offsets within the box, so it is unchanged. Unwhitening takes that row of the patch,
  and the center-voxel "lr" weight reads `C` alone, so it needs nothing. In mean mode
  each distinct box counts once. Until 2026-11-05 the box wrapped to the opposite face,
  which mixed in voxels that are not neighbours: on a test volume an edge voxel came out
  51 to 421 off. Interior voxels were bit for bit unchanged in center mode.
- **One global `rho`.** There is no per-shell pattern and no per-shell level. `rho` is fit
  once across every volume.
- **The default patch is dwidenoise's rule** (`patch.default_patch`, MRtrix 3.0.4): the
  smallest odd cube with at least N voxels, each side clipped to the largest odd size
  that fits the image. MRtrix clips to the image size, even or not. `patch=None` means
  this everywhere, except `denoise_with_noise_model`, which takes the patch the model was
  estimated for: the loop fitted `rho` to that patch's split.
- **Center-voxel by default** — no overlap averaging. Intentional; matches the MP theory
  the noise estimate rests on. `aggregate="mean"` (with `stride`) averages the overlapping
  patch reconstructions as an option (`denoiser/aggregate.py`): plain mean, no weights,
  only patches centered in the region denoised, nothing written outside it, `sigma`
  the root of the mean `sigma^2`, `rank` the mean rank (float32). The whole rebuilt patch is
  `X V diag(w) V^T` for both denoisers: camppca's `C^(1/2) C^(-1/2) = I`, so the
  whole-patch unwhitening needs no `C^(1/2)` product. The center voxel is the same
  identity for one row: `e_c^T C^(1/2) Yw = e_c^T X`. Since 2026-11-05 the engine takes
  the voxel's row of `X` directly. Through `C^(1/2)` in float32 it was off a float64
  reference by 0.5 to 0.7 on values near 200, directly by 0.04 to 0.1.
- **Hard threshold by default, shrinkage opt-in** (`shrink="hard" | "gd" | "lr"`,
  `denoiser/shrink.py`). ⚠ The default is NOT hard because shrinkage would break
  exactness — it would not. The unwhitening identity `eᶜᵀ C^(1/2) · (Yw V diag(f) Vᵀ)` is
  plain associativity and never used idempotency of the projector, so it holds for ANY
  `f` (measured at 4e-16 for hard threshold, a GD profile, and a random profile). The
  reasons: a hard threshold needs only the edge LOCATION, while shrinkage scales every
  retained component by a continuous function of `sigma_mppca` — and it is fragile
  at small N (see Exp2 below). `"gd"` also optimizes in the WHITENED metric while the
  output lands in data units. `"lr"` (Leeb–Romanov) corrects for that, but its guarantee
  is for the whole patch at large sizes, and camppca keeps only the center voxel of a
  small patch. There its noise loudness `q_k` uses `C` alone, since the center level
  drops out. With `aggregate="mean"` the whole patch reaches the output, so `q_k` uses
  the patch's own levels, `|D C^(1/2) u_k|^2 / mean(d^2)`. Both shrinkers keep the MP
  rank and change only the weights. Standard MPPCA maps `"lr"` to `"gd"` (white noise).

## API rules

- **`sigma` is returned in data units.** The engine rescales the denoised image by
  `d` at the center voxel but *not* `sigma_mppca`, so the API must apply that multiply by
  default. σ is the headline metric — the scale of `d` must not leak into the reported
  number.
- **The reported σ leaves the flat mode out** (`_flat_mode_factor`, the paper's Eq. 6 in
  its per-voxel form): `σ² = σ²_MPPCA d² M/(M−1) [(I − uuᵀ) C (I − uuᵀ)]_cc`. The loop
  cannot see the flat direction of the table and fixes it by assuming the far lags
  average to zero. Off by `c`, the voxel's full level moves by about `c/2`, this one only
  at second order. It is one global factor, 1 for white noise, 0.984 on EDDEN 2 mm at
  5³. With `aggregate="mean"` the center entry becomes the mean diagonal. Do not move it
  into the level pass or the engine: `d` is defined up to scale, and the engine's σ
  feeds the shrinkers.
- `_effective_d` in `noise_model/__init__.py` is the **only** place `d` is guarded (bad or
  outside-mask voxels get the nearest good mask voxel's level). The engine takes the
  field as given. Bad includes a level below 1% of the median (`LEVEL_FLOOR`): a patch of
  near-constant data, such as zero-filled slices, has an MP level near zero, and
  whitening would divide by it. The loop also skips patches holding such a voxel.
  ideas01 sub-103 hit this in its top two slices: pooling returned NaN, and sub-01
  finished with a suspect correction of 16.8%.
- NaN and inf are zeroed for the fit and put back in the output (`gather.zero_nonfinite`),
  in `denoise_mppca`, `denoise_whitened` and `estimate_noise_model`. One NaN would
  otherwise poison every patch that holds it, and `eigh` runs with `check_finite=False`.
- **Every voxel is denoised by default**, inside the mask or not. camppca's `mask` says
  where the noise model is estimated, and a brain mask always misses some brain at its
  edge. `mask_only=True` (`--mask-only`) denoises only inside it and passes the rest
  through, about 2.7 times faster on EDDEN 2 mm. The loop fits only patches wholly
  inside the mask, so the output region never changes `rho`. Standard MPPCA takes a
  mask only as the region to denoise, and without one denoises everything.

## Phase stabilization

`phase.py`, `camppca phase`: complex data rotated onto the real axis, and the real part
denoised like a magnitude image. A separate step before `estimate`/`denoise`, so the
noise model is estimated on exactly the data that is denoised. One recipe, no options:
EDDEN's preprocessing with Sprenger's `"dk"` filter in place of their k-space low-pass.
The raw phase of a reference volume, the first within 95% of the brightest mean, is
removed from every volume. Then per slice and volume the phase of the dk smooth is removed.

- Scope: the paper is about magnitude data, complex is a supplement example. The recipe
  had to be a published one, so it is EDDEN's. Do not grow phase options again.
- EDDEN's own filter (`NIFTI_COMP_to_REAL.m`, a Hann window cubed, their "Tukey, FWHM
  58%") leaves the real part's mean over pure noise at 0.54σ, dk at 0.01σ. The reference
  step costs nothing on noise and removes a rough fixed phase dk alone does not follow:
  signal loss 0.19 down to 0.015 at a b0 SNR of 20 (2026-11-06). The reference volume
  itself ends up as its magnitude.
- The removed `mean_phase` option, the phase of the mean over all volumes, biased the
  real part by about σ/√N, since the mean carries each voxel's own noise. One reference
  volume does not: its noise is independent of the other volumes'.

- ⚠ **Do not copy or translate NORDIC code** (NORDIC_Raw, NIFTI_NORDIC, NORDIC-python).
  It is University of Minnesota copyright, research use only, no redistribution, and
  NORDIC is patented (US 10,768,260). EDDEN's `NIFTI_COMP_to_REAL.m` is NORDIC-derived,
  by Steen Moeller, with no license. Treat it the same. The recipe here comes from the
  EDDEN paper's description, the filter from Sprenger et al. 2017.
- The `"dk"` weights are Sprenger's Opt5, which the paper gives only as a color map
  (Fig. 3). They were read off it by color and sum to 1.006.
- The volumes are stabilized in the worker pool, chunks of volumes per task, and the
  result does not depend on the number of workers. On EDDEN 0.9 mm the step took about
  125 s on one core before and 22 s on 16 workers after (2026-11-05), the phasor now
  taken as `conj(s) / |s|` without `angle`.
- **The phase format is declared, never guessed** (`phase_range`, `--phase-range LOW HIGH`,
  radians by default). Only the width of the turn matters. A phase wider than it raises,
  narrower than half warns. Until 2026-11-09 `phase_scale` took the data's range as one
  turn, so radians that were unwrapped or offset came out scaled wrong without a word.
- NaN and inf are zeroed before the smoothing: a zero adds nothing to the complex sum
  whose angle is the smooth phase. Left in, one NaN left its 5 x 5 neighbours unturned.
  They stay NaN in the output.
- The phase is removed with a smooth estimate, which leaves neighbours' noise correlation
  in the real part close to the complex one. The estimator measures it from the real
  part anyway.

## Complex denoising

Both denoisers take complex64 data and return it. The Gram is `X^H X`, the rebuild uses
`V^H`, and LR's `q_k` takes the real part of the conjugate inner products.

- ⚠ **σ is one channel's level.** The MP fit sees the complex entry variance, `2σ²`, so
  σ is divided by √2 after the fit. Do not drop it.
- The noise model is estimated on the real part (`provenance["data"]`). The estimator
  stays real throughout.
- After phase stabilization, real-part denoising beat complex on EDDEN 2 mm run-01 (5³,
  against the mean of runs 02-06): RMSE 273.6 against 277.7 at b1000. The imaginary part
  adds noise and raises the MP edge, rank 12 against 20. Complex is a supported input,
  not the recommended path.

## Gotchas

- **Every full-volume pass and the loop's split run in worker processes by default**
  (`pool.py`, joblib's loky workers, `workers=None` = one per physical core, `workers=1`
  = in-process).
  Measured 2026-11-03 on the EDDEN crop at 297 volumes and 7³, 16 cores: estimate 17 s
  (6 loop rounds), camppca denoise 8 s, MPPCA denoise 5 s. main with P2S and JAX took
  74 s for the estimate.
  - Processes, not threads: scipy's `eigh` holds the GIL, and OpenBLAS queues callers
    from several threads (5.9x at 8 threads, 2.7x at 16, called through ctypes). MKL
    scales across threads but would be a large x86-only dependency, and only ties.
  - Workers run OpenBLAS with one thread and are not pinned; pinning gained nothing.
  - The volume and the relative noise map reach the workers as joblib memory maps in the temp
    directory, once per pass (the loop keeps one pool for all its rounds). joblib's own
    default, `/dev/shm`, pinned the whole volume in RAM for no speed gain.
  - Each task returns only its voxels. The gather builds each box by index arithmetic,
    so no padded copy exists. In mean mode a task returns its sums over its
    box (the x-slices its patches reach) and the main process adds them in chunk order,
    over a fixed 32 chunks, so the result does not depend on the number of workers.
  - A pooled pass is the serial pass bit for bit. So is the loop: its sampled patches go
    in 32 fixed chunks (`LOOP_CHUNKS`), pooled in chunk order. When the chunks followed
    the worker count, a near-tied rank moved `rho` by about 1e-4.
  - EDDEN 0.9 mm (228 x 228 x 132, 199 volumes, 2.0M brain voxels, 7³), whole camppca
    pipeline: 298 s (estimate 103 s, of which the level pass 90 s; denoise 195 s).
    Peak system RAM above idle, image included: loading 5.5 GB, estimate 8.4 GB, denoise
    12.8 GB. The main process holds the input and the output, 5.5 GB each.
  - What the 17 s is: the level pass 3 s, six splits at 1.0 s, six fits at 1.2 s. A fit
    is 3 to 4 preconditioned steps of two bulk maps each, plus 0.25 s to build the
    preconditioner. The splits are slower in the pool than alone, about 1.5 times, from
    shared memory bandwidth and clock.
  - ⚠ OpenBLAS in the main process starts one thread per hardware thread, 32 on 16
    cores. Held to 16 threads, a bulk map took 24 ms against 49 ms and the estimate
    18 s against 21 s, before the split moved to float32. Not set in the code yet.

- **Tests run one worker and one BLAS thread each**, set in `tests/conftest.py`, and
  pytest runs them side by side (`-n auto` in `pyproject.toml`): 22 s against 85 s with a
  pool per test. Serial with one BLAS thread the suite takes 103 s on one core. A serial
  test with OpenBLAS at its default took 158 s, from a thread per hardware thread on
  small matrices. `test_pool.py` tests the pool with explicit worker counts.

- **The denoiser computes only the eigenvectors each patch keeps** (`spectrum.py`,
  `_whiten_split(kept_only=True)`). One tridiagonal reduction per Gram, all eigenvalues
  from it, the MP cut, then the top P vectors by inverse iteration, mapped back by
  `ormtr` through ctypes, since scipy has no wrapper. P is a median 9 of 199 on EDDEN
  0.9 mm. The whole-image denoise went from 673 s to 497 s (2026-11-09). A patch keeping
  more than a quarter of its components gets the full `eigh`: inverse iteration fails on
  a large cluster of equal eigenvalues, as in noiseless data. The loop needs the bulk
  vectors and complex data has no such route yet, so both use the full `eigh`.
  ⚠ Test references take the engine's own vectors. At a cut near float32 rounding any
  two solvers differ by up to 0.1 on the test volumes, and both are that far from
  float64. 32 workers on the 16-core host gained only 5%.

- **Exp2 degenerates at small N.** Its width normalizer collapses as retained rank
  approaches `min(P, N)`. At 125 voxels, 34 volumes the noise estimate blew up (52% vs a
  true 5%). Keep ≳ 50 volumes in the pooled patch. `estimate_noise_model` warns below 50
  (`MIN_VOLUMES`).
