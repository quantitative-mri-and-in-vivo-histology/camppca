# camppca — scope and notation

**`camppca` — covariance-aware MP-PCA.** The implementation of the denoising algorithm
described in the manuscript at
`~/gdrive/UKE/manuscripts/covar_denoising/initial/`, which cites this repository in its Data
and Code Availability section. MP-PCA denoising for diffusion MRI that accounts for
spatially correlated noise.

⚠ **The name is `camppca`** — lower case, no hyphen, settled 2026-08-28 to match the package
and the paper. Not CAMP-PCA, not CAMPPCA. Expanded as *covariance-aware MP-PCA*.

Standard MP-PCA assumes white noise. Real EPI reconstruction (GRAPPA/SENSE, partial
Fourier, apodization) correlates noise across neighboring voxels, so MP-PCA
under-estimates the noise level and under-denoises. camppca estimates the spatial noise
covariance **from the data itself** (no noise-only scan), then denoises in the whitened
domain where the Marchenko–Pastur assumption actually holds.

Arrays in, arrays out. `nibabel` only at the I/O boundary.

Conventions for user-facing text: **MP-PCA** (hyphenated), American spelling.

## Scope

**In:** noise covariance estimation from P2S residuals; the noise-level field; whitened
MP-PCA (whiten → MP-PCA → unwhiten); the CLI around them.

**Out:** the warp/alignment lane (reusing FSL eddy displacement fields to denoise in a
common anatomical frame) — a separate project. The research repo this is ported from is
called `warp_denoise` for historical reasons; nothing warp-related comes with it.

Being ported from `~/workspace/cdmri_26_adaptive_denoising` @ `ec08de0`. Verbatim source
copies live in `reference/` (gitignored scratch). Brief: `docs/local/port_core_brief.md`.
File:line citations below refer to those **reference** files, not to ported modules.

## The noise model

The patch noise covariance factorizes into a **level** that varies in space and a
**correlation pattern** that does not:

    Sigma = sigma^2 * D * C * D          D = diag(d), d the relative level field

`rho` (the lag table) and `d` are the noise model. `C` is derived from `rho` plus a
patch shape, so it is computed on demand, never stored.

⚠ `D` holds **levels, not variances** — hence `D C D`, not `D^(1/2) C D^(1/2)`.

## Notation

| object | paper | code | notes |
|---|---|---|---|
| patch matrix, rows = voxels | X (M×N) | — | |
| unit-diagonal patch correlation | C | `C` | **not** `R` |
| lag correlation table | ρ(h) | `rho` | `rho[dx+lag, dy+lag, abs(dz)]`, 1.0 at zero lag |
| patch noise covariance | Σ = σ²DCD | — | **paper only** — never materialized in code |
| whitener | W = (DCD)^(-1/2) | `whitener` + row division by `d` | **σ-free by design** |
| relative level field | d(x) | `d` | **not** `g` — that is the g-factor |
| diagonal level matrix | D = diag(d) | `D` | |
| true noise level | σ | — | |
| MP estimate | σ̂ | `sigma_hat` | hats are for estimates only |
| P2S level (ratios only) | σ_P2S | `sigma_p2s` | `d` = `sigma_p2s` pooled over a 3³ box |
| retained rank | P | `rank` | ⚠ see the P collision below |
| aspect ratio | γ = N/M | — | |
| box shape, e.g. (5,5,5) | — | `patch` | a **denoiser** knob, not a noise property |
| max lag covered by `rho` | — | `lag` / `LAG` | |

⚠ **The whitener is σ-free on purpose.** `W = (DCD)^(-1/2)`, *not* `Σ^(-1/2)`. Whitening
by `Σ^(-1/2)` would divide the noise level out, MP-PCA would estimate σ̂ = 1, and the
"σ is an output, not an input" claim collapses. `W` normalizes shape only; the level
stays in the data for the per-patch MP fit to find.

**Banned words.** All three are silent-corruption hazards:
- **`kernel`** — upstream it means the patch shape *and* the lag table. Use `patch` and
  `rho`. In the manuscript `kernel` means `rho`; the ban only holds if you know that.
- **`sigma`** unqualified — `sigma_p2s`/`d` go in, `sigma_hat` comes out.
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

⚠ **`P` still means two different things across the two documents.** Paper `P` = retained
rank; the reference code's `P` = voxels in a patch (now `n_vox`).

Worked example: the story doc's "M = 125, N = 34" (paper convention: 125 voxels,
34 volumes) lands in code as `Mp = 34`, `Np = 125`. "Rank → Mp" means rank approaching
the 34 volumes.

## The seam

    estimator (source-specific)          engine (source-agnostic)
    ---------------------------          ------------------------
    P2S residuals -> rho, d      ---->    C -> whiten -> MP-PCA -> unwhiten

The two halves meet at **the noise model** (`rho` + `d`). A follow-up paper estimates
`rho` from the alignment/interpolation operator and feeds the same engine, so the engine
must never assume `rho` came from P2S.

P2S is numpy/scipy. The engine is JAX, and so is the loop kernel, which reuses the
engine's own split. The old rule that the estimator never imports JAX was dropped
2026-10-03 with the loop.

### The loop kernel

`estimate_noise_model(kernel="loop")`, the default, starts from the P2S `rho` and refines
it until it explains the MP-PCA bulk of 3000 sampled patches at once
(`noise_model/loop.py`). It fits to the denoiser's own split: `algo` and `noise_field`
are passed to `estimate_noise_model`, recorded in provenance, and a mismatch at denoising
warns.
Every volume enters, b0s included, because the denoiser whitens them all. The b0
exclusion above is for the P2S fit only. `kernel="p2s"` skips the loop.

⚠ **The loop runs in float64** (`jax.enable_x64`). Its Richardson solve leaves the
patch-mean direction empty, so each round's `C` is near singular there. In float32 the
amplified direction's rounding error floods the bulk and the ranks hit the maximum
within one round. Measured: float64 and float32-with-an-eigenvalue-floor agree to 0.001
in U, so this is purely numerical. The one exception is the final least squares: its
design is built in float32 from the float64 factors (same table within 1e-4 at 7³, the
build ~20x faster on a GPU).

### Two whiteners, one result

The paper applies `W = (DCD)^(-1/2)` in one symmetric step. The code splits it: divide
rows by `d`, then apply `C^(-1/2)` (`whitened.py:166`, `:200-206`), i.e.
`W = C^(-1/2) D^(-1)`. Different matrices, but `WᵀW = (DCD)^(-1)` either way, so the
Gram `Xᵀ(DCD)^(-1)X`, its spectrum, the MP threshold, the retained subspace and the
reconstruction are all identical. **Do not "fix" one to match the other.**

## Layout

    camppca/
      noise_model/
        __init__.py        NoiseModel, estimate_noise_model() -> NoiseModel
        p2s.py             Patch2Self: residuals, rho, sigma_p2s        numpy
        masks.py           brain mask, stable region                    numpy
        loop.py            the loop kernel, refines rho from P2S        JAX, float64
      denoiser/
        __init__.py        denoise_camppca(), denoise_with_noise_model()
        camppca.py         covariance-aware MP-PCA — the method
        mppca.py           standard white-noise MP-PCA
        mp_threshold.py    mp_threshold() — the Marchenko-Pastur cutoff rule
        gather.py          patch extraction and the device-memory decision
        result.py          DenoiseResult, what both denoisers return
      patch.py             patch geometry — both halves of the seam need it
      io.py                nibabel edge
      cli.py               camppca denoise | estimate
      __init__.py          public names; XLA env var set before jax loads

One subpackage per half of the seam. The package name carries the noun, so files inside
do not repeat it, and each `__init__.py` is that half's public face.

**The public API is flat.** Everything a user calls is `camppca.<name>`:
`denoise_camppca`, `denoise_mppca`, `denoise_with_noise_model`, `estimate_noise_model`,
`brain_mask`, `NoiseModel`, `DenoiseResult`. Nothing public needs a `camppca.denoiser.`
or `camppca.noise_model.` prefix.

- **Never `estimate` or `model` bare** — they do not say *what* is estimated or modeled.
  It is always the **noise model**: `estimate_noise_model`, `denoise_with_noise_model`.
  This deviates from the port brief's frozen `estimate_noise` / `denoise_with_model`,
  deliberately.
- `mp_threshold.py`, `gather.py` and `result.py` sit in `denoiser/`. `patch.py` sits
  above it because P2S needs it too.
- `loop.py` imports denoiser internals, and `denoiser/__init__.py` imports
  `noise_model`. So `estimate_noise_model` imports `loop` inside the function, or the
  import is circular.
- `NoiseModel` lives in `noise_model/__init__.py`, not with the P2S math: a future
  alignment-based estimator builds the same object and must not import P2S to get it.
- Flat, with `noise_model_p2s.py`-style prefixes, is the same structure with the noun
  repeated in every filename. Do not "simplify" back to it.

## Method invariants

Breaking any of these corrupts results without an error:

- `LAG = max(patch[0], patch[1]) - 1` in-plane, and `dzmax = patch[2] - 1`. Smaller, and
  lag lookup wraps around. `dzmax` used to equal `LAG`, which crashed a one-slice patch
  on a thin volume.
- `C` must have smallest eigenvalue ≥ 1e-3 (`MIN_EIGENVALUE`). A table valid for one
  patch can be indefinite for a larger one, and whitening by a near-singular `C` sends
  σ̂ to ~1e-7. `patch_correlation` raises. `_build_C` still projects silently, because
  the loop needs that each round.
- Patch flattening is **C-order, z fastest**; `rho[dx+lag, dy+lag, abs(dz)]`. Paired —
  the two orders must agree.
- `rho` is a correlation (unit at zero lag): **pattern only**. The absolute level comes
  from the per-patch MP fit. Only *ratios* of `d` matter.
- **`d` is defined up to scale.** Rows are divided by it and the centre voxel multiplied
  back, so a constant cancels in the denoised image, the rank and σ̂ in data units (×0.25
  is bit-identical; a generic factor flips near-tied MP cuts by float32 rounding).
  Outside the mask, which edge patches include, `d` is its median over the mask.
- **One global `rho`.** There is no per-shell pattern and no per-shell level. `rho` is fit
  once across every diffusion volume, and b0s stay out of that fit because their residual
  level does not represent diffusion-weighted noise.
- **Centre-voxel only** — no overlap averaging. Intentional; matches the MP theory the
  noise estimate rests on. Users coming from `dwidenoise` may expect averaging.
- **P2S radius 0 is mandatory**, not a simplification. A radius>0 predictor regresses on
  spatial neighbors, which biases the very spatial correlation being estimated.
- **Hard threshold by default, shrinkage opt-in** (`shrink="hard" | "gd" | "lr"`,
  `denoiser/shrink.py`). ⚠ The default is NOT hard because shrinkage would break
  exactness — it would not. The unwhitening identity `eᶜᵀ C^(1/2) · (Yw V diag(f) Vᵀ)` is
  plain associativity and never used idempotency of the projector, so it holds for ANY
  `f` (measured at 4e-16 for hard threshold, a GD profile, and a random profile). The
  reasons: a hard threshold needs only the edge LOCATION, while shrinkage scales every
  retained component by a continuous function of `sigma_hat` — and `sigma_hat` is fragile
  at small N (see Exp2 below). `"gd"` also optimizes in the WHITENED metric while the
  output lands in data units. `"lr"` (Leeb–Romanov) corrects for that, but its guarantee
  is for the whole patch at large sizes, and camppca keeps only the centre voxel of a
  small patch. Both shrinkers keep the MP rank and change only the weights.

## API rules

- **`sigma_hat` is returned in data units.** The engine rescales the denoised image by
  `d` at the center voxel but *not* `sigma_hat`, so the API must apply that multiply by
  default. σ̂ is the headline metric — the scale of `d` must not leak into the reported
  number.
- `_effective_d` in `denoiser/__init__.py` is the **only** place `d` is guarded (bad or
  outside-mask voxels get the median). The engine takes the field as given.

## Gotchas

- **P2S needs V >> N.** It fits N-1 predictors across the mask voxels, so in-sample
  overfitting deflates BOTH `rho` and `sigma_p2s` when the mask is small. Measured at
  N = 60 on synthetic data with a known kernel:

      V = 576    p/V = 0.10    rho_x1 0.589 / 0.665    sigma 7.95 / 10
      V = 2904   p/V = 0.020   rho_x1 0.649 / 0.665    sigma 9.53 / 10
      V = 11560  p/V = 0.005   rho_x1 0.657 / 0.665    sigma 9.82 / 10

  Real data sits near p/V = 1e-4, so it never bites in practice -- but a small synthetic
  test reads as a broken estimator if you forget.
- **Exp2 degenerates at small N.** Its width normalizer collapses as retained rank
  approaches `min(P, N)`. At 125 voxels, 34 volumes the noise estimate blew up (52% vs a
  true 5%). Keep ≳ 50 volumes in the pooled patch.
