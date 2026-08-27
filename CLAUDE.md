# camppca — scope and notation

**CAMP-PCA: Covariance-Aware Marchenko–Pastur PCA.** MP-PCA denoising for diffusion MRI
that accounts for spatially correlated noise.

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
| unit-diagonal patch correlation | C | `C`, stacked `Cs` | **not** `R` |
| lag correlation table | ρ(h) | `rho` | `rho[dx+lag, dy+lag, abs(dz)]`, 1.0 at zero lag |
| patch noise covariance | Σ = σ²DCD | — | **paper only** — never materialized in code |
| whitener | W = (DCD)^(-1/2) | `Wcol` + row division by `d` | **σ-free by design** |
| relative level field | d(x) | `d` | **not** `g` — that is the g-factor |
| diagonal level matrix | D = diag(d) | `D` | |
| true noise level | σ | — | |
| MP estimate | σ̂ | `sigma_hat` | hats are for estimates only |
| P2S level (ratios only) | σ_P2S | `sigma_p2s` | `d = sigma_p2s / median(sigma_p2s)` |
| retained rank | P | `rank` | ⚠ see the P collision below |
| aspect ratio | γ = N/M | — | |
| box shape, e.g. (5,5,5) | — | `patch` | a **denoiser** knob, not a noise property |
| max lag covered by `rho` | — | `lag` / `LAG` | |
| volume → shell index | — | `group` | |

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
| — | `S` | shells |

In code these are spelled `n_vox`, `n_vols`, `n_mask`, `n_shells` — the bare letters were
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

The estimator half is pure numpy/scipy. The engine is JAX.

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
      denoise/
        __init__.py        denoise(), denoise_with_noise_model() -> DenoiseResult
        camppca.py         covariance-aware MP-PCA — the method         jax
        mppca.py           plain white MP-PCA — baseline arm + the shared MP threshold
      io.py                nibabel edge
      cli.py               camppca denoise | estimate
      __init__.py          public names; XLA env var set before jax loads

One subpackage per half of the seam. The package name carries the noun, so files inside
do not repeat it, and each `__init__.py` is that half's public face.

- **Never `estimate` or `model` bare** — they do not say *what* is estimated or modeled.
  It is always the **noise model**: `estimate_noise_model`, `denoise_with_noise_model`.
  This deviates from the port brief's frozen `estimate_noise` / `denoise_with_model`,
  deliberately.
- `denoise/camppca.py` is the method; `denoise/mppca.py` is a **reference/baseline arm**
  only. Say so in its module header — it is the one file not on the main path.
- `NoiseModel` lives in `noise_model/__init__.py`, not with the P2S math: a future
  alignment-based estimator builds the same object and must not import P2S to get it.
- Flat, with `noise_model_p2s.py`-style prefixes, is the same structure with the noun
  repeated in every filename. Do not "simplify" back to it.

## Method invariants

Breaking any of these corrupts results without an error:

- `LAG = max(patch) - 1`, and `dzmax = LAG`. Smaller, and lag lookup wraps around.
- Patch flattening is **C-order, z fastest**; `rho[dx+lag, dy+lag, abs(dz)]`. Paired —
  the two orders must agree.
- `rho` is a correlation (unit at zero lag): **pattern only**. The absolute level comes
  from the per-patch MP fit. Only *ratios* of `d` and the per-shell level matter.
- **`d` is normalized by the median**, in both the row-scale field and per-patch.
- Per-shell patterns are fit **within** each shell, never pooled across shells.
- **Centre-voxel only** — no overlap averaging. Intentional; matches the MP theory the
  noise estimate rests on. Users coming from `dwidenoise` may expect averaging.
- **P2S radius 0 is mandatory**, not a simplification. A radius>0 predictor regresses on
  spatial neighbors, which biases the very spatial correlation being estimated.
- **Hard threshold, no shrinkage.** ⚠ NOT because shrinkage would break exactness — it
  would not. The unwhitening identity `eᶜᵀ C^(1/2) · (Yw V diag(f) Vᵀ)` is plain
  associativity and never used idempotency of the projector, so it holds for ANY `f`
  (measured at 4e-16 for hard threshold, a GD profile, and a random profile). The real
  reasons: a hard threshold needs only the edge LOCATION, while shrinkage scales every
  retained component by a continuous function of `sigma_hat` — and `sigma_hat` is fragile
  at small N (see Exp2 below). Shrinkage would also optimize in the WHITENED metric while
  the output lands in data units, so it minimizes the wrong error. And the rule is the
  method, not a knob.

## API rules

- **`sigma_hat` is returned in data units.** The engine rescales the denoised image by
  `d` at the center voxel but *not* `sigma_hat` (`whitened.py:208` vs `:211`), so the API
  must apply that multiply by default. σ̂ is the headline metric — a normalization
  convention must not leak into the reported number. Normalized form is opt-in.
- Before porting, capture the reference `selftest()` output as a **golden baseline**.
  "Selftest green" only means something against the pre-port numbers.

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
