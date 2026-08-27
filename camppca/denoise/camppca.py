"""Covariance-aware MP-PCA -- THE METHOD.

Whitens each patch column with its own shell's C_s^(-1/2), runs ONE Marchenko-Pastur fit
on the pooled patch, and unwhitens. Standard MP-PCA assumes the patch noise is i.i.d.;
here it is Sigma = sigma^2 D C D, so the fit happens where that assumption holds.

    Y[:, n] = signal + eps_n,   Cov(eps_n) = sigma_{s(n)}^2 C_{s(n)}
    Yw[:, n] = C_{s(n)}^(-1/2) Y[:, n] / sigma_{s(n)}      ->  Cov = I  per column

BOTH corrections are needed. Whitening with patterns of different amplitude gives
C_s^(-1/2) different gains, trading spatial correlation for column heteroscedasticity
unless sigma_s is divided out too.

WHY THIS IS NOT A WRAPPER AROUND mppca.py. Plain MP-PCA takes the centre spectrum as a
literal slice of the patch and projects only that row back. Whitening mixes voxels, so
the centre voxel becomes a dense combination of all P whitened rows:

    xhat_c = (e_c^T C^(1/2)) Yw Pi

Only the CENTRE ROW of C^(1/2) is needed, which is what keeps unwhitening near-free. With
a per-column whitener the projector mixes shells, so S candidate centre rows are carried --
one per shell -- and column n's own shell is selected. Cost is S x the fast path, not P x.

WHITENER FACTORIZATION. The paper applies W = (DCD)^(-1/2) in one symmetric step; this
code divides rows by d then applies C^(-1/2), i.e. W = C^(-1/2) D^(-1). W^T W = (DCD)^(-1)
either way, so the Gram, spectrum, threshold, retained subspace and reconstruction are
identical. Do NOT "fix" one to match the other.

Ported verbatim from ``src/warp_denoise/whitened.py`` (see reference/), so it keeps the
upstream spelling. Notation mapping to CLAUDE.md:

    Rs         -> Cs, the per-shell patch correlation   kernel   -> patch
    row_scale  -> d, the relative level field           P        -> paper M
    col_scale  -> the per-shell sigma ratios            Mp, Np   -> min(P,N), max(P,N)

``build_R`` and ``kernel_offsets`` are NOT here: C is derived by
``NoiseModel.patch_correlation`` (it belongs to the model) and the patch offsets live in
``camppca.patch`` (both halves need them). The clean names are presented by
``camppca.denoise``; this module is the ported core.

PATCH ORDER. C is built from explicit (dx, dy, dz) lags in the engine's C order (z
fastest), not as a Kronecker product, so it is Toeplitz in the true relative lags with no
wraparound.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..patch import patch_offsets as kernel_offsets
from .mppca import _device_gather_budget_bytes, _mp_threshold_jax


# patch correlation helpers
# ----------------------------------------------------------------------------------------------
def sqrtm_psd(A, p=0.5, floor=1e-10):
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor * max(w.max(), 1e-30))
    B = (V * w ** p) @ V.T
    return 0.5 * (B + B.T)


# ----------------------------------------------------------------------------------------------
# tile core
# ----------------------------------------------------------------------------------------------
def _tile_percol(patches, Wcol, Sc, group, scale, Mp: int, Np: int, algo: str):
    """``patches`` (B,P,N); ``Wcol`` (S,P,P) = R_s^(-1/2); ``Sc`` (S,P) = centre row of
    R_s^(1/2); ``group`` (N,) shell index per column; ``scale`` (N,) = sigma_{s(n)}."""
    N = patches.shape[2]
    if Wcol.shape[0] == 1:
        # single kernel: a plain matmul.  The general per-column einsum below lets XLA
        # materialise a (B,N,P,P)-scale intermediate, which at P=343, N=199 is ~6 GB per tile.
        Xw = jnp.einsum("pq,bqn->bpn", Wcol[0], patches) / scale[None, None, :]
    else:
        Xw = jnp.einsum("bqn,npq->bpn", patches, Wcol[group]) / scale[None, None, :]
    G = jnp.einsum("bpn,bpm->bnm", Xw, Xw)
    evals, V = jnp.linalg.eigh(G)
    vals = jnp.maximum(evals[:, ::-1], 0.0)
    Vd = V[:, :, ::-1]
    t, sigma = _mp_threshold_jax(vals, Mp, Np, algo)
    fac = (jnp.arange(N)[None, :] < t[:, None]).astype(patches.dtype)
    # S candidate centre rows: T[b,g,m] = e_c^T R_g^(1/2) Xw[b,:,m]
    T = jnp.einsum("gp,bpm->bgm", Sc, Xw)                          # (B,S,N)
    coords = jnp.einsum("bgm,bmk->bgk", T, Vd)
    proj = jnp.einsum("bgk,bnk->bgn", coords * fac[:, None, :], Vd)
    idx = jnp.broadcast_to(group[None, None, :], (proj.shape[0], 1, N))
    xhat = jnp.take_along_axis(proj, idx, axis=1)[:, 0, :]         # pick column n's own shell
    return xhat * scale[None, :], sigma, t.astype(patches.dtype)


_tile_percol_jit = partial(jax.jit, static_argnums=(5, 6, 7))(_tile_percol)

def denoise_white_percol(dwi, mask, Rs, group, col_scale=None, kernel=(5, 5, 5),
                         algo="cordero-grande", tile=256, row_scale=None,
                         row_scale_norm="global"):
    """MP-PCA on the pooled patch with a per-column spatial whitener.

    ``Rs``        (S,P,P) patch correlation matrices, unit diagonal, P = prod(kernel).
    ``group``     (N,) index into ``Rs`` for every volume (e.g. its shell).
    ``col_scale`` (N,) per-column noise scale sigma_{s(n)}; ``None`` -> all ones.  Only ratios
                  matter; pass sigma_s / mean(sigma) to keep sigma-hat in the input's units.
    ``row_scale`` (X,Y,Z) RELATIVE noise level field g(x) (g-factor-like).  The patch model
                  becomes Sigma = sigma^2 D R D with D = diag(g at the patch voxels): each
                  voxel row is divided by its g before the R^(-1/2) whitening (exact for DRD),
                  and the centre row is rescaled by g(centre) on the way back.  Only ratios
                  matter -- the per-patch MP fit sets the absolute level -- so any global
                  scaling of g cancels.  What it adds over per-patch sigma is the WITHIN-patch
                  heteroscedasticity, which widens the noise bulk exactly like off-diagonal
                  colour does.  The returned sigma map is in g-normalised units; multiply by
                  g(centre) for data units.

    Returns ``(denoised (X,Y,Z,N), sigma (X,Y,Z), rank (X,Y,Z))``, centre-voxel only (no overlap
    averaging), matching the rest of the engine.
    """
    dwi = np.asarray(dwi, np.float32)
    sx, sy, sz, N = dwi.shape
    kernel = np.asarray(kernel, int)
    kk = kernel // 2
    P = int(np.prod(kernel))
    center = P // 2
    Mp, Np = min(P, N), max(P, N)

    Rs = np.asarray(Rs, np.float64)
    if Rs.ndim == 2:
        Rs = Rs[None]
    assert Rs.shape[1:] == (P, P), f"Rs must be (S,{P},{P}), got {Rs.shape}"
    group = np.asarray(group, int)
    assert group.shape == (N,) and group.min() >= 0 and group.max() < len(Rs)
    scale = np.ones(N) if col_scale is None else np.asarray(col_scale, float)
    assert scale.shape == (N,) and scale.min() > 0

    Wm = np.stack([sqrtm_psd(R, -0.5) for R in Rs])                # (S,P,P)
    Sc = np.stack([sqrtm_psd(R, 0.5)[center] for R in Rs])         # (S,P) centre rows only
    Wcol = jnp.asarray(Wm, jnp.float32)                            # (S,P,P); gathered per tile
    Scj = jnp.asarray(Sc, jnp.float32)
    gj = jnp.asarray(group, jnp.int32)
    sj = jnp.asarray(scale, jnp.float32)

    padded = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    Xp, Yp, Zp = padded.shape[:3]
    if row_scale is not None:
        g = np.asarray(row_scale, np.float32)
        good = np.isfinite(g) & (g > 0) & mask
        g = g / np.median(g[good])                       # scale-neutral by construction
        g = np.clip(np.where(np.isfinite(g) & (g > 0), g, 1.0), 0.2, 5.0)
        gpad = np.pad(g, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2])), mode="wrap")
    offs = kernel_offsets(kernel)
    xs, ys, zs = np.where(mask)
    tile = min(int(tile), max(1, int(5.0e7 / (N * N))))
    # 2x safety factor: the budget is computed from the JAX pool's own stats, which do not see
    # other processes on a shared GPU -- a 6 GB padded volume that "fits" a 12 GB pool still
    # OOMs when the desktop holds 3 GB.  Host gather is the slow-but-sure fallback.
    gpu = (2 * Xp * Yp * Zp * N * 4) < _device_gather_budget_bytes()
    if gpu:
        pf = jnp.asarray(padded.reshape(-1, N))
        doff = jnp.asarray((offs[:, 0] * Yp * Zp + offs[:, 1] * Zp + offs[:, 2]).astype(np.int32))
        base = jnp.asarray((xs * Yp * Zp + ys * Zp + zs).astype(np.int32))
        if row_scale is not None:
            gf = jnp.asarray(gpad.ravel())

    out = dwi.copy()
    sig = np.zeros((sx, sy, sz), np.float32)
    rnk = np.zeros((sx, sy, sz), np.float32)
    for s in range(0, xs.size, tile):
        ix, iy, iz = xs[s:s + tile], ys[s:s + tile], zs[s:s + tile]
        if gpu:
            pat = pf[base[s:s + tile][:, None] + doff[None, :]]
            gp = gf[base[s:s + tile][:, None] + doff[None, :]] if row_scale is not None else None
        else:
            hp = np.empty((ix.size, P, N), np.float32)
            for vi, (dx, dy, dz) in enumerate(offs):
                hp[:, vi, :] = padded[ix + dx, iy + dy, iz + dz, :]
            pat = jnp.asarray(hp)
            gp = jnp.asarray(gpad[ix[:, None] + offs[None, :, 0],
                                  iy[:, None] + offs[None, :, 1],
                                  iz[:, None] + offs[None, :, 2]]) if row_scale is not None \
                else None
        if gp is not None:
            if row_scale_norm == "patch":
                # normalise g PER PATCH: makes explicit that only within-patch ratios enter.
                # Must be output-identical to "global" -- the patch-common scalar is absorbed
                # by the per-patch MP sigma-hat (scale equivariance).  Kept as a sanity switch.
                gp = gp / jnp.median(gp, axis=1, keepdims=True)
            pat = pat / gp[:, :, None]                   # D^(-1): homoscedastic rows
        xh, sg, nn = _tile_percol_jit(pat, Wcol, Scj, gj, sj, Mp, Np, algo)
        if gp is not None:
            xh = xh * gp[:, center, None]                # back to data units at the centre
        out[ix, iy, iz, :] = np.asarray(xh)
        sig[ix, iy, iz] = np.asarray(sg)
        rnk[ix, iy, iz] = np.asarray(nn)
    return out, sig, rnk


# ----------------------------------------------------------------------------------------------
# reference (slow, float64) path used by the self-test
# ----------------------------------------------------------------------------------------------
def slow_patch_percol(Y, Rs, group, scale, algo="cordero-grande"):
    """Literal path: whiten each column, denoise the FULL patch, unwhiten, take the centre row."""
    P, N = Y.shape
    W = [sqrtm_psd(R, -0.5) for R in Rs]
    S = [sqrtm_psd(R, 0.5) for R in Rs]
    Yw = np.stack([W[group[n]] @ Y[:, n] / scale[n] for n in range(N)], 1)
    ev, V = np.linalg.eigh(Yw.T @ Yw)
    vals = np.maximum(ev[::-1], 0.0)
    Vd = V[:, ::-1]
    t, sigma = _mp_threshold_jax(jnp.asarray(vals[None, :]), min(P, N), max(P, N), algo)
    t = int(np.asarray(t)[0])
    Xw = Yw @ Vd[:, :t] @ Vd[:, :t].T
    X = np.stack([S[group[n]] @ Xw[:, n] * scale[n] for n in range(N)], 1)
    return X[P // 2], t, float(np.asarray(sigma)[0])


def selftest(seed=0, kernel=(5, 5, 3), N=40, shape=(24, 24, 8), npatch=24,
             global_whitener=None, verbose=True):
    """Three reductions: W = I == plain engine; S = 1 == global whitener; S > 1 == literal.

    ``global_whitener`` is the S = 1 reference implementation
    (``denoise_white(dwi, mask, W, kernel)``).  It lives outside the package -- pass the
    test-only port to run that reduction; omit it and the reduction reports "skipped"
    rather than silently vanishing.

    Returns a dict of METRICS, not pass/fail: the caller supplies the thresholds.
    """
    from .mppca import denoise_adaptive
    from ..noise_model import NoiseModel
    rng = np.random.default_rng(seed)
    P = int(np.prod(kernel))
    dwi = rng.standard_normal(shape + (N,)).astype(np.float32) * 30 + 100
    mask = np.zeros(shape, bool)
    mask[4:20:2, 4:20:2, 1:7] = True
    rep = {}

    def build_C(rho_table, lag):
        """Derive C through the public model path -- exercises the real seam."""
        model = NoiseModel(rho=rho_table[None], group=np.zeros(N, int),
                           d=np.ones(shape, np.float32), sigma_p2s=np.ones(shape, np.float32),
                           mask=mask, lag=lag)
        return model.patch_correlation(kernel)[0]

    # --- 1. identity whitener == denoise_adaptive(adaptive=False) -----------------------------
    ref, rs, rn = denoise_adaptive(dwi, mask, kernel=kernel, algorithm="cordero-grande",
                                   shrinkage="threshold", adaptive=False)
    o1, s1, n1 = denoise_white_percol(dwi, mask, np.eye(P)[None], np.zeros(N, int), kernel=kernel)
    sc = float(np.abs(ref[mask]).max())
    rep["identity_vs_engine"] = dict(signal_rel=float(np.abs(o1[mask] - ref[mask]).max()) / sc,
                                     sigma=float(np.abs(s1[mask] - rs[mask]).max()),
                                     rank=float(np.abs(n1[mask] - rn[mask]).max()))

    # --- 2. S = 1 coloured == the global-whitener implementation -------------------------------
    lag = int(max(kernel)) - 1          # derived, not hardcoded: build_C bounds-checks it
    rho = np.zeros((2 * lag + 1, 2 * lag + 1, lag + 1))
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            rho[i + lag, j + lag, 0] = 0.6 ** abs(i) * 0.35 ** abs(j)
    C1 = build_C(rho, lag)
    if global_whitener is None:
        rep["global_vs_percol_S1"] = "skipped: no global_whitener supplied"
    else:
        o2, s2, n2 = global_whitener(dwi, mask, sqrtm_psd(C1, -0.5).astype(np.float32), kernel)
        o2b, s2b, n2b = denoise_white_percol(dwi, mask, C1[None], np.zeros(N, int), kernel=kernel)
        rep["global_vs_percol_S1"] = dict(
            signal_rel=float(np.abs(o2[mask] - o2b[mask]).max()) / float(np.abs(o2[mask]).max()),
            sigma=float(np.abs(s2[mask] - s2b[mask]).max()),
            rank=float(np.abs(n2[mask] - n2b[mask]).max()))

    # --- 3. S = 2 with different sigma == the literal float64 path -----------------------------
    rho2 = np.zeros_like(rho)
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            rho2[i + lag, j + lag, 0] = 0.25 ** abs(i) * 0.55 ** abs(j)
    Rs = np.stack([C1, build_C(rho2, lag)])
    group = (np.arange(N) % 2).astype(int)
    scale = np.where(group == 0, 0.8, 1.3)
    o3, _, n3 = denoise_white_percol(dwi, mask, Rs, group, scale, kernel=kernel)
    kk = np.asarray(kernel) // 2
    pad = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    offs = kernel_offsets(kernel)
    vox = np.argwhere(mask)
    vox = vox[rng.permutation(len(vox))[:npatch]]
    dif, agree = [], 0
    for (x, y, z) in vox:
        Y = np.stack([pad[x + dx, y + dy, z + dz, :] for dx, dy, dz in offs]).astype(np.float64)
        lit, t, _ = slow_patch_percol(Y, Rs, group, scale)
        if t == int(n3[x, y, z]):
            agree += 1
            dif.append(np.abs(lit - o3[x, y, z]).max() / max(np.abs(lit).max(), 1e-9))
    rep["percol_vs_literal"] = dict(rank_agree=agree, rank_total=int(len(vox)),
                                    max_rel_diff=float(max(dif)) if dif else float("nan"))
    if verbose:
        for k, v in rep.items():
            print(f"  {k:24s} {v}")
    return rep
