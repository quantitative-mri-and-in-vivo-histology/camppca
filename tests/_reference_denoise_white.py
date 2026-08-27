"""TEST-ONLY: the global-whitener (S = 1) reference implementation.

``camppca.denoiser.camppca.selftest`` reduction 2 checks the per-column engine against
this: with one shell, the per-column whitener must collapse exactly onto whitening the
whole patch once.

Copied from ``scripts/henriques/whitened_mppca.py`` (see reference/) -- the FUNCTIONS
only. That module cannot be imported: its import block mutates ``sys.path`` twice, calls
``warnings.filterwarnings("ignore")`` process-wide, and pulls in a model dict of hardcoded
``/workspace/data/*.npz`` paths.

Not shipped. This is a third copy of the padded-gather/tile loop (the package has two);
that duplication is upstream's, kept so the comparison stays literal.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from camppca.patch import patch_offsets as kernel_offsets
from camppca.denoiser.mppca import _device_gather_budget_bytes, _mp_threshold_jax


def _tile_white(patches, Wmat, center: int, Mp: int, Np: int, algo: str):
    Xw = jnp.einsum("pq,bqn->bpn", Wmat, patches)                  # whiten the VOXEL axis
    xc = patches[:, center, :]                                     # ORIGINAL centre row
    G = jnp.einsum("bpn,bpm->bnm", Xw, Xw)
    evals, V = jnp.linalg.eigh(G)
    vals = jnp.maximum(evals[:, ::-1], 0.0)
    Vd = V[:, :, ::-1]
    t, sigma = _mp_threshold_jax(vals, Mp, Np, algo)
    fac = (jnp.arange(patches.shape[2])[None, :] < t[:, None]).astype(patches.dtype)
    coords = jnp.einsum("bn,bnm->bm", xc, Vd)
    xhat = jnp.einsum("bm,bnm->bn", coords * fac, Vd)
    return xhat, sigma, t.astype(patches.dtype)


_tile_white_jit = partial(jax.jit, static_argnums=(2, 3, 4, 5))(_tile_white)

def denoise_white(dwi, mask, Wmat, kernel, algo="cordero-grande", tile=256):
    """Same tiling/gather as denoise_adaptive, plain box patches, voxel-axis whitening."""
    dwi = np.asarray(dwi, np.float32)
    sx, sy, sz, N = dwi.shape
    kernel = np.asarray(kernel, int)
    kk = kernel // 2
    P = int(np.prod(kernel))
    center = P // 2
    Mp, Np = min(P, N), max(P, N)
    padded = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    Xp, Yp, Zp = padded.shape[:3]
    offs = kernel_offsets(kernel)
    xs, ys, zs = np.where(mask)
    tile = min(int(tile), max(1, int(5.0e7 / (N * N))))
    gpu = (Xp * Yp * Zp * N * 4) < _device_gather_budget_bytes()
    if gpu:
        pf = jnp.asarray(padded.reshape(-1, N))
        doff = jnp.asarray((offs[:, 0] * Yp * Zp + offs[:, 1] * Zp + offs[:, 2]).astype(np.int32))
        base = jnp.asarray((xs * Yp * Zp + ys * Zp + zs).astype(np.int32))
    Wj = jnp.asarray(Wmat, jnp.float32)
    out = dwi.copy()
    sig = np.zeros((sx, sy, sz), np.float32)
    npr = np.zeros((sx, sy, sz), np.float32)
    for s in range(0, xs.size, tile):
        ix, iy, iz = xs[s:s + tile], ys[s:s + tile], zs[s:s + tile]
        if gpu:
            pat = pf[base[s:s + tile][:, None] + doff[None, :]]
        else:
            hp = np.empty((ix.size, P, N), np.float32)
            for vi, (dx, dy, dz) in enumerate(offs):
                hp[:, vi, :] = padded[ix + dx, iy + dy, iz + dz, :]
            pat = jnp.asarray(hp)
        xh, sg, nn = _tile_white_jit(pat, Wj, center, Mp, Np, algo)
        out[ix, iy, iz, :] = np.asarray(xh)
        sig[ix, iy, iz] = np.asarray(sg)
        npr[ix, iy, iz] = np.asarray(nn)
    return out, sig, npr


def slow_patch(Y, Wm, Sm, center, Mp, Np, algo):
    """Literal path: whiten, denoise the FULL patch, unwhiten, take the centre row."""
    Yw = Wm @ Y
    G = Yw.T @ Yw
    ev, V = np.linalg.eigh(G)
    vals = np.maximum(ev[::-1], 0.0)
    Vd = V[:, ::-1]
    t, _ = _mp_threshold_jax(jnp.asarray(vals[None, :]), Mp, Np, algo)
    t = int(np.asarray(t)[0])
    Xw = Yw @ Vd[:, :t] @ Vd[:, :t].T
    lit = (Sm @ Xw)[center]                                         # X = R^1/2 X_w
    cid = (Y @ Vd[:, :t] @ Vd[:, :t].T)[center]                     # the centre-row identity
    return lit, t, cid


# ----------------------------------------------------------------------------------------------
def acf3(F, mask, lags=(1, 2, 3)):
    """Directional ACF of a 3-D field over a mask, normalised by lag 0."""
    m = mask
    v = float((F[m] ** 2).mean())
    out = {}
    for d in lags:
        r = []
        for ax in range(3):
            a = np.take(F, np.arange(0, F.shape[ax] - d), ax)
            b = np.take(F, np.arange(d, F.shape[ax]), ax)
            ma = np.take(m, np.arange(0, m.shape[ax] - d), ax) & np.take(
                m, np.arange(d, m.shape[ax]), ax)
            r.append(float((a[ma] * b[ma]).mean() / v))
        out[d] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="henriques", choices=sorted(MODEL))
    ap.add_argument("--kernel", default="9,9,3")
    ap.add_argument("--algo", default="cordero-grande")
    ap.add_argument("--out", default="/workspace/data/henriques/out/wmppca/")
    ap.add_argument("--nsim", type=int, default=40)
    ap.add_argument("--nslow", type=int, default=40)
    ap.add_argument("--checks-only", action="store_true")
    a = ap.parse_args()
    kernel = tuple(int(v) for v in a.kernel.split(","))
    P = int(np.prod(kernel))
    import os
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    npz, key = MODEL[a.data]
    cfg = DATA[a.data]
    rho = rho_from_P(np.load(npz)[key])
    R = build_RP(rho, kernel)
    Wm, Sm, rep = sqrtm_pair(R)
    print(f"=== {a.data}: R_P from {key} of {npz.split('/')[-1]}, kernel {kernel} ({P} vox) ===")
    print(f"  rho: lag(1,0) {rho[1,0]:+.4f}  lag(0,1) {rho[0,1]:+.4f}  "
          f"lag(4,0) {rho[4,0]:+.4f}  lag(8,0) {rho[8,0]:+.4f}  lag(8,8) {rho[8,8]:+.4f}")
    print(f"  R_P: lam_min {rep['lam_min']:.6f}  lam_max {rep['lam_max']:.4f}  "
          f"cond {rep['cond']:.2f}  m_eff/m {rep['m_eff_over_m']:.4f}")

    # ---- check 1: W R W^T = I ----------------------------------------------------------------
    err = np.abs(Wm @ R @ Wm.T - np.eye(P)).max()
    print(f"\n[check 1] max |W R_P W^T - I| = {err:.3e}   "
          f"max |R^1/2 R^-1/2 - I| = {np.abs(Sm @ Wm - np.eye(P)).max():.3e}")

    img = nib.load(cfg["srcs"][0])
    mask = np.asarray(nib.load(cfg["mask"]).dataobj) > 0.5
    dwi = np.asarray(img.dataobj, np.float32)
    N = dwi.shape[3]
    Mp, Np = min(P, N), max(P, N)

    # ---- check 2: identity control == existing implementation --------------------------------
    print(f"\n[check 2] identity control vs denoise_adaptive (kernel {kernel}, {a.algo}, box)")
    sub = np.zeros_like(mask)
    ii = np.argwhere(mask)
    ii = ii[np.random.default_rng(0).permutation(len(ii))[:4000]]
    sub[tuple(ii.T)] = True
    ta = time.time()
    ref, rs, rn = denoise_adaptive(dwi, sub, kernel=kernel, algorithm=a.algo,
                                   shrinkage="threshold", adaptive=False)
    tb = time.time()
    idn, isg, inp = denoise_white(dwi, sub, np.eye(P, dtype=np.float32), kernel, a.algo)
    print(f"  max |identity - existing| : signal {np.abs(idn[sub]-ref[sub]).max():.3e}   "
          f"sigma {np.abs(isg[sub]-rs[sub]).max():.3e}   rank {np.abs(inp[sub]-rn[sub]).max():.0f}"
          f"   ({tb-ta:.0f}s vs {time.time()-tb:.0f}s on {int(sub.sum())} voxels)")

    # ---- check 2b: fast centre-row path == literal whiten/denoise/unwhiten -------------------
    kk = np.asarray(kernel) // 2
    pad = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    offs = kernel_offsets(kernel)
    jj = ii[:a.nslow]
    fastn, _, fastr = denoise_white(dwi, sub, Wm.astype(np.float32), kernel, a.algo)
    alg, rep32, same = [], [], 0
    for (x, y, z) in jj:
        Y = np.stack([pad[x + dx, y + dy, z + dz, :] for dx, dy, dz in offs]).astype(np.float64)
        lit, t64, cid = slow_patch(Y, Wm, Sm, P // 2, Mp, Np, a.algo)
        sc = max(np.abs(lit).max(), 1e-9)
        alg.append(np.abs(cid - lit).max() / sc)                 # SAME t and V: pure algebra
        if t64 == int(fastr[x, y, z]):
            same += 1
            rep32.append(np.abs(lit - fastn[x, y, z]).max() / sc)
    print(f"[check 2b] on {len(jj)} patches:")
    print(f"    algebra   max | (Y V V^T)_c  -  (R^1/2 X_w)_c |  = {max(alg):.3e}   "
          f"(the explicit unwhitening IS the identity for a centre-row denoiser)")
    print(f"    float32   rank agrees on {same}/{len(jj)}; where it does, max rel diff "
          f"fast-vs-float64 = {max(rep32) if rep32 else float('nan'):.3e}")
    print(f"              (rank ties at the MP crossing are an existing-engine float32 "
          f"property, not introduced by whitening)")

    # ---- check 3: coloured-noise null through the exact denoising code -----------------------
    print(f"\n[check 3] coloured-noise null, {a.nsim} sims, full 4-D volumes through the arms")
    rng = np.random.default_rng(7)
    Pn = regularize(np.load(npz)[key], 0.0)
    Wspec = Pn.shape[0]
    sh = (48, 48, 8)
    nullmask = np.zeros(sh, bool)
    nullmask[kk[0]:sh[0]-kk[0]:3, kk[1]:sh[1]-kk[1]:3, kk[2]:sh[2]-kk[2]] = True
    res = {k: [] for k in ("white", "col", "col-wht")}
    for it in range(a.nsim):
        g = rng.standard_normal((N, sh[2], Wspec, Wspec))
        col = np.fft.ifft2(np.fft.fft2(g, axes=(2, 3)) * np.sqrt(Pn)[None, None],
                           axes=(2, 3)).real[:, :, :sh[0], :sh[1]].transpose(2, 3, 1, 0)
        wht = rng.standard_normal(col.shape)
        _, _, n1 = denoise_white(wht.astype(np.float32), nullmask, np.eye(P, dtype=np.float32),
                                 kernel, a.algo)
        _, _, n2 = denoise_white(col.astype(np.float32), nullmask, np.eye(P, dtype=np.float32),
                                 kernel, a.algo)
        _, _, n3 = denoise_white(col.astype(np.float32), nullmask, Wm.astype(np.float32),
                                 kernel, a.algo)
        for k, n in (("white", n1), ("col", n2), ("col-wht", n3)):
            res[k].append(n[nullmask].mean())
    for k in res:
        print(f"  false rank  {k:8s} {np.mean(res[k]):6.3f} +- {np.std(res[k]):.3f}")

    # ---- check 4: singular-value bulk on HELD-OUT b0 residual patches ------------------------
    print(f"\n[check 4] singular-value bulk, held-out b0 residual patches")
    Rtr, Rva, Ctr, Cva, mk, nvol, b0, imgs, Dmap, info = load_sets(cfg, 1.5, np.random.default_rng(0),
                                                                   verbose=False)
    Rv = np.moveaxis(Rva, 0, -1)
    from scipy.ndimage import distance_transform_edt
    dt = distance_transform_edt(mask)
    cand = np.argwhere(dt > max(kk) + 2)
    cand = cand[np.random.default_rng(1).permutation(len(cand))[:600]]
    padr = np.pad(Rv, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    lb, lw = [], []
    for (x, y, z) in cand:
        Y = np.stack([padr[x + dx, y + dy, z + dz, :] for dx, dy, dz in offs])
        lb.append(np.linalg.svd(Y, compute_uv=False) ** 2)
        lw.append(np.linalg.svd(Wm @ Y, compute_uv=False) ** 2)
    lb, lw = np.array(lb), np.array(lw)
    # the residual chain removed the mean and one mode, so only R-2 eigenvalues are non-zero
    nb = Rva.shape[0] - 2
    edge = (np.sqrt(min(P, nb)) + np.sqrt(max(P, nb))) ** 2
    print(f"    {Rva.shape[0]} held-out b0s -> {nb} dof, so the 'bulk' is only {nb} "
          f"eigenvalues; this check has little power (see check 3 for the powered version)")
    for nm, L in (("before", lb), ("after ", lw)):
        L = L[:, :nb] / L[:, :nb].sum(1, keepdims=True) * (P * nb)     # unit mean entry variance
        print(f"  {nm}: lam_max/MPedge {np.mean(L[:,0])/edge:6.3f} +-{np.std(L[:,0])/edge:.3f}"
              f"   lam_1/lam_{nb} {np.mean(L[:,0]/L[:,nb-1]):6.2f}"
              f"   bulk spread sd/mean {np.mean(L.std(1)/L.mean(1)):6.3f}")

    if a.checks_only:
        return

    # ---- real data: three arms ---------------------------------------------------------------
    print(f"\n=== real data, {int(mask.sum())} voxels, {N} volumes ===")
    aff = img.affine
    ARMS = {}
    for nm, Wmat in (("std", None), ("white", Wm), ("ident", np.eye(P))):
        t1 = time.time()
        if nm == "std":
            o, sg, nn = denoise_adaptive(dwi, mask, kernel=kernel, algorithm=a.algo,
                                         shrinkage="threshold", adaptive=False)
        else:
            o, sg, nn = denoise_white(dwi, mask, Wmat.astype(np.float32), kernel, a.algo)
        rt = time.time() - t1
        ARMS[nm] = (o, sg, nn, rt)
        print(f"  {nm:6s} {rt:7.1f}s   rank mean {nn[mask].mean():6.2f} "
              f"median {np.median(nn[mask]):5.1f}   sigma mean {sg[mask].mean():8.2f}")
        nib.save(nib.Nifti1Image(sg, aff), f"{a.out}{nm}_sigma.nii.gz")
        nib.save(nib.Nifti1Image(nn, aff), f"{a.out}{nm}_rank.nii.gz")
        if nm != "ident":          # ident is bit-identical to std; verified numerically below
            for tag, arr in (("", o), ("_resid", dwi - o)):
                nib.save(nib.Nifti1Image(arr.astype(np.float32), aff),
                         f"{a.out}{nm}{tag}.nii.gz")
    nib.save(nib.Nifti1Image((ARMS["white"][2] - ARMS["std"][2]).astype(np.float32), aff),
             f"{a.out}rank_diff_white_minus_std.nii.gz")
    print(f"  identity arm vs std on the FULL volume: max|dsignal| "
          f"{np.abs(ARMS['ident'][0][mask]-ARMS['std'][0][mask]).max():.3e}  "
          f"max|drank| {np.abs(ARMS['ident'][2][mask]-ARMS['std'][2][mask]).max():.0f}")

    # ---- report ------------------------------------------------------------------------------
    S0 = dwi[..., b0].mean(-1)
    gm = np.linalg.norm(np.stack(np.gradient(S0)), axis=0)
    q = np.percentile(gm[mask], [25, 75])
    zc = np.argwhere(mask)[:, 2]
    REG = {"homogeneous (low |grad S0|)": mask & (gm <= q[0]),
           "edges (high |grad S0|)": mask & (gm >= q[1]),
           "superior (top 25% slices)": mask & (np.arange(mask.shape[2])[None, None, :]
                                                >= np.percentile(zc, 75))}
    print(f"\n  {'region':30s} {'n':>8s} {'rank std':>9s} {'rank wht':>9s} {'d rank':>7s} "
          f"{'sig std':>9s} {'sig wht':>9s} {'d sig %':>8s}")
    for nm, m in [("ALL", mask)] + list(REG.items()):
        rs_, rw = ARMS["std"][2][m], ARMS["white"][2][m]
        ss, sw = ARMS["std"][1][m], ARMS["white"][1][m]
        print(f"  {nm:30s} {int(m.sum()):8d} {rs_.mean():9.2f} {rw.mean():9.2f} "
              f"{rw.mean()-rs_.mean():+7.2f} {ss.mean():9.2f} {sw.mean():9.2f} "
              f"{100*(sw.mean()/ss.mean()-1):+8.2f}")

    print(f"\n  rank distribution (percentiles over the mask)")
    pc = [1, 5, 25, 50, 75, 95, 99]
    print(f"    {'arm':6s} " + " ".join(f"p{p:<3d}" for p in pc))
    for nm in ("std", "white"):
        print(f"    {nm:6s} " + " ".join(f"{v:4.0f}" for v in
                                         np.percentile(ARMS[nm][2][mask], pc)))
    dr = (ARMS["white"][2] - ARMS["std"][2])[mask]
    print(f"    rank difference white-std: mean {dr.mean():+.2f}  sd {dr.std():.2f}  "
          f"frac lower {100*(dr<0).mean():.1f}%  frac equal {100*(dr==0).mean():.1f}%")

    print(f"\n  residual variance (mean over mask, relative to std) and directional ACF")
    print(f"    {'arm':6s} {'resvar/std':>11s} {'ACFx1':>8s} {'ACFy1':>8s} {'ACFz1':>8s} "
          f"| whitened coords {'ACFx1':>8s} {'ACFy1':>8s} {'ACFz1':>8s}")
    vst = None
    from inplane_corr_model import kernel_from_P, regularize as _reg, conv2
    hker, _ = kernel_from_P(_reg(np.load(npz)[key], 0.0), 6)
    for nm in ("std", "white"):
        Rr = (dwi - ARMS[nm][0])
        v = float((Rr[mask] ** 2).mean())
        vst = v if vst is None else vst
        j = int(np.argmax(np.loadtxt(cfg["bval"]).ravel() > 500))
        A = acf3(Rr[..., j], mask)
        Fw = conv2(Rr[..., j][None, :, :, :], hker)[0]
        B = acf3(Fw, mask)
        print(f"    {nm:6s} {v/vst:11.4f} {A[1][0]:8.4f} {A[1][1]:8.4f} {A[1][2]:8.4f} "
              f"| {'':15s} {B[1][0]:8.4f} {B[1][1]:8.4f} {B[1][2]:8.4f}")

    json.dump({nm: dict(runtime_s=ARMS[nm][3], rank_mean=float(ARMS[nm][2][mask].mean()),
                        sigma_mean=float(ARMS[nm][1][mask].mean())) for nm in ARMS},
              open(f"{a.out}summary.json", "w"), indent=1)
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
