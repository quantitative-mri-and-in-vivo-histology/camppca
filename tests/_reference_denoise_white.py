"""TEST-ONLY: the global-whitener (n_shells = 1) reference implementation.

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

from camppca.patch import patch_offsets as patch_offsets
from camppca.gather import gather_budget_bytes
from camppca.mp import mp_threshold


def _tile_white(patches, Wmat, center: int, n_small: int, n_large: int, algo: str):
    Xw = jnp.einsum("pq,bqn->bpn", Wmat, patches)                  # whiten the VOXEL axis
    xc = patches[:, center, :]                                     # ORIGINAL centre row
    G = jnp.einsum("bpn,bpm->bnm", Xw, Xw)
    evals, V = jnp.linalg.eigh(G)
    vals = jnp.maximum(evals[:, ::-1], 0.0)
    Vd = V[:, :, ::-1]
    t, sigma = mp_threshold(vals, n_small, n_large, algo)
    fac = (jnp.arange(patches.shape[2])[None, :] < t[:, None]).astype(patches.dtype)
    coords = jnp.einsum("bn,bnm->bm", xc, Vd)
    xhat = jnp.einsum("bm,bnm->bn", coords * fac, Vd)
    return xhat, sigma, t.astype(patches.dtype)


_tile_white_jit = partial(jax.jit, static_argnums=(2, 3, 4, 5))(_tile_white)

def denoise_white(dwi, mask, Wmat, patch, algo="cordero-grande", tile=256):
    """Same tiling/gather as the package denoisers, plain box patches, voxel-axis whitening."""
    dwi = np.asarray(dwi, np.float32)
    sx, sy, sz, n_vols = dwi.shape
    patch = np.asarray(patch, int)
    kk = patch // 2
    n_vox = int(np.prod(patch))
    center = n_vox // 2
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    padded = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    Xp, Yp, Zp = padded.shape[:3]
    offs = patch_offsets(patch)
    xs, ys, zs = np.where(mask)
    tile = min(int(tile), max(1, int(5.0e7 / (n_vols * n_vols))))
    gpu = (Xp * Yp * Zp * n_vols * 4) < gather_budget_bytes()
    if gpu:
        pf = jnp.asarray(padded.reshape(-1, n_vols))
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
            hp = np.empty((ix.size, n_vox, n_vols), np.float32)
            for vi, (dx, dy, dz) in enumerate(offs):
                hp[:, vi, :] = padded[ix + dx, iy + dy, iz + dz, :]
            pat = jnp.asarray(hp)
        xh, sg, nn = _tile_white_jit(pat, Wj, center, n_small, n_large, algo)
        out[ix, iy, iz, :] = np.asarray(xh)
        sig[ix, iy, iz] = np.asarray(sg)
        npr[ix, iy, iz] = np.asarray(nn)
    return out, sig, npr
