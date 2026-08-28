"""Plain white MP-PCA -- the BASELINE ARM, plus the MP threshold shared with camppca.py.

⚠ NOT ON THE MAIN PATH. camppca's method is ``denoiser/camppca.py``; this module exists
for two reasons: it is the standard-MP-PCA comparison arm, and it owns
``_mp_threshold_jax`` (the Marchenko-Pastur cutoff rule) and
``_device_gather_budget_bytes``, both of which the covariance-aware engine imports.

Ported from ``src/warp_denoise/mppca.py`` (see reference/). The bodies are verbatim apart
from the two dropped branches noted below; the NAMES were moved to CLAUDE.md's notation
once the self-test was green:

    kernel     -> patch                the box shape (paper M = n_vox voxels in it)
    P, N       -> n_vox, n_vols
    Mp, Np     -> n_small, n_large     ⚠ min/max of (n_vox, n_vols) -- the SORTED pair.
                                       The old names read as patch-M and patch-N; they
                                       never meant that.

Port of DESIGNER-v2's ``lib/mpdenoise_adaptive.py`` (``MP`` class) with a single JAX
implementation: the per-voxel loop is replaced by a batched device kernel, so the same
estimator runs on the CPU (default JAX backend) or the GPU (``jax`` with a CUDA build).

DESIGNER's two optional branches -- bilateral top-``n_keep`` patch selection and
Gavish-Donoho Frobenius shrinkage -- are not carried over. Nothing in camppca ever used
them: the comparison arm is the plain full cube with a hard threshold (mrtrix
``dwidenoise`` style), and CLAUDE.md fixes the hard threshold as the method.

Method per target voxel:

1. Extract the ``patch`` (default 5x5x5) box around the voxel from the ``mode='wrap'``
   padded volume -> data matrix ``X`` of shape ``(n_vox, n_vols)`` (rows = patch voxels, cols =
   diffusion volumes).
2. **Marchenko-Pastur threshold** (Veraart / Cordero-Grande / Jespersen; default
   ``jespersen``) on the eigenvalues of ``X``. No demeaning.
3. Hard-threshold the spectrum, reconstruct, and write back **only the centre voxel's**
   denoised signal.

Numerics: the tile core forms the ``n_vols x n_vols`` Gram over volumes and
eigendecomposes it, which
reproduces the centre-voxel output of a full SVD exactly (``x_centre . V`` equals
``u[centre] . s``). ``_mp_threshold`` and ``_denoise_patch`` are single-patch JAX
convenience wrappers over the batched core; ``denoise_mppca`` is the batched entry point.

References: Veraart 2016 (MP-PCA), Cordero-Grande 2019, Ades-Aron/Jespersen (DESIGNER-v2).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def _default_patch(nvols: int) -> np.ndarray:
    """Smallest odd isotropic patch with ``p**3 >= nvols`` (DESIGNER default)."""
    p = np.arange(3, max(nvols, 5), 2)
    pf = np.where(p**3 >= nvols)[0][0]
    return np.array([p[pf]] * 3)


def _device_gather_budget_bytes() -> float:
    """Byte budget for keeping the whole padded volume on-device for the fast indexed gather.

    On a CUDA device we can query free memory and use most of it (the per-tile eigh workspace is
    tiny); on CPU / when stats are unavailable, fall back to a conservative 2 GB (prior default).
    """
    try:
        dev = jax.devices()[0]
        if dev.platform != "gpu":
            return 2.0e9
        st = dev.memory_stats() or {}
        free = st.get("bytes_limit", 0) - st.get("bytes_in_use", 0)
        # leave ~1 GB headroom for the batched Gram/eigh workspace and output staging
        return max(2.0e9, free - 1.0e9)
    except Exception:
        return 2.0e9


# ---------------------------------------------------------------------------
# Batched JAX core.
# ---------------------------------------------------------------------------

def _mp_threshold_jax(vals, n_small: int, n_large: int, algo: str):
    """Batched Marchenko-Pastur threshold. ``vals`` (B, K>=n_small) descending eigenvalues
    (singular values squared). Returns ``t`` (B,) int32 = #retained components and
    ``sigma`` (B,) noise level (tn=0 crossing of Veraart / Cordero-Grande / Jespersen).
    """
    eps = jnp.finfo(vals.dtype).eps
    v = vals[:, :n_small]                                    # (B, n_small)
    p = jnp.arange(n_small)                                  # (n_small,)
    csum = jnp.cumsum(v[:, ::-1], axis=1)[:, ::-1]      # csum[:, i] = sum(v[:, i:])
    last = v[:, n_small - 1:n_small]                              # (B,1) smallest kept

    if algo == "veraart":
        sig1 = csum / ((n_small - p) * float(n_large))
        range_mp = 4.0 * jnp.sqrt((n_small - p) * float(n_large))
    elif algo == "cordero-grande":
        sig1 = csum / ((n_small - p) * (n_large - p))
        range_mp = 4.0 * jnp.sqrt((n_small - p) * (n_large - p))
    elif algo == "jespersen":
        sig1 = csum / ((n_small - p) * (n_large - p))
        range_mp = 4.0 * jnp.sqrt(float(n_large) * float(n_small))   # scalar
    else:
        raise ValueError(f"unknown algorithm {algo!r}")

    sig2 = (v - last) / range_mp                        # (B, n_small)
    cond = sig2 < sig1                                  # first True = first crossing
    any_hit = jnp.any(cond, axis=1)
    t = jnp.where(any_hit, jnp.argmax(cond, axis=1), n_small - 1)   # else n_small-1 (tn=0)
    sig1_t = jnp.take_along_axis(sig1, t[:, None], axis=1)[:, 0]
    sigma = jnp.sqrt(jnp.maximum(sig1_t, 0.0))
    sigma = jnp.where((sigma > 0) & jnp.isfinite(sigma), sigma, eps)
    return t.astype(jnp.int32), sigma


def _denoise_tile(patches, col_w, center: int, n_small: int, n_large: int, algo: str):
    """Denoise a tile of ``B`` boxes. ``patches`` (B, n_vox, n_vols); ``col_w``
    (B, n_vols) per-volume noise weights (all-zero -> no whitening). Returns the denoised centre
    spectrum ``xhat`` (B, n_vols), ``sigma`` (B,), ``npars`` (B,). See module docstring for the
    Gram-eigh == SVD-centre-row equivalence.
    """
    n_vols = patches.shape[2]
    dt = patches.dtype

    xc = patches[:, center, :]                                      # (B,n_vols) centre spectrum
    w = jnp.where(col_w > 0, col_w, 1.0)                            # (B,n_vols); all-zero -> ones
    Xs = patches / w[:, None, :]                                    # (B,n_vox,n_vols)
    xc = xc / w

    G = jnp.einsum("bpn,bpm->bnm", Xs, Xs)                          # (B,n_vols,n_vols) Gram
    evals, V = jnp.linalg.eigh(G)                                   # ascending
    vals = jnp.maximum(evals[:, ::-1], 0.0)                         # descending (B,n_vols)
    Vd = V[:, :, ::-1]                                              # eigenvectors, descending
    coords = jnp.einsum("bn,bnm->bm", xc, Vd)                       # project centre onto eigvecs

    t, sigma = _mp_threshold_jax(vals, n_small, n_large, algo)                # (B,), (B,)

    fac = (jnp.arange(n_vols)[None, :] < t[:, None]).astype(dt)     # (B,n_vols) keep mask

    xhat = jnp.einsum("bm,bnm->bn", coords * fac, Vd)              # reconstruct centre row
    xhat = xhat * w                                                # un-whiten
    return xhat, sigma, t.astype(dt)


_denoise_tile_jit = partial(jax.jit, static_argnums=(2, 3, 4, 5))(_denoise_tile)


# ---------------------------------------------------------------------------
# Single-patch JAX convenience wrappers (delegate to the batched core; unit-tested).
# ---------------------------------------------------------------------------

def _mp_threshold(vals, n_small: int, n_large: int, algo: str, tn: int = 0):
    """Single-patch Marchenko-Pastur threshold -> ``(t, sigma)``. ``vals`` descending
    eigenvalues (1-D). ``tn`` is accepted for signature compatibility (only ``tn=0``,
    the value used throughout this project, is supported)."""
    if tn:
        raise NotImplementedError("tn>0 is not supported by the JAX threshold")
    t, sigma = _mp_threshold_jax(jnp.asarray(vals)[None, :], n_small, n_large, algo)
    return int(t[0]), float(sigma[0])


def _denoise_patch(X, center, algo, tn: int = 0, col_weights=None):
    """Denoise one box ``X`` of shape ``(n_vox, n_vols)``; return ``(signal_N, sigma, npars)``.

    ``col_weights`` (length ``n_vols``): optional per-volume noise scale for column whitening.
    Thin wrapper over :func:`_denoise_tile` with a batch of one.
    """
    if tn:
        raise NotImplementedError("tn>0 is not supported by the JAX denoiser")
    X = jnp.asarray(X, jnp.float32)
    n_vox, n_vols = X.shape
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    col_w = (jnp.zeros((1, n_vols), jnp.float32) if col_weights is None
             else jnp.asarray(col_weights, jnp.float32)[None, :])
    xhat, sigma, npars = _denoise_tile(X[None], col_w, int(center), n_small, n_large, algo)
    return np.asarray(xhat[0]).ravel(), float(sigma[0]), int(npars[0])


# ---------------------------------------------------------------------------
# Public batched entry point.
# ---------------------------------------------------------------------------

def denoise_mppca(
    dwi: np.ndarray,
    mask: np.ndarray | None = None,
    patch=(5, 5, 5),
    algorithm: str = "jespersen",
    col_weights: np.ndarray | None = None,
    n_jobs: int = -1,
    tile: int = 256,
    on_device_gather: bool | None = None,
):
    """MPPCA denoise a 4D volume ``(X, Y, Z, n_vols)`` on the JAX backend (CPU or GPU).

    Every patch is the plain full cube (mrtrix ``dwidenoise`` style) and the spectrum is
    hard-thresholded. See the module docstring for the DESIGNER options left behind.

    ``col_weights`` (X,Y,Z,n_vols): optional per-voxel per-volume noise scale ``‖w‖``; each
    patch is column-whitened by its centre voxel's weights before the eigendecomposition.

    ``n_jobs`` is accepted for backwards compatibility and ignored (batching now runs on the
    JAX device); ``tile`` caps the per-batch patch count to bound device memory.

    ``on_device_gather`` controls how the overlapping patches are extracted: ``True`` keeps the
    whole padded volume on the device and gathers each tile's box with an indexed read (fast, but
    needs the padded volume + workspace to fit in device memory); ``False`` gathers each tile on
    the host and ships only the small block (bounded host memory, slower — the GPU can starve on
    the CPU gather). ``None`` (default) auto-selects: on-device when the padded volume fits the
    free device-memory budget (see :func:`_device_gather_budget_bytes`), else host.

    Only voxels inside ``mask`` are denoised; other voxels keep their input value.
    Returns ``(denoised (X,Y,Z,n_vols), sigma (X,Y,Z), npars (X,Y,Z))``; ``sigma``/``npars`` are 0
    outside the processed region.
    """
    del n_jobs  # batching moved to the device; kept in the signature for compatibility
    dwi = np.asarray(dwi, np.float32)
    sx, sy, sz, nvols = dwi.shape

    patch = np.array(patch) if patch is not None else _default_patch(nvols)
    patch = (patch + np.mod(patch, 2) - 1).astype(int)  # force odd
    kk = patch // 2
    n_vox = int(np.prod(patch))
    center = n_vox // 2

    # effective matrix shape seen by MP-PCA (rows = patch voxels, cols = volumes)
    n_small, n_large = min(n_vox, nvols), max(n_vox, nvols)

    # padded copy for border-safe box extraction (wrap = every in-mask voxel gets a full box)
    padded = np.pad(dwi, ((kk[0], kk[0]), (kk[1], kk[1]), (kk[2], kk[2]), (0, 0)), mode="wrap")
    Xp, Yp, Zp = padded.shape[:3]
    offs = [(dx, dy, dz) for dx in range(patch[0]) for dy in range(patch[1])
            for dz in range(patch[2])]

    if mask is None:
        mask = np.ones((sx, sy, sz), bool)
    else:
        mask = np.asarray(mask, bool)
    xs, ys, zs = np.where(mask)

    tile = min(int(tile), max(1, int(5.0e7 / (nvols * nvols))))

    # Patch extraction: gather each tile's box on the device from the flat padded volume when it
    # fits the device budget (~1.8x faster, bit-identical); else host-gather per tile so the big
    # padded array stays off-device. `on_device_gather` overrides the auto decision.
    if on_device_gather is None:
        gpu_patches = (Xp * Yp * Zp * nvols * 4) < _device_gather_budget_bytes()
    else:
        gpu_patches = bool(on_device_gather)
    if gpu_patches:
        pf = jnp.asarray(padded.reshape(-1, nvols))
        doff = jnp.asarray(np.array([dx * Yp * Zp + dy * Zp + dz
                                     for dx, dy, dz in offs], np.int32))
        base = jnp.asarray((xs * Yp * Zp + ys * Zp + zs).astype(np.int32))
    if col_weights is not None:
        col_weights = np.asarray(col_weights, np.float32)

    out = dwi.copy()
    sigma_map = np.zeros((sx, sy, sz), np.float32)
    npars_map = np.zeros((sx, sy, sz), np.float32)

    for s in range(0, xs.size, tile):
        ix, iy, iz = xs[s:s + tile], ys[s:s + tile], zs[s:s + tile]
        if gpu_patches:
            patches = pf[base[s:s + tile][:, None] + doff[None, :]]   # gathered on-device
        else:
            B = ix.size
            hp = np.empty((B, n_vox, nvols), np.float32)
            for vi, (dx, dy, dz) in enumerate(offs):
                hp[:, vi, :] = padded[ix + dx, iy + dy, iz + dz, :]   # per-tile host gather
            patches = jnp.asarray(hp)
        if col_weights is not None:
            cw = jnp.asarray(col_weights[ix, iy, iz, :])
        else:
            cw = jnp.zeros((ix.size, nvols), np.float32)
        xhat, sig, npar = _denoise_tile_jit(patches, cw, center, n_small, n_large, algorithm)
        out[ix, iy, iz, :] = np.asarray(xhat)
        sigma_map[ix, iy, iz] = np.asarray(sig)
        npars_map[ix, iy, iz] = np.asarray(npar)
    return out, sigma_map, npars_map
