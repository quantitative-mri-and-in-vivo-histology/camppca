"""The loop kernel: refine rho until it explains the MP-PCA bulk of every patch at once.

MP-PCA splits each whitened patch into kept directions and a bulk. Given a kernel, the
bulk of every patch must have the covariance that kernel says. A patch's bulk covariance
is ``(I - Q_b) Sigma (I - Q_b)^T``, with ``Q_b`` the projector onto its kept directions in
data units, so all patches together give one linear equation for ``Sigma``::

    mean_b (I - Q_b) Sigma (I - Q_b)^T  =  S_pooled

One round whitens every sampled patch with the current ``C``, splits it with the
denoiser's own code (:func:`~camppca.denoiser.camppca._whiten_split`), solves that equation by
Richardson iteration, and averages ``Sigma`` over lags into the next table. The current
kernel enters a round only through each patch's rank and kept directions. Directions a
patch kept are NOT filled in with the current kernel's values; that is what keeps the
loop from drifting.

The rounds stop when U of the patch correlation changes by less than :data:`LOOP_TOL`.
U closes about half its remaining gap each round, and a start far from the answer can dip
and turn, so a looser tolerance stops early on a single small step.

Directions EVERY patch keeps, such as the smooth patch-mean pattern, are invisible to the
bulk, and the Richardson solve leaves them at zero. That is harmless inside the loop but
not for describing the kernel, so the returned table is one least-squares fit over lag
classes at the fixed point, with those directions truncated at :data:`RCOND`. It is not
used inside the loop: there the smooth power it fills in pulls smooth leftover signal into
the bulk, and the kernel runs away.

Patch rows are divided by the guarded level field before whitening, exactly as the
denoiser divides them, so the loop fits ``C`` to the patches the denoiser whitens. The
MP rule (``algo``) and whether rows are divided by ``d`` (``noise_field``) are the
denoiser's settings, passed in, so the kernel is fitted to the split it will serve.

Lags the patch never spans, such as ``|dx| > 2`` for a 3 x 7 patch whose table reaches
lag 6, carry no information from the bulk. They keep the starting table's values.

⚠ The loop runs in float64. The Richardson solve leaves the patch-mean direction at
zero, so the next ``C`` is near singular there and whitening amplifies that direction by
~3e4. MP-PCA keeps it and the bulk is untouched, but only in float64: in float32 its
rounding error spills into the bulk and the ranks run to the maximum within one round.
"""

import warnings
from contextlib import ExitStack
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_erosion

from ..denoiser import _effective_d
from ..denoiser.camppca import _whiten_split, sqrtm_psd
from ..patch import patch_offsets
from ..pool import get_pool, resolve_workers, shared_volume
from . import _build_C

__all__ = ["loop_kernel"]

N_CENTRES = 3000        # sampled patches the bulk is pooled over
LOOP_TOL = 1e-3         # stop when U of C changes by less than this between rounds
MAX_ROUNDS = 20
N_INNER = 30            # Richardson steps per round
RCOND = 1e-2            # design directions below this share of the largest are unobservable
TILE = 256              # patches per device batch in the split
POOL_TILE = 32          # the same in a pool worker: ~1/8 the memory, and faster per core


def loop_kernel(dwi, mask, d, rho_start, patch, lag, seed, algo="cordero-grande",
                noise_field=True, workers=None):
    """Refine a kernel until it is consistent with the MP-PCA bulk of the patches.

    Args:
        dwi: (X, Y, Z, N) data. Every volume enters, as every volume enters the denoiser.
        mask: (X, Y, Z) bool.
        d: (X, Y, Z) level field. Guarded by ``_effective_d`` as the denoiser guards it.
        rho_start: (2*lag+1, 2*lag+1, dzmax+1) table to start from.
        patch: the patch the kernel must serve, as a length-3 int array.
        lag: in-plane reach of ``rho_start``.
        seed: seeds the choice of patch centres.
        algo: the MP rule the denoiser will cut with.
        noise_field: whether the denoiser will divide rows by ``d``.
        workers: pinned CPU processes the sampled patches are split over each round,
            see :mod:`camppca.pool`. Each gathers and splits its own share. The solve and
            the final least squares stay in this process.

    Returns:
        ``(rho, info)``: the stationary table, shaped like ``rho_start`` with unit zero
        lag and ``rho_start``'s values at lags the patch never spans, and a dict with
        ``rounds``, ``converged``, ``u_kernel`` (U of the returned table's patch
        correlation) and ``n_centres``.

    Raises:
        ValueError: if no mask voxel has its whole patch inside the mask.
    """
    patch = np.asarray(patch, int)
    dzmax = rho_start.shape[2] - 1
    workers = resolve_workers(workers)
    centres = _sample_centres(mask, patch, np.random.default_rng(seed))
    levels = _effective_d(d, mask) if noise_field else np.ones(mask.shape, np.float32)

    with ExitStack() as stack, jax.enable_x64(True):
        if workers > 1:
            path = stack.enter_context(shared_volume(dwi, levels))
            chunks = [tuple(c[part] for c in centres)
                      for part in np.array_split(np.arange(centres[0].size), workers)]

            def split(C):
                return _split_pooled(get_pool(workers), path, dwi.shape[3], chunks, patch,
                                     C, algo)
        else:
            patches = _gather(dwi, levels, centres, patch)

            def split(C):
                return _split(patches, C, algo)

        C = _build_C(rho_start, patch, lag)
        u_prev = _sphericity_excess(C)
        converged = False
        for rounds in range(1, MAX_ROUNDS + 1):  # noqa: B007 - read after the loop
            S, A, Bm, _ = split(C)
            sigma = np.asarray(_solve(jnp.asarray(S), A, Bm))
            C = _build_C(_lag_table(sigma, patch, lag, dzmax), patch, lag)
            u = _sphericity_excess(C)
            change = abs(u - u_prev)
            if change < LOOP_TOL:
                converged = True
                break
            u_prev = u
        if not converged:
            warnings.warn(f"loop kernel did not converge in {MAX_ROUNDS} rounds; U changed "
                          f"by {change:.4f} in the last one, against a tolerance "
                          f"of {LOOP_TOL}", RuntimeWarning, stacklevel=3)

        S, A, Bm, _ = split(C)
        rho = _stationary_table(S, A, Bm, patch, lag, dzmax)
    unspanned = ~_spanned(patch, lag, dzmax)
    rho[unspanned] = rho_start[unspanned]
    info = dict(rounds=rounds, converged=converged,
                u_kernel=_sphericity_excess(_build_C(rho, patch, lag)),
                n_centres=int(centres[0].size))
    return rho, info


def _sphericity_excess(C):
    """U = n tr(C^2) / tr(C)^2 - 1 of a unit-diagonal C; zero when C is white."""
    return float((C ** 2).sum() / len(C) - 1.0)


def _sample_centres(mask, patch, rng):
    """Up to :data:`N_CENTRES` mask voxels whose whole patch lies inside the mask."""
    eligible = binary_erosion(mask, structure=np.ones(tuple(patch), bool), border_value=0)
    idx = np.flatnonzero(eligible)
    if idx.size == 0:
        raise ValueError(f"no mask voxel has its whole {tuple(int(v) for v in patch)} "
                         "patch inside the mask, so the loop has no patch to fit")
    pick = np.sort(rng.choice(idx, size=min(N_CENTRES, idx.size), replace=False))
    return np.unravel_index(pick, mask.shape)


def _gather(dwi, d_eff, centres, patch):
    """(B, n_vox, N) float64 patches around ``centres``, rows divided by ``d_eff``.

    Rows in :func:`~camppca.patch.patch_offsets` order. Every patch lies inside the
    volume, so no padding is needed.
    """
    offsets = patch_offsets(patch) - patch // 2
    ix, iy, iz = (c[:, None] + offsets[None, :, k] for k, c in enumerate(centres))
    patches = np.asarray(dwi[ix, iy, iz, :], np.float64)
    return patches / np.asarray(d_eff, np.float64)[ix, iy, iz][:, :, None]


@partial(jax.jit, static_argnums=(2, 3, 4))
def _split_spectrum(patches, whitener, n_small: int, n_large: int, algo: str):
    """Whiten a tile and split each spectrum: what :func:`_whiten_split` returns."""
    return _whiten_split(patches, whitener, n_small, n_large, algo)


@partial(jax.jit, static_argnums=(7,))
def _split_bulk(whitened, evals, evecs, rank, sigma_hat, whitener, root, r: int):
    """One tile's share of the pooled bulk, and the factors of each kept projector.

    Only the first ``r`` eigenvectors enter, ``r`` at least the tile's largest rank.
    Every column past a patch's rank is zeroed anyway, and ``r`` is about a tenth of
    the volume count, so this skips most of the float64 work.

    Returns:
        ``(bulk, A, Bm)``: (n_vox, n_vox) sum over the tile of ``Rw Rw^T / (N
        sigma_hat^2)``, still whitened; (B, n_vox, r) ``C^(1/2) U`` and ``C^(-1/2) U``,
        whose product ``A Bm^T`` is the kept projector ``Q_b`` in data units, zero beyond
        each patch's rank.
    """
    n_vols = whitened.shape[2]
    evecs = evecs[:, :, :r]
    keep = (jnp.arange(r)[None, :] < rank[:, None])[:, None, :]
    coords = jnp.einsum("bpn,bnk->bpk", whitened, evecs)
    kept = jnp.where(keep, coords, 0.0)
    resid = whitened - jnp.einsum("bpk,bnk->bpn", kept, evecs)
    weight = 1.0 / (n_vols * sigma_hat ** 2)
    bulk = jnp.einsum("bpn,bqn,b->pq", resid, resid, weight)
    U = jnp.where(keep, coords / jnp.sqrt(jnp.maximum(evals[:, :r], 1e-12))[:, None, :],
                  0.0)
    A = jnp.einsum("pq,bqk->bpk", root, U)
    Bm = jnp.einsum("pq,bqk->bpk", whitener, U)
    return bulk, A, Bm


def _split_chunk(patches, C, algo, tile=TILE):
    """Whiten and split a batch of patches ``tile`` at a time. Call under float64.

    Returns:
        ``(bulk, A, Bm, rank)``: (n_vox, n_vox) whitened bulk summed over the batch; lists
        of each tile's ``A`` and ``Bm``, cut to the tile's largest rank; (B,) ranks.
    """
    n_b, n_vox, n_vols = patches.shape
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    whitener = jnp.asarray(sqrtm_psd(C, -0.5))
    root = jnp.asarray(sqrtm_psd(C, 0.5))

    bulk = np.zeros((n_vox, n_vox))
    A, Bm, rank = [], [], []
    for start in range(0, n_b, tile):
        whitened, evals, evecs, t_rank, sigma_hat = _split_spectrum(
            jnp.asarray(patches[start:start + tile]), whitener, n_small, n_large, algo)
        r = int(t_rank.max())
        cap = min(n_vols, max(8, -(-r // 8) * 8))     # few distinct shapes to compile
        t_bulk, t_A, t_Bm = _split_bulk(whitened, evals, evecs, t_rank, sigma_hat,
                                        whitener, root, cap)
        bulk += np.asarray(t_bulk)
        A.append(np.asarray(t_A[:, :, :r]))
        Bm.append(np.asarray(t_Bm[:, :, :r]))
        rank.append(np.asarray(t_rank))
    return bulk, A, Bm, np.concatenate(rank)


def _assemble(C, chunks):
    """Pool the chunks' bulks and stack their factors, in chunk order.

    Returns:
        ``(S, A, Bm, rank)``: (n_vox, n_vox) pooled bulk covariance in data units per unit
        noise variance; (B, n_vox, r) device arrays ``A`` and ``Bm``, with r the largest
        rank rounded up to a multiple of 8 so few shapes get compiled; (B,) ranks.
    """
    root64 = sqrtm_psd(C, 0.5)
    bulk = sum(chunk[0] for chunk in chunks)
    A = [a for chunk in chunks for a in chunk[1]]
    Bm = [b for chunk in chunks for b in chunk[2]]
    rank = np.concatenate([chunk[3] for chunk in chunks])
    r_max = max(8, -(-max(a.shape[2] for a in A) // 8) * 8)

    def stack(parts):
        return jnp.asarray(np.concatenate(
            [np.pad(p, ((0, 0), (0, 0), (0, r_max - p.shape[2]))) for p in parts]))

    S = root64 @ bulk @ root64 / rank.size
    return S, stack(A), stack(Bm), rank


def _split(patches, C, algo):
    """Whiten every patch by ``C``, split it, and pool the bulk. Call under float64.

    Returns what :func:`_assemble` returns.
    """
    return _assemble(C, [_split_chunk(patches, C, algo)])


def _split_task(args):
    """One worker's share of a round: gather its patches from the shared file, split."""
    path, n_vols, centres, patch, C, algo = args
    vol = np.load(path, mmap_mode="r")
    with jax.enable_x64(True):
        patches = _gather(vol[..., :n_vols], vol[..., n_vols], centres, patch)
        return _split_chunk(patches, C, algo, POOL_TILE)


def _split_pooled(pool, path, n_vols, chunks, patch, C, algo):
    """:func:`_split` with the patches gathered and split in the worker pool."""
    tasks = [(path, n_vols, chunk, patch, C, algo) for chunk in chunks]
    return _assemble(C, list(pool.map(_split_task, tasks)))


def _bulk_operator(sigma, A, Bm, Qm):
    """``mean_b (I - Q_b) sigma (I - Q_b)^T``, with ``Q_b = A_b Bm_b^T``."""
    inner = jnp.einsum("bpk,bpl->bkl", Bm, jnp.einsum("pq,bql->bpl", sigma, Bm))
    cross = jnp.einsum("bpl,bql->pq", jnp.einsum("bpk,bkl->bpl", A, inner), A)
    return sigma - Qm @ sigma - sigma @ Qm.T + cross / A.shape[0]


@jax.jit
def _solve(S, A, Bm):
    """The ``Sigma`` whose pooled bulk is ``S``: Richardson iteration from ``S`` itself."""
    Qm = jnp.einsum("bpk,bqk->pq", A, Bm) / A.shape[0]

    def step(_, sigma):
        sigma = S + sigma - _bulk_operator(sigma, A, Bm, Qm)
        return 0.5 * (sigma + sigma.T)

    return jax.lax.fori_loop(0, N_INNER, step, S)


def _lag_table(sigma, patch, lag, dzmax):
    """Average a (n_vox, n_vox) covariance over voxel pairs with the same lag.

    Returns a ``rho[dx+lag, dy+lag, |dz|]`` table with unit zero lag. Lags the patch
    never spans stay zero.
    """
    off = patch_offsets(patch)
    d = off[:, None, :] - off[None, :, :]
    idx = (d[..., 0] + lag, d[..., 1] + lag, np.abs(d[..., 2]))
    table = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    count = np.zeros_like(table)
    np.add.at(table, idx, sigma)
    np.add.at(count, idx, 1.0)
    table /= np.maximum(count, 1.0)
    return table / table[lag, lag, 0]


def _spanned(patch, lag, dzmax):
    """Bool table: the ``rho[dx+lag, dy+lag, |dz|]`` entries some voxel pair in the patch has."""
    off = patch_offsets(patch)
    d = off[:, None, :] - off[None, :, :]
    spanned = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1), bool)
    spanned[d[..., 0] + lag, d[..., 1] + lag, np.abs(d[..., 2])] = True
    return spanned


def _lag_classes(patch):
    """Every voxel pair's lag class, and the (dx, dy, |dz|) of each class.

    ``(dx, dy, dz)`` and ``(-dx, -dy, dz)`` share a class, because a table indexed by
    ``|dz|`` cannot tell them apart.
    """
    off = patch_offsets(patch)
    d = off[:, None, :] - off[None, :, :]
    dx, dy, dz = d[..., 0], d[..., 1], np.abs(d[..., 2])
    flip = (dx < 0) | ((dx == 0) & (dy < 0))
    dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
    lags, cls = np.unique(np.stack([dx, dy, dz], -1).reshape(-1, 3), axis=0,
                          return_inverse=True)
    return cls.reshape(dx.shape), lags


@partial(jax.jit, static_argnums=(3,))
def _design(cls, A, Bm, n_classes: int):
    """The bulk operator applied to every lag class's indicator matrix, (K, n_vox, n_vox)."""
    Qm = jnp.einsum("bpk,bqk->pq", A, Bm) / A.shape[0]
    return jax.lax.map(lambda c: _bulk_operator((cls == c).astype(A.dtype), A, Bm, Qm),
                       jnp.arange(n_classes))


def _stationary_table(S, A, Bm, patch, lag, dzmax):
    """The smallest stationary kernel whose bulk fits ``S``, as a lag table.

    One least squares over lag classes, with the design's unobservable directions
    truncated at :data:`RCOND`.

    The design is built in float32 from the float64 factors, at full matmul precision,
    and solved in float64. It is one linear map of fixed factors, so it does not meet the
    baseline problem the rounds have. Measured at 7^3 on a tensor phantom: the same table
    within 1e-4, the same sigma_hat, and the build ~20x faster on a GPU (75 s to 4 s).
    """
    cls, lags = _lag_classes(patch)
    with jax.default_matmul_precision("highest"):
        design = _design(jnp.asarray(cls), jnp.asarray(A, jnp.float32),
                         jnp.asarray(Bm, jnp.float32), len(lags))
    design = np.asarray(design, np.float64)
    design = design.reshape(len(lags), -1).T
    values = np.linalg.lstsq(design, S.ravel(), rcond=RCOND)[0]
    table = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    for v, (dx, dy, dz) in zip(values, lags, strict=True):
        table[dx + lag, dy + lag, dz] = table[-dx + lag, -dy + lag, dz] = v
    return table / table[lag, lag, 0]
