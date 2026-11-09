"""The loop: refine rho until it explains the MPPCA bulk of every patch at once.

MPPCA splits each whitened patch into kept directions and a bulk. Given a correlation
pattern, the bulk of every patch must have the covariance it says. A patch's bulk covariance
is ``(I - Q_b) Sigma (I - Q_b)^T``, with ``Q_b`` the projector onto its kept directions in
data units, so all patches together give one linear equation for ``Sigma``::

    mean_b (I - Q_b) Sigma (I - Q_b)^T  =  S_pooled

One round whitens every sampled patch with the current ``C``, splits it with the
denoiser's own code (:func:`~camppca.denoiser.camppca._whiten_split`), and fits that
equation with ``Sigma`` restricted to a stationary one: one value per lag class, the voxel
pairs that share a lag up to its sign, ``h`` and ``-h``. 1098 classes at 7^3, against
58,996 unknowns
for a free ``Sigma``. The current pattern enters a round only through each patch's rank
and kept directions. Directions a patch kept are NOT filled in with the current pattern's
values; that is what keeps the loop from drifting.

THE HOLE. Every patch keeps its smooth patch-mean pattern as signal, so the bulk barely
sees the noise along it. Among stationary tables this is exactly one direction, the same
value added to every lag: on the EDDEN crop its weight in the fit was 2.5e-4 of the
largest, the next weakest 4.8e-2. It is fixed by an assumption, noise correlation dies
off with distance, so the mean over the far classes, those on the edge of the patch's lag
box, is held at zero.

The fit used to solve for a free ``Sigma`` and average it over lags. There the hole is an
eigenvalue of 1.1e-4 along the patch mean, and a converged solve divides by it: the
pattern ran away on the crop, sphericity 0.16 to 9 in eleven rounds. 30 Richardson steps hid that
by stopping early. With the far classes pinned, the loop converges, and on the paper's
phantom with a signal MPPCA separates perfectly (one flat value per volume) it lands
within 0.005 of the true correlation, against 0.014 for the converged free solve.

⚠ The split itself is biased, and no fit undoes it. Signal components near the MP edge
are kept with patterns tilted toward the noise, and weaker ones stay in the bulk. Both
are smooth in space and read as extra correlation. On the paper's phantom (Hann window
f = 0.5, g = 0.33, 80 volumes, 5^3) a round started at the truth gave a z correlation
of 0.037 where the truth is 0, and the converged loop 0.051. Patch2Self overshot too.

The rounds stop when the sphericity of the patch correlation changes by less than
:data:`LOOP_TOL`. It closes about half its remaining gap each round, and a start far
from the answer can dip and turn, so a looser tolerance stops early on a single small
step. The table of the last
round is returned.

Patch rows are divided by the guarded relative noise map before whitening, exactly as the
denoiser divides them, so the loop fits ``C`` to the patches the denoiser whitens. The
MP rule (``algorithm``) and whether rows are divided by ``d`` (``noise_field``) are the
denoiser's settings, passed in, so the pattern is fitted to the split it will serve.

Lags the patch never spans, such as ``|dx| > 2`` for a 3 x 7 patch whose table reaches
lag 6, carry no information from the bulk. They keep the starting table's values.

The split runs in float32, with each tile's bulk product in float64. It used to need
float64: with the free solve the patch-mean direction of the next ``C`` was near
singular, and float32 rounding spilled into the bulk and ran the ranks to the maximum.
The pinned fit keeps the smallest eigenvalue of ``C`` near 0.2 to 0.4. Against a float64
split the table moved by at most 2.4e-5 on the crop and 1.3e-5 on the paper's phantom,
with the sphericity the same to four digits; 1% of patch ranks flipped on near-tied MP
cuts. The fit runs in float32 too, within 1e-4 of a float64 least squares.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial

import numpy as np
from joblib import delayed
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.ndimage import binary_erosion

from ..denoiser.camppca import _whiten_split, sqrtm_psd
from ..denoiser.gather import TILE
from ..patch import patch_offsets, table_index
from ..pool import resolve_workers, worker_pool
from . import (
    FALLBACK_REASON,
    MIN_EIGENVALUE,
    _build_C,
    _effective_d,
    _good_level,
    _table_C,
    _white_rho,
)

__all__ = ["fit_rho"]

MAX_CENTERS = 5000      # patch centers the bulk is pooled over, at most
MIN_CENTERS = 500       # fewer tiles than this and the grid narrows, so patches overlap
LOOP_TOL = 1e-3         # stop when C's sphericity changes by less than this per round
MAX_ROUNDS = 20
LOOP_CHUNKS = 32        # fixed chunks of sampled patches, so workers cannot change the sums
FIT_TOL = 1e-5          # the fit stops at this gradient, relative to its start
MAX_FIT_STEPS = 300
PRODUCT_THREADS = 8     # threads for the bulk map's per-patch products
PRODUCT_CHUNK = 64      # patches per thread task

_product_pools = {}     # one thread pool per thread count, kept: starting one costs time


def fit_rho(dwi, mask, d, rho_start, patch, lag, algorithm="cordero-grande",
                noise_field=True, workers=None, sample=None):
    """Refine rho until it is consistent with the MPPCA bulk of the patches.

    Args:
        dwi: (X, Y, Z, N) data. Every volume enters, as every volume enters the denoiser.
        mask: (X, Y, Z) bool.
        d: (X, Y, Z) relative noise map. Guarded by ``_effective_d`` as the denoiser guards it.
        rho_start: (2*lag+1, 2*lag+1, dzmax+1) table to start from.
        patch: the patch the pattern must serve, as a length-3 int array.
        lag: in-plane reach of ``rho_start``.
        algorithm: the MP rule the denoiser will cut with.
        noise_field: whether the denoiser will divide rows by ``d``.
        workers: CPU processes the sampled patches are split over each round, see
            :mod:`camppca.pool`. Each gathers and splits its own share. The fit stays
            in this process.
        sample: (X, Y, Z) bool, the voxels patch centers are taken from, or None for the
            mask voxels whose level passes the floor of ``_effective_d``. The level rounds
            pass the first one, so every round fits the same patches.

    Returns:
        ``(rho, info)``: the stationary table, shaped like ``rho_start`` with unit zero
        lag and ``rho_start``'s values at lags the patch never spans, and a dict with
        ``rounds``, ``converged``, ``sphericity`` (that of the returned table's patch
        correlation), ``n_centers``, ``stride`` (of the grid of patch centers, see
        :func:`_sample_centers`) and ``fallback``: None, or why the table fell back to
        white noise because a round's fit was not a valid correlation.

    Raises:
        ValueError: if no mask voxel has its whole patch inside the mask, among the
            voxels whose level passes the floor of ``_effective_d``.
    """
    patch = np.asarray(patch, int)
    dzmax = rho_start.shape[2] - 1
    workers = resolve_workers(workers)
    # Patches with a level below the floor hold near-constant data, which the fit cannot use.
    if sample is None:
        sample = _good_level(d, mask)
    centers, stride = _sample_centers(sample, patch)
    levels = _effective_d(d, mask) if noise_field else np.ones(mask.shape, np.float32)

    # Fixed chunks, pooled in chunk order, so the model does not depend on ``workers``.
    parts = np.array_split(np.arange(centers[0].size), min(LOOP_CHUNKS, centers[0].size))
    with ExitStack() as stack:
        if workers > 1:
            # One pool for every round: joblib maps the volume and the levels once.
            parallel = stack.enter_context(worker_pool(workers))
            chunks = [tuple(c[part] for c in centers) for part in parts]

            def split(C):
                return _assemble(C, parallel(
                    delayed(_split_task)(dwi, levels, chunk, patch, C, algorithm)
                    for chunk in chunks))
        else:
            patches = _gather(dwi, levels, centers, patch)

            def split(C):
                return _assemble(C, [_split_chunk(patches[part], C, algorithm)
                                     for part in parts])

        rho = rho_start
        C = _build_C(rho, patch, lag)
        u_prev = _sphericity_excess(C)
        converged, fallback = False, None
        for rounds in range(1, MAX_ROUNDS + 1):  # noqa: B007 - read after the loop
            S, factors = split(C)
            rho = _fit_table(S, factors, C, patch, lag, dzmax, rho,
                             threads=min(PRODUCT_THREADS, workers))
            # Check before projecting: whitening by a near-singular C overflowed float32.
            smallest = float(np.linalg.eigvalsh(_table_C(rho, patch, lag))[0])
            if smallest < MIN_EIGENVALUE:
                fallback = (f"the loop's correlation pattern is not valid after round "
                            f"{rounds}: the smallest eigenvalue of C is {smallest:.1e}, "
                            f"below {MIN_EIGENVALUE:g}. {FALLBACK_REASON}")
                warnings.warn(fallback, RuntimeWarning, stacklevel=3)
                rho = _white_rho(rho.shape)
                break
            C = _build_C(rho, patch, lag)
            u = _sphericity_excess(C)
            change = abs(u - u_prev)
            if change < LOOP_TOL:
                converged = True
                break
            u_prev = u
        if not converged and fallback is None:
            warnings.warn(f"the loop did not converge in {MAX_ROUNDS} rounds; the "
                          f"sphericity changed by {change:.4f} in the last one, against a "
                          f"tolerance "
                          f"of {LOOP_TOL}", RuntimeWarning, stacklevel=3)
    rho = rho.copy()
    unspanned = ~_spanned(patch, lag, dzmax)
    rho[unspanned] = rho_start[unspanned]
    info = dict(rounds=rounds, converged=converged, fallback=fallback,
                sphericity=_sphericity_excess(_build_C(rho, patch, lag)),
                n_centers=int(centers[0].size), stride=stride)
    return rho, info


def _sphericity_excess(C):
    """The sphericity n tr(C^2) / tr(C)^2 - 1 of a unit-diagonal C; zero when C is white."""
    return float((C ** 2).sum() / len(C) - 1.0)


def _sample_centers(mask, patch):
    """Patch centers on a grid, each patch wholly inside the mask.

    The stride is the patch, so the patches tile the mask without overlap and each brings
    its own noise. Of the grid's phases the one with the most centers is taken. Above
    :data:`MAX_CENTERS` the stride widens until under it, below :data:`MIN_CENTERS` it
    narrows until over it, down to every voxel. Every axis scales by the same factor, in
    steps of one voxel on the longest, so the grid keeps the patch's shape. Adding one
    voxel per axis instead grew the z stride of a 7 x 7 x 3 patch by a third against a
    seventh in plane, and skipped slices.

    Measured 2026-11-09 on EDDEN 0.9, 1.5 and 2 mm (``docs/local/2026-11-09_loop_sampling.md``):
    sigma moved by at most 0.1% above a few hundred patches, and rho stopped improving at
    about 700 patches of 7^3. At 2 mm and 5^3 overlapping centers beyond the tiling, up
    to 7060, came no closer than the tiling's 1508. A grid was no more precise than random
    centers at the same count. It ties the count to the mask and needs no seed. Patches
    reaching outside the mask lowered the sphericity a little and gained nothing.

    Returns:
        ``(centers, stride)``: the centers as index arrays, C order, and the stride used.
    """
    eligible = binary_erosion(mask, structure=np.ones(tuple(patch), bool), border_value=0)
    if not eligible.any():
        raise ValueError(f"no mask voxel has its whole {tuple(int(v) for v in patch)} "
                         "patch inside the mask, so the loop has no patch to fit")
    longest = int(max(patch))

    def scaled(k):                    # the stride with k voxels added on the longest axis
        return tuple(max(1, int(np.floor(int(p) * (longest + k) / longest + 0.5)))
                     for p in patch)

    k = 0
    stride = scaled(k)
    centers = _grid(eligible, stride)
    if centers.sum() > MAX_CENTERS:
        while centers.sum() > MAX_CENTERS:
            k += 1
            stride = scaled(k)
            centers = _grid(eligible, stride)
        return np.nonzero(centers), stride
    while centers.sum() < MIN_CENTERS and max(stride) > 1:
        k -= 1
        stride = scaled(k)
        centers = _grid(eligible, stride)
    return np.nonzero(centers), stride


def _grid(eligible, stride):
    """The eligible voxels on the grid of this stride, at the phase that keeps the most."""
    best, keep = -1, None
    for phase in np.ndindex(*stride):
        on = tuple(slice(p, None, s) for p, s in zip(phase, stride, strict=True))
        count = int(eligible[on].sum())
        if count > best:
            best, keep = count, on
    grid = np.zeros_like(eligible)
    grid[keep] = eligible[keep]
    return grid


def _gather(dwi, d_eff, centers, patch):
    """(B, n_vox, N) float32 patches around ``centers``, rows divided by ``d_eff``.

    Rows in :func:`~camppca.patch.patch_offsets` order. Every patch lies inside the
    volume, so no padding is needed.
    """
    offsets = patch_offsets(patch) - patch // 2
    ix, iy, iz = (c[:, None] + offsets[None, :, k] for k, c in enumerate(centers))
    patches = np.asarray(dwi[ix, iy, iz, :], np.float32)
    return patches / np.asarray(d_eff, np.float32)[ix, iy, iz][:, :, None]


class _Factors:
    """Every sampled patch's kept whitened basis ``U_b`` (n_vox, rank_b), grouped by rank.

    ``U_b`` has orthonormal columns: the left singular vectors of the whitened patch above
    the MP edge. In data units the patch's kept projector is ``Q_b = A_b Bm_b^T`` with
    ``A_b = C^(1/2) U_b`` and ``Bm_b = C^(-1/2) U_b``.

    No padding to a common rank: patches of equal rank are stacked, and ``U_cat`` holds
    every real column side by side, so the large products touch nothing that is zero.
    """

    def __init__(self, parts, rank):
        self.n_b = len(parts)
        self.rank = np.asarray(rank)
        self.groups = []                      # (column offset, k, (B_k, n_vox, k))
        cols, off = [], 0
        for k in np.unique(self.rank):
            if k == 0:
                continue
            Ug = np.stack([parts[b] for b in np.flatnonzero(self.rank == k)])
            self.groups.append((off, int(k), Ug))
            cols.append(Ug.transpose(1, 0, 2).reshape(Ug.shape[1], -1))
            off += Ug.shape[0] * int(k)
        # No patch may keep anything, as on pure noise: then there is no column at all.
        self.U_cat = np.concatenate(cols, axis=1) if cols else \
            np.zeros((parts[0].shape[0], 0), parts[0].dtype)          # (n_vox, sum rank)


def _split_chunk(patches, C, algorithm, tile=TILE):
    """Whiten and split a batch of patches ``tile`` at a time, in float32.

    Each tile's bulk product is taken in float64 and summed in float64.

    Returns:
        ``(bulk, parts, rank)``: (n_vox, n_vox) whitened bulk summed over the batch, each
        ``Rw Rw^T / (N sigma_mppca^2)``; a list of each patch's kept basis ``U_b``
        (n_vox, rank_b); (B,) ranks.
    """
    n_b, n_vox, n_vols = patches.shape
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    whitener = sqrtm_psd(C, -0.5).astype(np.float32)
    bulk = np.zeros((n_vox, n_vox))
    parts, ranks = [], []
    for start in range(0, n_b, tile):
        whitened, evals, evecs, rank, sigma_mppca = _whiten_split(
            np.asarray(patches[start:start + tile], np.float32), whitener, n_small, n_large,
            algorithm)
        r = int(rank.max())
        V = evecs[:, :, :r]
        keep = (np.arange(r)[None, :] < rank[:, None])[:, None, :]
        coords = np.where(keep, np.matmul(whitened, V), 0.0)              # (b, n_vox, r)
        resid = whitened - np.matmul(coords, V.transpose(0, 2, 1))
        scaled = resid / (np.sqrt(n_vols) * sigma_mppca)[:, None, None]
        flat = scaled.transpose(1, 0, 2).reshape(n_vox, -1).astype(np.float64)
        bulk += flat @ flat.T
        U = coords / np.sqrt(np.maximum(evals[:, :r], 1e-12))[:, None, :]
        parts.extend(U[b, :, :rank[b]] for b in range(U.shape[0]))
        ranks.append(rank)
    return bulk, parts, np.concatenate(ranks)


def _assemble(C, chunks):
    """Pool the chunks' bulks and collect their factors, in chunk order.

    Returns:
        ``(S, factors)``: (n_vox, n_vox) pooled bulk covariance in data units per unit
        noise variance, and the :class:`_Factors`.
    """
    root = sqrtm_psd(C, 0.5)
    bulk = sum(chunk[0] for chunk in chunks)
    parts = [u for chunk in chunks for u in chunk[1]]
    rank = np.concatenate([chunk[2] for chunk in chunks])
    return root @ bulk @ root / rank.size, _Factors(parts, rank)


def _split(patches, C, algorithm):
    """Whiten every patch by ``C``, split it, and pool the bulk. Returns ``(S, factors)``."""
    return _assemble(C, [_split_chunk(patches, C, algorithm)])


def _split_task(dwi, levels, centers, patch, C, algorithm):
    """One worker's share of a round: gather its patches and split them."""
    return _split_chunk(_gather(dwi, levels, centers, patch), C, algorithm)


class _WhitenedBulk:
    """``Lw(X) = mean_b P_b X P_b``, ``P_b = I - U_b U_b^T``: the bulk map in whitened units.

    Symmetric and positive semi-definite in the Frobenius inner product. Two large GEMMs
    over the real columns, and per rank group two batched products on strided views of
    their results, never copies.

    The per-patch products are many small GEMMs, which OpenBLAS runs on one thread each.
    They run in chunks of :data:`PRODUCT_CHUNK` patches on up to :data:`PRODUCT_THREADS`
    Python threads, never more than ``workers``, each chunk writing its own columns, so
    the thread count cannot change the result. numpy releases the GIL around them, and
    at 8 threads OpenBLAS barely queues them (see :mod:`camppca.pool`). On the crop at
    7^3 the products took 14.5 ms on one thread, 5.6 ms on 4 and 5.3 ms on 8.
    """

    def __init__(self, factors, threads=PRODUCT_THREADS, dtype=np.float32):
        self.threads = int(threads)
        self.n_b = factors.n_b
        self.groups = [(off, k, np.ascontiguousarray(Ug, dtype),
                        np.ascontiguousarray(Ug.transpose(0, 2, 1), dtype))
                       for off, k, Ug in factors.groups]
        self.U_cat = np.ascontiguousarray(factors.U_cat, dtype)
        self.Um = self.U_cat @ self.U_cat.T / self.n_b                     # mean U_b U_b^T
        self.Z = np.empty_like(self.U_cat)
        self.chunks = [(off + s * k, k, Ug[s:s + PRODUCT_CHUNK], Ugt[s:s + PRODUCT_CHUNK])
                       for off, k, Ug, Ugt in self.groups
                       for s in range(0, Ug.shape[0], PRODUCT_CHUNK)]

    def _products(self, Y, chunk):
        off, k, Ug, Ugt = chunk
        n = Y.shape[0]
        cols = slice(off, off + Ug.shape[0] * k)
        y = Y[:, cols].reshape(n, -1, k).transpose(1, 0, 2)
        z = self.Z[:, cols].reshape(n, -1, k).transpose(1, 0, 2)
        np.matmul(Ug, np.matmul(Ugt, y), out=z)

    def __call__(self, X):
        Y = X @ self.U_cat
        products = partial(self._products, Y)
        if self.threads > 1:                     # each chunk writes its own columns
            if self.threads not in _product_pools:
                _product_pools[self.threads] = ThreadPoolExecutor(self.threads)
            list(_product_pools[self.threads].map(products, self.chunks))
        else:
            for chunk in self.chunks:
                products(chunk)
        return X - self.Um @ X - X @ self.Um + (self.Z @ self.U_cat.T) / self.n_b


def _fit_table(S, factors, C, patch, lag, dzmax, start, threads=PRODUCT_THREADS):
    """The stationary table whose pooled bulk best fits ``S``, far lags pinned.

    One value per lag class, least squares in whitened units::

        min_v | Lw(W E(v) W) - W S W |,    E(v) = sum_c v_c E_c,   W = C^(-1/2)

    The one direction the bulk cannot see, the same value added to every lag, is fixed
    by making the mean over the far classes zero, those on the edge of the patch's lag
    box. The values are solved for on that pinned space, in an orthonormal basis.

    Conjugate gradients on the normal equations, two bulk maps a step in float32,
    started from ``start``'s values. Preconditioned by the same normal
    matrix with every patch's removal replaced by the average one,
    ``X -> Pm X Pm`` with ``Pm = I - mean_b U_b U_b^T``. That matrix is
    ``A^T (G (x) G) A`` with ``G = W Pm^2 W`` and ``A`` the class indicators, one product
    ``G E_c G`` per class and no bulk map. On the crop's factors at 7^3 it cut the
    condition number of the normal matrix from 5614 to 1.6 and the steps from 154 to 4,
    built in 0.25 s. Dropping the removal altogether left 125 and 19 steps.

    Returns:
        the correlation pattern, see :func:`~camppca.patch.table_index`, with unit zero lag. Lags
        the patch never spans are 0.
    """
    whitener64 = sqrtm_psd(C, -0.5)
    whitener = whitener64.astype(np.float32)
    op = _WhitenedBulk(factors, threads)
    cls, lags = _lag_classes(patch)
    n_cls, flat = len(lags), cls.ravel()
    reach = np.abs(lags).max(0)
    far = np.any((np.abs(lags) == reach) & (reach > 0), axis=1)
    basis = null_space(far[None, :].astype(np.float64))        # every v with far mean 0

    def forward(y):
        return op(whitener @ (basis @ y).astype(np.float32)[cls] @ whitener)

    def adjoint(R):
        X = whitener @ op(R) @ whitener
        return basis.T @ np.bincount(flat, weights=X.ravel(), minlength=n_cls)

    removed = np.eye(len(C)) - op.Um.astype(np.float64)
    G = whitener64 @ removed @ removed @ whitener64
    M = np.empty((n_cls, n_cls))
    for c in range(n_cls):
        M[c] = np.bincount(flat, weights=(G @ (cls == c).astype(np.float64) @ G).ravel(),
                           minlength=n_cls)
    pre = cho_factor(basis.T @ (0.5 * (M + M.T)) @ basis)

    rhs = whitener @ S.astype(np.float32) @ whitener
    y = basis.T @ start[lags[:, 0] + lag, lags[:, 1] + lag, lags[:, 2]]
    R = rhs - forward(y)
    g = adjoint(R)
    h = cho_solve(pre, g)
    p, gh = h.copy(), float(g @ h)
    norm0 = float(np.linalg.norm(adjoint(rhs)))
    for _ in range(MAX_FIT_STEPS):
        if np.linalg.norm(g) < FIT_TOL * norm0:
            break
        q = forward(p)
        alpha = gh / float(np.sum(q * q, dtype=np.float64))
        y += alpha * p
        R -= alpha * q
        g = adjoint(R)
        h = cho_solve(pre, g)
        gh_new = float(g @ h)
        p = h + (gh_new / gh) * p
        gh = gh_new
    else:
        warnings.warn(f"loop fit stopped at {MAX_FIT_STEPS} steps, gradient "
                      f"{np.linalg.norm(g) / norm0:.1e} against {FIT_TOL}",
                      RuntimeWarning, stacklevel=3)
    values = basis @ y
    table = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    for val, (dx, dy, dz) in zip(values, lags, strict=True):
        table[dx + lag, dy + lag, dz] = val
        if dz == 0:                                 # the dz = 0 plane holds both signs
            table[-dx + lag, -dy + lag, 0] = val
    return table / table[lag, lag, 0]


def _spanned(patch, lag, dzmax):
    """Bool table: the ``rho`` entries some voxel pair in the patch reads."""
    off = patch_offsets(patch)
    d = off[:, None, :] - off[None, :, :]
    spanned = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1), bool)
    spanned[table_index(d[..., 0], d[..., 1], d[..., 2], lag)] = True
    return spanned


def _lag_classes(patch):
    """Every voxel pair's lag class, and the (dx, dy, dz) of each class, dz >= 0.

    ``h`` and ``-h`` share a class, the one symmetry every correlation has. 364 classes
    at 5^3 besides lag zero, 1098 at 7^3.
    """
    off = patch_offsets(patch)
    d = off[:, None, :] - off[None, :, :]
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    flip = (dz < 0) | ((dz == 0) & ((dx < 0) | ((dx == 0) & (dy < 0))))
    dx, dy, dz = np.where(flip, -dx, dx), np.where(flip, -dy, dy), np.where(flip, -dz, dz)
    lags, cls = np.unique(np.stack([dx, dy, dz], -1).reshape(-1, 3), axis=0,
                          return_inverse=True)
    return cls.reshape(dx.shape), lags
