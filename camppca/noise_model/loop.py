"""The loop: refine rho until it explains the MPPCA bulk of every patch at once.

MPPCA splits each whitened patch into kept directions and a bulk. With the split fixed,
the bulk is noise, and the loop fits its correlation by maximum likelihood. Patch b,
whitened by ``W = C^(-1/2)``, keeps ``r_b`` orthonormal directions ``U_b``. Its residual
``res_b = P_b W X_b``, ``P_b = I - U_b U_b^T``, has ``nu_b = N - r_b`` independent columns,
each Gaussian with covariance ``s_b P_b M P_b``, ``M = W C' W``, ``C'`` the stationary
patch correlation sought and ``s_b`` the patch's level. The levels are nuisances and
are profiled out, so over the patches, with ``d_b = n_vox - r_b``::

    l = -1/2 sum_b nu_b [ log det+(P_b M P_b) + d_b log tr((P_b M P_b)+ res_b res_b^T) ]

``C'`` has one value per lag class, the voxel pairs that share a lag up to its sign,
``h`` and ``-h``: 364 classes at 5^3, 1098 at 7^3, against 58,996 unknowns for a free
covariance. Its zero lag is 1: with the levels free, the scale is not identified.
MPPCA's own level for a patch is not used, only its rank and kept directions.

One round whitens every sampled patch with the current ``C``, splits it with the
denoiser's own code (:func:`~camppca.denoiser.camppca._whiten_split`), and takes one
Fisher scoring step on ``l`` (:func:`_fisher_step`). A backtracking line search halves
the step until ``l`` rises by :data:`ARMIJO` of the predicted amount and ``C'`` stays a
valid correlation (:data:`~camppca.noise_model.MIN_EIGENVALUE`). ``l`` is minus
infinity at a singular ``C'``. The next round splits again with the new ``C``.
Directions a patch kept are not filled in with the current pattern's values; that is
what keeps the loop from drifting.

The hole. Every patch keeps its smooth patch-mean pattern as signal, so the bulk barely
sees the noise along it. Among stationary covariances this is exactly one direction, the
same value added to every lag. With the zero lag held at 1 it becomes
``1 - rho`` at each lag, the same value only near white noise. It is fixed by an
assumption, noise correlation dies off with distance, so the mean over the far classes,
those on the edge of the patch's lag box, is held at its starting value. That is zero
for white noise and for every table the loop returns, the only starts it is given.

The split itself is biased, and no fit undoes it. Signal components near the MP edge
are kept with patterns tilted toward the noise, and weaker ones stay in the bulk. Both
are smooth in space and read as extra correlation. The fixed point enlarges the bias of
a single round: if a round moves a table by ``b`` plus a share ``a`` of its distance from
the fixed point, the fixed point lies ``b / (1 - a)`` from the truth.

The rounds stop on a full step that moves no table value by more than :data:`LOOP_TOL`:
a shorter step moves it little because it is short. The table of the last round is
returned. A full step inside the valid correlations that no fraction of raises the
likelihood beyond rounding also ends the loop, as converged.

The stationary model does not explain the bulk, and the loop falls back to white noise,
when the full step leaves the valid correlations and no shorter one down to
:data:`MIN_STEP` raises the likelihood, or when :data:`MAX_ROUNDS` end with the last full
step outside them. Near a regular optimum Fisher scoring takes full steps inside them.
The real part of data whose phase was never stabilized ends this way. A loop whose last step was
shortened by the likelihood alone keeps its table and warns, as one that has not
converged on full steps does. A step that is not finite or does not point uphill, or a
split that is not finite, is a numerical failure and falls back saying so: a solved
Fisher step is finite and points uphill.

Patch rows are divided by the guarded relative noise map before whitening, exactly as the
denoiser divides them, so the loop fits ``C`` to the patches the denoiser whitens. The
MP rule (``algorithm``) and whether rows are divided by ``d`` (``noise_field``) are the
denoiser's settings, passed in, so the pattern is fitted to the split it will serve.

Lags the patch never spans, such as ``|dx| > 2`` for a 3 x 7 patch whose table reaches
lag 6, carry no information from the bulk. They keep the starting table's values.

The split runs in float32, with each tile's bulk product in float64, and the Fisher step
in float32 too, unless the smallest eigenvalue of ``C`` is below :data:`PRECISE_BELOW`.
Rounding in a float32 operator grows with the condition of ``C``. Each patch's kept directions are orthonormalized in float64, and the
line search evaluates the change in likelihood in float64, see :func:`_gain_chunk`.
"""
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial

import numpy as np
from joblib import delayed
from scipy.linalg import cho_factor, cho_solve, cholesky, null_space
from scipy.ndimage import binary_erosion

from ..denoiser.camppca import _whiten_split, sqrtm_psd
from ..denoiser.gather import TILE
from ..patch import patch_offsets, table_index
from ..pool import blas_threads, resolve_workers, worker_pool
from ..warn import warn
from . import (
    FALLBACK_REASON,
    MIN_EIGENVALUE,
    NUMERICAL_REASON,
    _build_C,
    _effective_d,
    _good_level,
    _white_rho,
)

__all__ = ["fit_rho"]

MAX_CENTERS = 5000      # patch centers the bulk is pooled over, at most
MIN_CENTERS = 500       # fewer tiles than this and the grid narrows, so patches overlap
LOOP_TOL = 5e-4         # stop when a full step moves no table value by more than this
MAX_ROUNDS = 20
LOOP_CHUNKS = 32        # fixed chunks of sampled patches, so workers cannot change the sums
FIT_TOL = 1e-5          # the fit stops at this residual, relative to its right-hand side
MAX_FIT_STEPS = 300
MIN_STEP = 2.0 ** -10   # the shortest step the line search tries
ARMIJO = 1e-4           # the share of the predicted rise in likelihood a step must reach
PRECISE_BELOW = 1e-4    # the Fisher step runs in float64 below this smallest eigenvalue of C
PRODUCT_THREADS = 8     # threads for the bulk map's per-patch products
PRODUCT_CHUNK = 64      # patches per thread task

_product_pools = {}     # one thread pool per thread count, kept: starting one costs time


def fit_rho(dwi, mask, d, rho_start, patch, lag, algorithm="cordero-grande",
                noise_field=True, workers=None, sample=None, centers=None):
    """Refine rho until it is consistent with the MPPCA bulk of the patches.

    Args:
        dwi: (X, Y, Z, N) data. Every volume enters, as every volume enters the denoiser.
        mask: (X, Y, Z) bool.
        d: (X, Y, Z) relative noise map. Guarded by ``_effective_d`` as the denoiser guards it.
        rho_start: (2*lag+1, 2*lag+1, dzmax+1) table to start from, a valid correlation.
        patch: the patch the pattern must serve, as a length-3 int array.
        lag: in-plane reach of ``rho_start``.
        algorithm: the MP rule the denoiser will cut with.
        noise_field: whether the denoiser will divide rows by ``d``.
        workers: CPU processes the sampled patches are split over each round, see
            :mod:`camppca.pool`. Each gathers and splits its own share, and evaluates
            its share of the likelihood for the line search. The fit stays in this
            process.
        sample: (X, Y, Z) bool, the voxels patch centers are taken from, or None for the
            mask voxels whose level passes the floor of ``_effective_d``. The level rounds
            pass the first one, so every round fits the same patches.
        centers: the patch centers as index arrays, as :func:`_sample_centers` returns
            them, or None to sample them from ``sample``. The QC fits subsets of the
            centers, see :mod:`camppca.noise_model.qc`.

    Returns:
        ``(rho, info)``: the stationary table, shaped like ``rho_start`` with unit zero
        lag and ``rho_start``'s values at lags the patch never spans, and a dict with
        ``rounds``, ``converged``, ``steps`` (each round's line-search step, 1 for a
        full one, 0 for a round that found the table at the optimum), ``unsolved`` (the rounds whose Fisher step stopped unsolved), ``sphericity`` (that of the returned table's patch correlation),
        ``n_centers``, ``stride`` (of the grid of patch centers, see
        :func:`_sample_centers`, None when ``centers`` is given) and ``fallback``: None,
        or why the table fell back to white noise.

    Raises:
        ValueError: if no mask voxel has its whole patch inside the mask, among the
            voxels whose level passes the floor of ``_effective_d``.
    """
    patch = np.asarray(patch, int)
    dzmax = rho_start.shape[2] - 1
    workers = resolve_workers(workers)
    # Patches with a level below the floor hold near-constant data, which the fit cannot use.
    if centers is None:
        if sample is None:
            sample = _good_level(d, mask)
        centers, stride = _sample_centers(sample, patch)
    else:
        stride = None
    levels = _effective_d(d, mask) if noise_field else np.ones(mask.shape, np.float32)

    # Fixed chunks, pooled in chunk order, so the model does not depend on ``workers``.
    slices = np.array_split(np.arange(centers[0].size), min(LOOP_CHUNKS, centers[0].size))
    with ExitStack() as stack:
        if workers > 1:
            # One pool for every round: joblib maps the volume and the levels once.
            parallel = stack.enter_context(worker_pool(workers))
            chunks = [tuple(c[part] for c in centers) for part in slices]

            def split(C):
                return parallel(delayed(_split_task)(dwi, levels, chunk, patch, C, algorithm)
                                for chunk in chunks)

            def gain(W, G, results):
                return parallel(delayed(_gain_task)(dwi, levels, chunk, patch, W, G,
                                                    res[1], res[3])
                                for chunk, res in zip(chunks, results, strict=True))
        else:
            patches = _gather(dwi, levels, centers, patch)

            def split(C):
                with blas_threads(1):          # per-patch work, as in the workers
                    return [_split_chunk(patches[part], C, algorithm) for part in slices]

            def gain(W, G, results):
                with blas_threads(1):
                    return [_gain_chunk(patches[part], W, G, res[1], res[3])
                            for part, res in zip(slices, results, strict=True)]

        cls, lags = _lag_classes(patch)
        rho = rho_start
        C = _build_C(rho, patch, lag)
        converged, fallback, steps, unsolved, bounded = False, None, [], {}, False
        for rounds in range(1, MAX_ROUNDS + 1):  # noqa: B007 - read after the loop
            results = split(C)
            bulk, factors = _assemble(results)
            W = sqrtm_psd(C, -0.5)
            outcome = _line_search(
                bulk, factors, W, rho, patch, lag, cls, lags,
                lambda G, W=W, results=results: sum(gain(W, G, results)),
                threads=min(PRODUCT_THREADS, workers))
            if outcome.residual is not None:
                unsolved[rounds] = outcome.residual
            if outcome.values is None:
                # The full step left the valid correlations and no shorter one helped:
                # the stationary model does not explain the bulk. Anything else is
                # numerical, and blaming the data would mislead.
                reason = FALLBACK_REASON if outcome.bounded else NUMERICAL_REASON
                fallback = (f"the loop found no step that raises the likelihood of the "
                            f"noise bulk in round {rounds}: {outcome.why}. {reason}")
                warn(fallback)
                rho = _white_rho(rho.shape)
                break
            if outcome.step == 0.0:
                # A full step inside the valid correlations, and no fraction of it raises
                # the likelihood beyond rounding: the table is at the optimum.
                steps.append(0.0)
                converged = True
                break
            steps.append(outcome.step)
            bounded = outcome.bounded
            change = float(np.abs(outcome.values - _class_values(rho, lags, lag)).max())
            rho = _table_from_values(outcome.values, lags, lag, dzmax)
            C = _build_C(rho, patch, lag)
            if change < LOOP_TOL and outcome.step == 1.0:
                converged = True
                break
        if not converged and fallback is None and bounded:
            # Near a regular optimum Fisher scoring takes full steps inside the valid
            # correlations. Still cut by them in the last round, the pattern is pressing
            # against them.
            fallback = (f"the loop did not converge in {MAX_ROUNDS} rounds, and its last "
                        f"full step left the valid correlations. {FALLBACK_REASON}")
            warn(fallback)
            rho = _white_rho(rho.shape)
        elif not converged and fallback is None:
            warn(f"the loop did not converge in {MAX_ROUNDS} rounds; the last round "
                 f"moved the table by up to {change:.1e} with a step of {steps[-1]:g}, "
                 f"against a tolerance of {LOOP_TOL:g} on a full step")
        # After a fallback the unsolved steps are a symptom of the same misfit, already
        # reported. A kept table reports them once.
        if unsolved and fallback is None:
            warn(f"the Fisher step's solve stopped at {MAX_FIT_STEPS} conjugate-gradient "
                 f"steps in round(s) {sorted(unsolved)}, residual up to "
                 f"{max(unsolved.values()):.1e} against {FIT_TOL}. The line search still "
                 f"accepted each step only where it raised the likelihood.")
    rho = rho.copy()
    unspanned = ~_spanned(patch, lag, dzmax)
    rho[unspanned] = rho_start[unspanned]
    info = dict(rounds=rounds, converged=converged, steps=steps, fallback=fallback,
                unsolved=sorted(unsolved),
                sphericity=_sphericity_excess(_build_C(rho, patch, lag)),
                n_centers=int(centers[0].size), stride=stride)
    return rho, info


_Outcome = namedtuple("_Outcome", "step values why residual bounded")


def _line_search(bulk, factors, W, rho, patch, lag, cls, lags, gain,
                 threads=PRODUCT_THREADS):
    """One Fisher scoring step from ``rho``, halved until the likelihood rises enough.

    Args:
        bulk: (n_vox, n_vox) ``sum_b S_b``, each patch's whitened residual scatter
            ``res_b res_b^T`` at its own profiled level, see :func:`_split_chunk`.
        factors: the round's :class:`_Factors`.
        W: ``C^(-1/2)`` of the round's ``C``, the patch correlation of ``rho``.
        gain: maps ``G = W C' W``, ``C'`` a trial patch correlation, to the rise in
            profile log-likelihood from ``rho``, see :func:`_gain_chunk`.

    Returns:
        an ``_Outcome``: ``step``, the step taken, 0 when the table is already at the
        optimum; ``values``, the class values of the new table, unit at zero lag, or None
        when no step was found, with ``why``; ``residual``, the relative residual of the
        Fisher step's solve where it stopped unsolved, else None; ``bounded``, whether
        the full step left the valid correlations.
    """
    n_vox = len(W)
    current = _class_values(rho, lags, lag)
    if not np.isfinite(bulk).all():
        return _Outcome(None, None, "the split of the whitened patches is not finite",
                        None, False)
    rhs = bulk - factors.total * (np.eye(n_vox) - factors.Um)    # sum_b (S_b - nu_b P_b)
    # Rounding in a float32 operator grows with the condition of C, 1 / smallest eigenvalue.
    precise = float(np.linalg.eigvalsh(W)[-1]) ** -2 < PRECISE_BELOW
    delta, residual = _fisher_step(rhs / factors.total, factors, W, patch, threads,
                                   dtype=np.float64 if precise else np.float32)
    if not np.isfinite(delta).all():
        return _Outcome(None, None, "the Fisher step is not finite", residual, False)
    if not delta.any():
        return _Outcome(0.0, current, None, residual, False)
    slope = 0.5 * float(np.sum((W @ delta[cls] @ W) * rhs))
    if not slope > 0:
        # The Fisher operator is positive semi-definite, so a solved step points uphill.
        return _Outcome(None, None, f"the step does not point uphill, slope {slope:.2e}",
                        residual, False)
    full = float(np.linalg.eigvalsh((current + delta)[cls])[0])
    bounded = full < MIN_EIGENVALUE
    step = 1.0
    while step >= MIN_STEP:
        values = current + step * delta
        if (step == 1.0 and not bounded) or \
                float(np.linalg.eigvalsh(values[cls])[0]) >= MIN_EIGENVALUE:
            if gain(W @ values[cls] @ W) >= ARMIJO * step * slope:
                return _Outcome(step, values, None, residual, bounded)
        step /= 2
    if not bounded:
        return _Outcome(0.0, current, None, residual, False)
    return _Outcome(None, None, (f"none down to {MIN_STEP:g} did; the full step's C has "
                                 f"smallest eigenvalue {full:.1e}"), residual, True)


def _sphericity_excess(C):
    """The sphericity n tr(C^2) / tr(C)^2 - 1 of a unit-diagonal C; zero when C is white."""
    return float((C ** 2).sum() / len(C) - 1.0)


def _sample_centers(mask, patch, min_centers=None):
    """Patch centers on a grid, each patch wholly inside the mask.

    The stride is the patch, so the patches tile the mask without overlap and each brings
    its own noise. Of the grid's phases the one with the most centers is taken. Above
    :data:`MAX_CENTERS` the stride widens until under it, below :data:`MIN_CENTERS` it
    narrows until over it, down to every voxel. Every axis scales by the same factor, in
    steps of one voxel on the longest, so the grid keeps the patch's shape.

    A grid ties the count to the mask and needs no seed.

    ``min_centers`` None is :data:`MIN_CENTERS`. 0 keeps the tiling however few centers
    it has. The QC needs that: overlapping patches share noise, so a region of them counts
    for less than the same number spread over the mask.

    Returns:
        ``(centers, stride)``: the centers as index arrays, C order, and the stride used.
    """
    eligible = binary_erosion(mask, structure=np.ones(tuple(patch), bool), border_value=0)
    if not eligible.any():
        raise ValueError(f"no mask voxel has its whole {tuple(int(v) for v in patch)} "
                         "patch inside the mask, so the loop has no patch to fit")
    longest = int(max(patch))
    min_centers = MIN_CENTERS if min_centers is None else min_centers

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
    while centers.sum() < min_centers and max(stride) > 1:
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
    ``A_b = C^(1/2) U_b`` and ``Bm_b = C^(-1/2) U_b``. Each patch weighs ``nu_b``, the
    independent columns of its residual, ``N - rank_b``, or 0 for a patch with no bulk
    to fit (``weights``, their sum ``total``). ``Um`` is the weighted mean of
    ``U_b U_b^T``, and ``bulk_dim`` holds ``d_b = n_vox - rank_b``.

    No padding to a common rank: patches of equal rank are stacked, and ``U_cat`` holds
    every real column side by side, so the large products touch nothing that is zero.
    """

    def __init__(self, parts, rank, nu):
        n_vox = parts[0].shape[0]
        self.n_b = len(parts)
        self.rank = np.asarray(rank)
        self.weights = np.asarray(nu, np.float64)
        self.total = float(self.weights.sum())
        self.bulk_dim = (n_vox - self.rank).astype(np.float64)
        self.groups = []                      # (column offset, k, patch indices, (B_k, n_vox, k))
        cols, col_patch, off = [], [], 0
        for k in np.unique(self.rank):
            if k == 0:
                continue
            members = np.flatnonzero(self.rank == k)
            Ug = np.stack([parts[b] for b in members])
            self.groups.append((off, int(k), members, Ug))
            cols.append(Ug.transpose(1, 0, 2).reshape(n_vox, -1))
            col_patch.append(np.repeat(members, int(k)))
            off += Ug.shape[0] * int(k)
        # No patch may keep anything, as on pure noise: then there is no column at all.
        self.U_cat = np.concatenate(cols, axis=1) if cols else \
            np.zeros((n_vox, 0), parts[0].dtype)                     # (n_vox, sum rank)
        self.col_patch = np.concatenate(col_patch) if col_patch else np.zeros(0, int)
        U64 = self.U_cat.astype(np.float64)
        self.Um = (U64 * (self.weights[self.col_patch] / self.total)) @ U64.T


def _split_chunk(patches, C, algorithm, tile=TILE):
    """Whiten and split a batch of patches ``tile`` at a time, in float32.

    Each patch's level is profiled out: its residual scatter is scaled so that its trace
    is ``nu_b d_b``, the level at which the patch alone is most likely under the current
    ``C``. MPPCA's own level for the patch is not used, only its rank and kept
    directions. A patch that keeps nearly every component has a bulk of a few eigenvalues
    below float32 resolution next to its signal, and a plugged-in level near zero made
    its scatter 1e10 times too large. Profiled, its weight is its few independent
    columns. Each tile's bulk product is taken in float64 and summed in float64.

    Returns:
        ``(bulk, parts, rank, nu)``: (n_vox, n_vox) ``sum_b S_b`` over the batch; a list
        of each patch's kept basis ``U_b`` (n_vox, rank_b); (B,) ranks; (B,) weights
        ``nu_b``, 0 for a patch with no bulk to fit.
    """
    n_b, n_vox, n_vols = patches.shape
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    whitener = sqrtm_psd(C, -0.5).astype(np.float32)
    bulk = np.zeros((n_vox, n_vox))
    parts, ranks, nus = [], [], []
    for start in range(0, n_b, tile):
        whitened, evals, evecs, rank, _ = _whiten_split(
            np.asarray(patches[start:start + tile], np.float32), whitener, n_small, n_large,
            algorithm)
        r = int(rank.max())
        V = evecs[:, :, :r]
        keep = (np.arange(r)[None, :] < rank[:, None])[:, None, :]
        coords = np.where(keep, np.matmul(whitened, V), 0.0)              # (b, n_vox, r)
        # Orthonormal in float64, column by column, so each P_b is a projector of rank
        # n_vox - r_b: the Fisher operator is positive semi-definite only then. Taken by
        # the eigenvalues, the kept columns of a patch that keeps nearly everything had
        # lengths far from 1, and round 1 under strong correlation went indefinite.
        U = np.where(keep, np.linalg.qr(coords.astype(np.float64))[0], 0.0).astype(np.float32)
        resid = whitened - np.matmul(U, np.matmul(U.transpose(0, 2, 1), whitened))
        energy = np.einsum("bpn,bpn->b", resid, resid, dtype=np.float64)
        nu = (n_vols - rank).astype(np.float64)
        dof = nu * (n_vox - rank)
        usable = (energy > 0) & (dof > 0)
        nu = np.where(usable, nu, 0.0)
        level = np.sqrt(np.where(usable, dof / np.where(usable, energy, 1.0), 0.0))
        flat = (resid * level[:, None, None].astype(np.float32)).transpose(1, 0, 2)
        flat = flat.reshape(n_vox, -1).astype(np.float64)
        bulk += flat @ flat.T
        parts.extend(U[b, :, :rank[b]] for b in range(U.shape[0]))
        ranks.append(rank)
        nus.append(nu)
    return bulk, parts, np.concatenate(ranks), np.concatenate(nus)


def _assemble(chunks):
    """Pool the chunks' bulks and collect their factors, in chunk order.

    Returns:
        ``(bulk, factors)``: (n_vox, n_vox) ``sum_b S_b`` in whitened units, and the
        :class:`_Factors`.
    """
    bulk = sum(chunk[0] for chunk in chunks)
    parts = [u for chunk in chunks for u in chunk[1]]
    rank = np.concatenate([chunk[2] for chunk in chunks])
    nu = np.concatenate([chunk[3] for chunk in chunks])
    return bulk, _Factors(parts, rank, nu)


def _split(patches, C, algorithm):
    """Whiten every patch by ``C``, split it, and pool the bulk. Returns ``(bulk, factors)``."""
    return _assemble([_split_chunk(patches, C, algorithm)])


def _split_task(dwi, levels, centers, patch, C, algorithm):
    """One worker's share of a round: gather its patches and split them."""
    return _split_chunk(_gather(dwi, levels, centers, patch), C, algorithm)


def _gain_chunk(patches, W, G, parts, nu):
    """A batch's share of the rise in profile log-likelihood from ``M = I`` to ``M = G``.

    Patch b contributes ``-1/2 nu_b [log det+(P G P) + d_b log tr((P G P)+ res res^T)]``,
    its level profiled out, and ``M = I`` at the current table. For orthonormal
    ``[U, V]``, ``V^T G V`` is ``P G P`` on the bulk, with
    ``det(V^T G V) = det(G) det(U^T G^-1 U)`` and inverse, on the bulk,
    ``G^-1 - G^-1 U (U^T G^-1 U)^-1 U^T G^-1``. With ``E = G^-1 - I``, and ``res`` in the
    bulk so that ``U^T G^-1 res = U^T E res``, the rise is::

        -1/2 nu [log det G + log det(I + U^T E U) + d log(1 + t / |res|^2)]
        t = <res, E res> - tr((I + U^T E U)^-1 F F^T),   F = U^T E res

    ``G`` is the same for every patch, so its inverse and determinant are taken once,
    and each patch needs only products with ``E`` and an r x r solve. No two large
    values are subtracted: near convergence the rise is a millionth of the likelihood.
    ``G`` is positive definite wherever the line search evaluates it.

    Returns:
        the share.
    """
    n_vox = len(G)
    root = cholesky(G, lower=True, check_finite=False)
    log_det = 2.0 * np.log(np.diag(root)).sum()
    E = cho_solve((root, True), np.eye(n_vox), check_finite=False) - np.eye(n_vox)
    whitened = np.matmul(W.astype(np.float32), np.asarray(patches, np.float32))
    total = 0.0
    for Y, U, weight in zip(whitened, parts, nu, strict=True):
        if weight == 0:
            continue
        U = np.linalg.qr(U.astype(np.float64))[0]        # orthonormal to float64 precision
        res = Y.astype(np.float64)
        res -= U @ (U.T @ res)
        E_res = E @ res
        t = float(np.sum(res * E_res))
        kept = 0.0
        if U.shape[1]:
            inner = np.eye(U.shape[1]) + U.T @ (E @ U)
            F = U.T @ E_res
            kept = np.linalg.slogdet(inner)[1]
            t -= float(np.trace(np.linalg.solve(inner, F @ F.T)))
        total -= 0.5 * weight * (log_det + kept
                                 + (n_vox - U.shape[1]) * np.log1p(t / float(np.sum(res * res))))
    return total


def _gain_task(dwi, levels, centers, patch, W, G, parts, nu):
    """One worker's share of the rise: gather its patches again and evaluate."""
    return _gain_chunk(_gather(dwi, levels, centers, patch), W, G, parts, nu)


class _WhitenedBulk:
    """The Fisher operator of the profile likelihood at ``M = I``, in whitened units::

        F(X) = sum_b nu_b [P_b X P_b - tr(P_b X) P_b / d_b] / total,   P_b = I - U_b U_b^T

    The first term is the bulk map, the second removes each patch's level, which the
    likelihood profiles out. ``F(I) = 0``: the scale of a table is not identified, so the
    zero lag is held at 1. Symmetric and positive semi-definite in the Frobenius inner
    product. Two large GEMMs over the real columns, and per rank group two batched
    products on strided views of their results, never copies, then one more GEMM for
    the level term.

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
        share = factors.weights / factors.total
        self.share = share
        self.level_share = np.where(factors.bulk_dim > 0, share / np.maximum(
            factors.bulk_dim, 1.0), 0.0)                                 # nu_b / (d_b total)
        self.col_patch = factors.col_patch
        self.U_cat = np.ascontiguousarray(factors.U_cat, dtype)
        self.Um = factors.Um.astype(dtype)                       # weighted mean U_b U_b^T
        self.Z = np.empty_like(self.U_cat)
        self.chunks = []
        for off, k, members, Ug in factors.groups:
            Ug = np.ascontiguousarray(Ug, dtype)
            Ugt = np.ascontiguousarray(Ug.transpose(0, 2, 1), dtype)
            for s in range(0, Ug.shape[0], PRODUCT_CHUNK):
                w = share[members[s:s + PRODUCT_CHUNK]].astype(dtype)[:, None, None]
                self.chunks.append((off + s * k, k, w, Ug[s:s + PRODUCT_CHUNK],
                                    Ugt[s:s + PRODUCT_CHUNK]))

    def _products(self, Y, chunk):
        off, k, w, Ug, Ugt = chunk
        n = Y.shape[0]
        cols = slice(off, off + Ug.shape[0] * k)
        y = Y[:, cols].reshape(n, -1, k).transpose(1, 0, 2)
        z = self.Z[:, cols].reshape(n, -1, k).transpose(1, 0, 2)
        np.matmul(Ug, np.matmul(Ugt, y), out=z)
        z *= w

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
        bulk = X - self.Um @ X - X @ self.Um + self.Z @ self.U_cat.T
        # tr(P_b X) = tr(X) - sum of u^T X u over the patch's kept columns
        inside = np.bincount(self.col_patch, weights=np.einsum("ij,ij->j", self.U_cat, Y),
                             minlength=self.n_b)
        coef = self.level_share * (float(np.trace(X)) - inside)
        level = coef.sum() * np.eye(len(X), dtype=X.dtype) - \
            (self.U_cat * coef[self.col_patch].astype(X.dtype)) @ self.U_cat.T
        return bulk - level


def _fisher_step(rhs, factors, W, patch, threads=PRODUCT_THREADS, dtype=np.float32):
    """The Fisher scoring step on the profile likelihood, in class values.

    The change ``delta`` of the table, one value per lag class, that solves::

        <W E_c W, F(W E(delta) W)> = <W E_c W, rhs>   for every class c,   E(v) = sum_c v_c E_c

    with ``F`` the Fisher operator (:class:`_WhitenedBulk`) and ``rhs`` the gradient,
    ``sum_b (S_b - nu_b P_b) / total``. The zero lag does not change: ``F`` cannot see
    the scale. The direction the bulk barely sees, see the module docstring, is fixed by
    keeping the mean over the far classes unchanged, those on the edge of the patch's
    lag box. ``delta`` is solved for on that space, in an orthonormal basis.

    Conjugate gradients from zero, one Fisher operator a step in ``dtype``. Preconditioned
    by the bulk map with every patch's removal replaced by the average one,
    ``X -> Pm X Pm`` with ``Pm = I - Um``, without the level term. That matrix is
    ``A^T (G (x) G) A`` with ``G = W Pm W`` and ``A`` the class indicators, one product
    ``G E_c G`` per class and no bulk map.

    The operator is positive semi-definite, so every conjugate-gradient iterate from zero
    points uphill, ``g . y = y^T F y``. Where rounding breaks that, ``p . q`` not
    positive, the iterations stop at the last good iterate. If none is left, the step is
    the preconditioned gradient, which points uphill by construction.

    Returns:
        ``(delta, residual)``: the change in class values, see :func:`_lag_classes`, and
        the relative residual if the solve stopped short of :data:`FIT_TOL`, else None.
        The caller reports it, see :func:`fit_rho`.
    """
    W32 = W.astype(dtype)
    op = _WhitenedBulk(factors, threads, dtype=dtype)
    cls, lags = _lag_classes(patch)
    n_cls, flat = len(lags), cls.ravel()
    reach = np.abs(lags).max(0)
    far = np.any((np.abs(lags) == reach) & (reach > 0), axis=1)
    zero = (lags == 0).all(1)
    basis = null_space(np.stack([far, zero]).astype(np.float64))   # far mean 0, zero lag 0

    def normal(y):
        X = op(W32 @ (basis @ y).astype(dtype)[cls] @ W32)
        return basis.T @ np.bincount(flat, weights=(W32 @ X @ W32).ravel(), minlength=n_cls)

    G = W @ (np.eye(len(W)) - factors.Um) @ W
    M = np.empty((n_cls, n_cls))
    for c in range(n_cls):
        M[c] = np.bincount(flat, weights=(G @ (cls == c).astype(np.float64) @ G).ravel(),
                           minlength=n_cls)
    pre = _preconditioner(basis.T @ (0.5 * (M + M.T)) @ basis)

    g = basis.T @ np.bincount(flat, weights=(W32 @ rhs.astype(dtype) @ W32).ravel(),
                              minlength=n_cls)
    norm0 = float(np.linalg.norm(g))
    if norm0 == 0 or not np.isfinite(norm0):
        return basis @ np.where(np.isfinite(g), 0.0, np.nan), None
    y = np.zeros(basis.shape[1])
    resid = g.copy()
    h = pre(resid)
    p, rh = h.copy(), float(resid @ h)
    solved = False
    for _ in range(MAX_FIT_STEPS):
        if np.linalg.norm(resid) <= FIT_TOL * norm0:
            solved = True
            break
        q = normal(p)
        curvature = float(p @ q)
        if not (np.isfinite(curvature) and curvature > 0):
            break
        alpha = rh / curvature
        y += alpha * p
        resid -= alpha * q
        h = pre(resid)
        rh_new = float(resid @ h)
        p = h + (rh_new / rh) * p
        rh = rh_new
    if not y.any() or not float(g @ y) > 0:
        y = pre(g)
    return basis @ y, None if solved else float(np.linalg.norm(resid) / norm0)


def _preconditioner(M):
    """``r -> M^-1 r`` by Cholesky, or with ``M``'s eigenvalues clipped where it fails.

    ``M`` is singular when every patch keeps the same direction, as with few patches of
    strongly correlated noise: a 32 x 32 x 10 phantom crashed here in round 1. Only the
    speed of the conjugate gradients depends on it, not their answer, as long as they
    reach :data:`FIT_TOL`.
    """
    try:
        factor = cho_factor(M)
        return lambda r: cho_solve(factor, r, check_finite=False)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, 1e-8 * w.max())
        return lambda r: V @ ((V.T @ r) / w)


def _class_values(table, lags, lag):
    """A table's value at each lag class, see :func:`_lag_classes`."""
    return table[lags[:, 0] + lag, lags[:, 1] + lag, lags[:, 2]]


def _table_from_values(values, lags, lag, dzmax):
    """The table holding each class's value, 0 at lags the patch never spans."""
    table = np.zeros((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    for val, (dx, dy, dz) in zip(values, lags, strict=True):
        table[dx + lag, dy + lag, dz] = val
        if dz == 0:                                 # the dz = 0 plane holds both signs
            table[-dx + lag, -dy + lag, 0] = val
    return table


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
