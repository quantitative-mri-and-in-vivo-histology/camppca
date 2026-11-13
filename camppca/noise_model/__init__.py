"""The noise model: what the noise is, and how to estimate it from the data.

The patch noise covariance is assumed to factorize into a level that varies in space and
a correlation pattern that does not::

    Sigma = sigma^2 * D * C * D          D = diag(d)

``rho`` (the correlation pattern) and ``d`` (the relative noise map) are the model. ``C``,
the n_vox x n_vox patch correlation, follows from rho and a patch shape. The patch shape is
a denoiser setting and no property of the noise. So ``C`` is computed on demand by
:meth:`NoiseModel.patch_correlation` and never stored.

This half of the seam is source-specific: here both come from MPPCA itself. ``d`` is
standard MPPCA's noise level averaged over patch footprints, and rho comes from the loop (``loop.py``), started
from white. A future alignment-based estimator builds the same :class:`NoiseModel` a
different way and feeds the same engine, so nothing downstream may assume this one.

The loop runs MPPCA's own split, so it uses the denoiser's internals.

References:
    Moeller S, Pisharady PK, Ramanna S, et al. NOise reduction with DIstribution
    Corrected (NORDIC) PCA in dMRI with complex-valued parameter-free locally
    low-rank processing. NeuroImage 2021;226:117539. The noise map as an average
    over patch footprints.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

from ..patch import check_patch, patch_offsets, required_lag, resolve_patch, table_index
from ..pool import limit_blas
from ..progress import elapsed
from ..warn import warn
from .masks import brain_mask

log = logging.getLogger(__name__)

__all__ = ["NoiseModel", "estimate_noise_model", "brain_mask"]

#: Fewest volumes the Marchenko-Pastur fit is trusted with. Below this the width
#: normalizer of the cordero-grande rule degenerates as the rank approaches the volume
#: count, and the noise level comes out far too high.
MIN_VOLUMES = 50

#: Stop the level rounds when the median relative change of ``d`` drops below this.
LEVEL_TOL = 5e-3

#: Smallest eigenvalue a patch correlation may have. A correlation pattern can be valid for one
#: patch and indefinite for a larger one. `_build_C` then projects it to a C with
#: eigenvalues near zero, whitening divides by them, and sigma collapses to ~1e-7
#: everywhere without an error. The bound is a guard and set low: the true C of noise
#: filtered over all of k-space can come close to it.
MIN_EIGENVALUE = 1e-6


#: Levels below this fraction of the median over the mask count as bad. Whitening divides
#: by ``d``, so a patch of near-constant data, whose MP level is near zero, would blow up.
LEVEL_FLOOR = 1e-2

#: The level pass computes one patch every this many voxels on each axis, see
#: :func:`_footprint_average`. MPPCA's level is a patch mean, so a full grid adds little.
LEVEL_STRIDE = 2


def _good_level(d, mask):
    """Mask voxels whose level is finite and at least :data:`LEVEL_FLOOR` of the median."""
    d = np.asarray(d, np.float32)
    good = np.isfinite(d) & (d > 0) & np.asarray(mask, bool)
    if not good.any():
        raise ValueError("noise field d is empty or non-positive everywhere in the mask")
    return good & (d >= LEVEL_FLOOR * np.median(d[good]))


def _effective_d(d, mask):
    """The relative noise map as the engine uses it. The one place ``d`` is guarded.

    ``d`` is defined up to scale: rows are divided by it and the center voxel multiplied
    back, so a constant cancels in the denoised image, the rank and ``sigma`` in data
    units. Voxels that are non-finite, below :data:`LEVEL_FLOOR` times the median over the
    mask, or outside the mask get the level of the nearest good mask voxel. Edge patches
    include outside voxels, and by default they are denoised too, so a brain voxel the
    mask missed gets a local level.

    The engine takes the field as given and rescales the denoised image by it at the center
    voxel, but not ``sigma_mppca``. Putting sigma_mppca back into data units therefore means
    applying this same field, which is why the denoiser takes it from
    :meth:`NoiseModel.effective_d` and passes it down.
    """
    d = np.asarray(d, np.float32)
    good = _good_level(d, mask)
    nearest = ndimage.distance_transform_edt(~good, return_distances=False,
                                             return_indices=True)
    return d[tuple(nearest)].astype(np.float32)


@dataclass
class NoiseModel:
    """Spatial noise model: correlation pattern ``rho`` plus relative noise map ``d``."""

    rho: np.ndarray
    """(2*lag+1, 2*lag+1, dzmax+1) correlation pattern, unit at zero lag. Pattern only,
    it carries no level."""

    d: np.ndarray
    """(X, Y, Z) relative noise map, defined up to scale: standard MPPCA's
    ``sigma_mppca`` averaged over patch footprints. Only ratios matter. Zero outside
    ``mask``."""

    mask: np.ndarray
    """(X, Y, Z) bool, the voxels the model was estimated on."""

    lag: int
    """Largest in-plane lag ``rho`` covers. With ``rho.shape[2] - 1``, the through-slice
    one, it bounds the patch this model can serve."""

    provenance: dict = field(default_factory=dict)
    """The record of how the model was made. The denoiser never reads it."""

    fitted_to: dict = field(default_factory=dict)
    """The denoiser settings ``rho`` was fitted to, which the denoiser reads: ``patch``,
    ``algorithm`` and ``noise_field``. Any key may be missing, for a model whose ``rho``
    came from elsewhere."""

    def __post_init__(self):
        # A uint8 mask (what `nib.load(...).get_fdata().astype(...)` hands you) would turn
        # `... & mask` into integer fancy indexing downstream, silently mis-normalizing d.
        self.mask = np.asarray(self.mask, bool)
        self.rho = np.asarray(self.rho, float)
        # Both read silently wrong when broken: a table narrower or wider than the lag is
        # indexed off center, and C's diagonal is set to 1 whatever rho holds at zero lag,
        # so a covariance passed as rho would scale every off-diagonal entry.
        side = 2 * int(self.lag) + 1
        if self.rho.ndim != 3 or self.rho.shape[:2] != (side, side):
            raise ValueError(f"rho must be shaped ({side}, {side}, dzmax + 1) for lag "
                             f"{self.lag}, got {self.rho.shape}")
        if abs(self.rho[self.lag, self.lag, 0] - 1.0) > 1e-6:
            raise ValueError(f"rho must be a correlation, 1 at zero lag, got "
                             f"{self.rho[self.lag, self.lag, 0]:g}")

    def effective_d(self):
        """``d`` as the denoiser uses it, guarded: see :func:`_effective_d`."""
        return _effective_d(self.d, self.mask)

    def patch_correlation(self, patch):
        """Derive ``C`` (n_vox, n_vox), the unit-diagonal patch correlation.

        ``C`` is ``rho`` re-indexed into the geometry of this patch. A table that is not
        a valid correlation for it raises, so ``C`` is never a projected stand-in.
        Raises if the patch needs lags this model never measured: beyond the
        table, the index runs off it and numpy wraps around silently. Within the table,
        a model fitted to a patch fills only the lags that patch spans, so a patch wider
        on any axis than ``fitted_to["patch"]`` raises too: the other lags hold the white
        start, and C would call those voxels uncorrelated.  Also raises if ``C`` is near
        singular (see :data:`MIN_EIGENVALUE`).
        """
        patch = check_patch(patch)
        fitted = self.fitted_to.get("patch")
        if fitted is not None and any(p > f for p, f in zip(patch, fitted, strict=True)):
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} reaches lags this noise model was "
                f"not fitted at: it was fitted to {tuple(int(v) for v in fitted)}. "
                f"Re-estimate with this patch.")
        need = required_lag(patch)
        if need > self.lag:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs lag {need}, but this noise "
                f"model only covers lag {self.lag}. Re-estimate with the larger patch.")
        dzmax = self.rho.shape[2] - 1
        if patch[2] - 1 > dzmax:
            raise ValueError(
                f"patch {tuple(int(v) for v in patch)} needs through-slice lag "
                f"{patch[2] - 1}, but this noise model only covers {dzmax}. Re-estimate "
                f"with the deeper patch.")
        C = _build_C(self.rho, patch, self.lag)
        smallest = float(np.linalg.eigvalsh(C)[0])
        if smallest < MIN_EIGENVALUE:
            raise ValueError(
                f"rho is not a valid correlation for patch "
                f"{tuple(int(v) for v in patch)}: the smallest eigenvalue of C is "
                f"{smallest:.2e}, below {MIN_EIGENVALUE:g}. Whitening divides by it, and "
                f"sigma would collapse toward zero. Estimate the noise model at this "
                f"patch.")
        return C


def _table_C(rho, patch, lag):
    """The patch correlation a correlation pattern spells out, symmetric with unit diagonal, not
    projected: its smallest eigenvalue says whether the table is a valid correlation."""
    o = patch_offsets(patch)
    dx = o[:, 0][:, None] - o[:, 0][None, :]
    dy = o[:, 1][:, None] - o[:, 1][None, :]
    dz = o[:, 2][:, None] - o[:, 2][None, :]
    C = rho[table_index(dx, dy, dz, lag)]
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def _white_rho(shape):
    """A correlation pattern of white noise: 1 at zero lag, 0 elsewhere."""
    rho = np.zeros(shape)
    rho[shape[0] // 2, shape[1] // 2, 0] = 1.0
    return rho


#: Why the loop fell back to white noise. Signal below the MP edge is smooth and
#: alike in every patch, so pooled it reads as strong correlation that no stationary
#: table explains. An unstabilized phase leaves such signal in the real part.
FALLBACK_REASON = ("The noise bulk most likely holds signal, as an unstabilized phase "
                   "leaves in the real part. The noise model falls back to white noise: "
                   "camppca then denoises like MPPCA with the relative noise map.")

#: Why the loop fell back when the arithmetic, not the data, gave out.
NUMERICAL_REASON = ("This is a numerical failure of the fit. The noise model falls back "
                    "to white noise: camppca then denoises like MPPCA with the relative "
                    "noise map.")


def _build_C(rho, patch, lag):
    """Patch correlation from a correlation pattern, read by :func:`~camppca.patch.table_index`.

    If the table is indefinite for this patch, the eigenvalues are clipped at 1e-9 and
    the diagonal rescaled to 1, silently. The loop needs that every round. The public
    path, :meth:`NoiseModel.patch_correlation`, refuses such a table instead.
    """
    C = _table_C(rho, patch, lag)
    w, V = np.linalg.eigh(C)
    if w.min() < 1e-9:
        C = (V * np.maximum(w, 1e-9)) @ V.T
        scale = np.sqrt(np.diag(C))
        C = C / np.outer(scale, scale)
    return C


def _real_data(dwi, spatial_shape=None):
    """``dwi`` as the noise model is estimated on it: 4D, the real part of complex data,
    float32 with NaN and inf zeroed. Also returns the imaginary part as a view, or None.
    The estimator and the check both read the data this way."""
    from ..denoiser.gather import as_data, check_dwi, zero_nonfinite

    dwi = check_dwi(dwi, spatial_shape)
    imag = None
    if np.iscomplexobj(dwi):
        imag = dwi.imag
        dwi = np.ascontiguousarray(dwi.real, np.float32)
    return zero_nonfinite(as_data(dwi))[0], imag


@limit_blas
def estimate_noise_model(dwi, mask=None, *, bvals=None, patch=None, lowb=100,
                         algorithm="cordero-grande", noise_field=True, level_rounds=0,
                         workers=None):
    """Estimate the noise model from the data itself. No noise-only scan required.

    The relative noise map ``d`` is standard MPPCA's ``sigma_mppca`` of the patches on a
    stride-2 grid, each spread over its patch and averaged where patches overlap, see
    :func:`_footprint_average`. rho is the loop's (:mod:`camppca.noise_model.loop`),
    started from white noise: 1 at lag 0, 0 elsewhere.

    Each level round splits the whitened patches on the same grid, which gives ``s``, the
    level in d-normalized units. The new ``d`` is ``d * s`` averaged the same way, and the
    loop is refitted from the previous table. The rounds stop when ``d`` moves by less than
    :data:`LEVEL_TOL`. At the fixed point ``s`` is flat: every whitened patch carries the
    same noise level. They cost three to four times the estimate for little change on
    real data, hence the default of 0.

    With ``noise_field=False``, ``d`` is 1 in the mask and only the loop runs.

    Args:
        dwi: (X, Y, Z, N) diffusion data. Every volume enters, b0s included, because the
            denoiser whitens them all. Complex data are estimated on their real part,
            which holds the signal once the phase is stabilized (:mod:`camppca.phase`).
            The noise has the same correlation and level in both channels.
        mask: (X, Y, Z) bool. Estimated from the mean b0 when omitted.
        bvals: (N,) b-values, used only to find the b0s the default mask is drawn from.
        patch: the patch this model must be able to serve, which sets the lag rho
            covers. None picks dwidenoise's default, see
            :func:`~camppca.patch.default_patch`.
        lowb: b-value at or below which a volume counts as a b0.
        algorithm: the MP rule the denoiser will cut with.
        noise_field: whether the denoiser will divide rows by ``d``. Pass the settings
            you will denoise with: the loop fits to them.
        level_rounds: how many times ``d`` is re-estimated from the whitened patches.
            0, the default, keeps standard MPPCA's level.
        workers: CPU processes for the passes over every voxel and the loop's split, see
            :mod:`camppca.pool`. The default is one per physical core. 1 runs everything
            in this process.

    Returns:
        A :class:`NoiseModel`.
    """
    # Imported here: the denoiser imports this package, so a top-level import is circular.
    from .. import __version__  # the package sets it after importing this module
    from ..denoiser.gather import check_dwi, check_range
    from ..denoiser.mp_threshold import check_algorithm

    check_algorithm(algorithm)
    dwi, imag = _real_data(dwi)      # float32 once: every pass and map reuses it
    complex_input = imag is not None
    n_vols = dwi.shape[3]
    patch = resolve_patch(patch, dwi.shape)
    check_range(dwi, patch)
    lag = required_lag(patch)
    dzmax = int(patch[2]) - 1
    level_rounds = int(level_rounds)
    if level_rounds < 0:
        raise ValueError(f"level_rounds must be >= 0, got {level_rounds}")
    if n_vols < MIN_VOLUMES:
        warn(f"{n_vols} volumes: camppca works best with {MIN_VOLUMES} or more. With "
             "fewer, the noise level can come out too high.")

    if bvals is None:
        bvals = np.zeros(n_vols)
    else:
        bvals = np.asarray(bvals, float).ravel()
        if bvals.size != n_vols:
            raise ValueError(f"bvals has {bvals.size} entries, dwi has {n_vols} volumes")

    if mask is None:
        b0 = np.where(bvals <= lowb)[0]
        b0_mean = dwi[..., b0].mean(-1) if b0.size else dwi.mean(-1)
        # erode=0: with mask_only the stored mask is also the set of voxels the engine
        # WRITES, so eroding here would leave an un-denoised rim.
        mask = brain_mask(b0_mean, erode=0)
    mask = np.asarray(mask, bool)
    check_dwi(dwi, mask.shape)

    rho = _white_rho((2 * lag + 1, 2 * lag + 1, dzmax + 1))
    if noise_field:
        d = _footprint_average(*_level_pass(dwi, mask, patch, algorithm, workers), mask,
                               patch)
    else:
        d = mask.astype(np.float32)
    if complex_input:
        _check_phase_stabilized(imag, dwi, mask, d if noise_field else None, patch,
                                algorithm, workers)
    sample = _good_level(d, mask)          # the same patches in every level round
    rho, info = _logged_fit(dwi, mask, d, rho, patch, lag, algorithm=algorithm,
                            noise_field=noise_field, workers=workers, sample=sample)

    rounds_run, change, spread = 0, None, None
    # A fallback leaves no correlation to refine the level with: no rounds then.
    n_rounds = level_rounds if noise_field and not info["fallback"] else 0
    for rounds_run in range(1, n_rounds + 1):  # noqa: B007
        d_eff = _effective_d(d, mask)
        s, centers = _level_pass(dwi, mask, patch, algorithm, workers,
                                 C=_build_C(rho, patch, lag), d_field=d_eff)
        q25, q50, q75 = np.percentile(s[centers], (25, 50, 75))
        spread = float((q75 - q25) / q50)
        new = _footprint_average(d_eff * s, centers, mask, patch)
        both = (new > 0) & (d > 0)         # voxels below the floor come out zero
        change = float(np.median(np.abs((new[both] / np.median(new[both]))
                                        / (d[both] / np.median(d[both])) - 1)))
        log.info("level round %d: the relative noise map moved by a median %.2f%%",
                 rounds_run, 100 * change)
        if change < LEVEL_TOL:
            break
        d = new
        rho, info = _logged_fit(dwi, mask, d, rho, patch, lag, algorithm=algorithm,
                                noise_field=noise_field, workers=workers, sample=sample)
        if info["fallback"]:
            break

    prov = dict(version=__version__,
                patch=tuple(int(v) for v in np.asarray(patch, int)), lag=lag, dzmax=dzmax,
                lowb=float(lowb), level_stride=LEVEL_STRIDE, n_mask=int(mask.sum()),
                n_vols=int(n_vols), bvals_sha=_bvals_hash(bvals),
                loop_rounds=info["rounds"], loop_converged=info["converged"],
                sphericity=info["sphericity"], n_centers=info["n_centers"],
                loop_stride=info["stride"], loop_steps=info["steps"],
                loop_algorithm=algorithm, loop_noise_field=bool(noise_field),
                level_rounds=level_rounds, level_rounds_run=rounds_run,
                fallback=info["fallback"])
    if complex_input:
        prov.update(data="complex, estimated on the real part")
    if rounds_run:
        prov.update(level_converged=change < LEVEL_TOL, level_change=change,
                    s_spread=spread)

    model = NoiseModel(rho=rho, d=d, mask=mask, lag=lag, provenance=prov,
                       fitted_to=dict(patch=prov["patch"], algorithm=algorithm,
                                      noise_field=bool(noise_field)))
    # Check here, not at denoising after the estimate is spent. The loop checks every
    # round, so this is the backstop: the same fallback, never a crash.
    try:
        model.patch_correlation(patch)
    except ValueError as err:
        reason = f"the loop's correlation pattern is not usable: {err} {FALLBACK_REASON}"
        warn(reason)
        model.rho = _white_rho(rho.shape)
        model.provenance.update(fallback=reason, sphericity=0.0)
    return model


def _logged_fit(dwi, mask, d, rho, patch, lag, **kwargs):
    """:func:`~camppca.noise_model.loop.fit_rho`, with a status line before and after.

    The rounds are logged at DEBUG after the fit, here and not in the loop, since the check
    refits it up to 23 times.
    """
    from .loop import fit_rho  # circular at the top level

    shape = "x".join(str(int(v)) for v in patch)
    log.info("correlation loop on patches of %s", shape)
    start = time.perf_counter()
    rho, info = fit_rho(dwi, mask, d, rho, patch, lag, **kwargs)
    for n, (step, change) in enumerate(zip(info["steps"], info["changes"], strict=True), 1):
        log.debug("round %d: step %g, largest change %.1e", n, step, change)
    if info["fallback"]:
        outcome = f"fell back to white noise in round {info['rounds']}"
    elif info["converged"]:
        outcome = f"converged after {info['rounds']} rounds"
    else:
        outcome = f"stopped after {info['rounds']} rounds"
    log.info("correlation loop %s in %s, %d patches", outcome,
             elapsed(time.perf_counter() - start), info["n_centers"])
    return rho, info


#: The imaginary part's rms over the real part's noise level above which complex data look
#: phase unstabilized. Stabilized data stay well below it, the imaginary part holding a
#: little structure. Unstabilized, it carries the signal.
PHASE_RATIO = 3.0


def _check_phase_stabilized(imag, real, mask, level, patch, algorithm, workers):
    """Warn when the imaginary part carries far more than noise.

    The estimator works on the real part, which holds the signal only once the phase is
    stabilized. Otherwise the real part is a random projection of the signal, and the
    model fits that. 10000 mask voxels, evenly spaced in C order, are enough and need no
    copy of the volume. Spaced, not drawn, so the estimate takes no seed: the loop's
    centers lie on a grid too.

    ``level`` is standard MPPCA's level of the real part, or None without a noise field,
    and then it is computed at the sampled voxels only. NaN and inf in the imaginary part are left out: the real part is
    cleaned before estimating, the imaginary part is not, and one NaN made the ratio NaN,
    which never exceeds the threshold.
    """
    from ..denoiser.camppca import whitened_level  # circular at the top level

    voxels = np.flatnonzero(mask)
    pick = voxels[np.linspace(0, voxels.size - 1, min(10000, voxels.size)).astype(int)]
    where = np.unravel_index(pick, mask.shape)
    if level is None:
        picked = np.zeros(mask.shape, bool)
        picked[where] = True
        level = whitened_level(real, picked, None, patch, algorithm, workers=workers)
        mask = picked
    values = imag[where]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ratio = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))
                  / np.median(level[mask]))
    if ratio > PHASE_RATIO:
        warn(f"the imaginary part is {ratio:.1f} times the real part's noise level "
             "inside the mask. The noise model is estimated on the real part, "
             "which holds the signal only once the phase is stabilized: run "
             "`camppca phase` first.")


def _level_pass(dwi, mask, patch, algorithm, workers, C=None, d_field=None):
    """The MP level of the patch at every level center, see :func:`_level_centers`.

    ``C`` and ``d_field`` as :func:`~camppca.denoiser.camppca.whitened_level` takes them:
    None for standard MPPCA, given for the level rounds. Returns ``(level, centers)``.
    """
    from ..denoiser.camppca import whitened_level  # circular at the top level

    centers = _level_centers(mask, patch)
    label = "relative noise map" if C is None else "relative noise map, whitened"
    level = whitened_level(dwi, centers, C, patch, algorithm, d_field=d_field,
                           workers=workers, progress_label=label)
    return level, centers


def _level_centers(mask, patch):
    """Mask voxels with every coordinate a multiple of :data:`LEVEL_STRIDE`, plus any mask
    voxel that none of their patches covers."""
    x, y, z = np.indices(mask.shape, sparse=True)
    on = (x % LEVEL_STRIDE == 0) & (y % LEVEL_STRIDE == 0) & (z % LEVEL_STRIDE == 0)
    centers = mask & on
    return centers | (mask & ~_covered(centers, patch))


def _covered(centers, patch):
    """The voxels at least one of the centers' patches reaches."""
    size = tuple(int(p) for p in patch)
    # A running sum: test against half of one center's share, not against zero.
    return uniform_filter(centers.astype(np.float64), size=size,
                          mode="constant") > 0.5 / np.prod(size)


def _footprint_average(level, centers, mask, patch):
    """Each center's level spread over its patch, averaged where patches overlap.

    Every center whose level passes the floor of ``_effective_d`` adds its variance to
    each voxel of its patch. Each mask voxel's level is the root of the mean over the
    patches that cover it. This is NORDIC's construction for its noise map (Moeller et
    al., NeuroImage 2021), here with 3D patches on a stride-2 grid. A patch's MP level is
    the noise of all its voxels, so it belongs to all of them, not to the center alone.

    The mean is over squares because the square is the unbiased quantity, being a second
    moment: averaging levels biases low. Averaging over the patch also smooths the grain
    an MP level map carries where neighbouring patches cut at different ranks. Unpooled,
    that grain made the loop's ``C`` near singular on strongly correlated data.

    Centers below the floor, such as patches of zero-filled slices, add nothing, so they
    do not pull their neighbours down. They stay zero themselves, as do mask voxels no
    good center reaches, for ``_effective_d`` to fill.

    Returns:
        (X, Y, Z) float32 level, zero outside ``mask``.
    """
    good = _good_level(level, centers)
    size = tuple(int(p) for p in patch)
    kw = dict(size=size, mode="constant", cval=0.0)
    num = uniform_filter(np.where(good, level, 0.0).astype(np.float64) ** 2, **kw)
    den = uniform_filter(good.astype(np.float64), **kw)
    out = np.zeros(mask.shape, np.float32)
    ok = mask & (den > 0.5 / np.prod(size))
    # The running sums round slightly below zero next to levels near zero: clip.
    out[ok] = np.sqrt(np.maximum(num[ok], 0.0) / den[ok])
    out[centers & ~good] = 0.0
    return out


def _bvals_hash(bvals):
    return hashlib.sha256(np.ascontiguousarray(bvals, np.float64).tobytes()).hexdigest()[:16]
