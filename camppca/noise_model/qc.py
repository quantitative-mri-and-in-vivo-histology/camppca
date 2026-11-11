"""Does one correlation pattern hold across the brain and across the volumes?

The noise model assumes one ``rho`` for every voxel and every volume. The check refits the
loop on parts of the data and asks what the difference would cost the denoiser.

- **Regions.** The patch centers split into a grid of regions, 2 x 2 x 2 by default,
  at the quantiles of the centers along each axis. Each region is refitted and compared
  with the model. A region is judged against random subsets of the centers of the same
  size, refitted the same way. Their spread is the noise of a fit on that many patches.
  A region is flagged beyond twice their largest change and beyond 1%. The centers are
  the loop's tiling, never narrowed to overlap: overlapping patches share noise, and on
  synthetic stationary noise every region then fell outside the random subsets.
- **Volumes.** Random 50/50 splits of the volumes, every center used. The two halves of
  a split are compared with each other. Never with the model: with half the volumes the
  split leaks more signal into the bulk, and every half differs from the model the same
  way. The splits should agree with each other.

The cost of a difference is measured on synthetic noise. Pure-noise patches are drawn with
one patch correlation ``C_true``, at the data's volume count, and whitened once with
``C_true`` and once with the other ``C``. The change is the median ratio of the two MP
noise levels, minus one. Positive means the other ``C`` would report a higher level.
The draws are the same for every comparison.
"""

from dataclasses import dataclass, field

import numpy as np
from joblib import delayed

from ..denoiser.camppca import _whiten_split, sqrtm_psd
from ..denoiser.gather import check_range
from ..pool import limit_blas, resolve_workers, worker_pool
from ..warn import warn
from . import MIN_VOLUMES, _build_C, _good_level, _real_data, _white_rho
from .loop import _sample_centers, fit_rho

__all__ = ["NoiseModelCheck", "check_noise_model"]

NOISE_BATCH = 64        # synthetic patches per split call
REFERENCE_FACTOR = 2    # a region stands out beyond this many times the random spread
MIN_FLAGGED = 0.01      # and beyond this change in the level, 1%


@dataclass
class NoiseModelCheck:
    """What :func:`check_noise_model` found. All changes are fractions, 0.01 is 1%."""

    regions: list
    """One dict per region: ``region`` (its name, e.g. "x0 y1 z0"), ``n_centers``,
    ``change`` (noise drawn with the region's ``C``, whitened with the model's) and
    ``fallback``."""

    reference: list
    """One dict per random subset of the centers, with the same keys."""

    volume_splits: list
    """One dict per split: ``n_vols`` of each half, ``change`` as a pair (noise from half
    A whitened with B's ``C``, and the reverse) and ``fallback``. Empty when the volumes
    were too few to split."""

    provenance: dict = field(default_factory=dict)
    """The settings, and why the volume splits were skipped if they were."""

    def reference_spread(self):
        """The largest change among the random subsets: a region's yardstick."""
        return max((abs(r["change"]) for r in self.reference), default=float("nan"))

    def flagged(self):
        """The regions that changed the level by more than 1% and by more than twice the
        largest random subset.

        Twice, not once: of 8 regions on stationary noise, one beats the largest of 5
        random subsets by chance, as a region beats 5 draws 1 time in 6. And 1%: on real
        data regions can differ beyond the random subsets, by too little to matter.
        """
        bound = max(REFERENCE_FACTOR * self.reference_spread(), MIN_FLAGGED)
        return [r["region"] for r in self.regions if abs(r["change"]) > bound]

    def report(self):
        """The findings as lines of text."""
        lines = [f"regions, against the model; random subsets of {self.reference[0]['n_centers']}"
                 f" centers changed by up to {self.reference_spread():.2%}; flagged beyond "
                 f"twice that and {MIN_FLAGGED:.0%}"
                 if self.reference else "regions, against the model"]
        flagged = self.flagged()
        for r in self.regions:
            flag = "  flagged" if r["region"] in flagged else ""
            fell = "  fell back to white noise" if r["fallback"] else ""
            lines.append(f"  {r['region']}  {r['n_centers']:>5} centers  "
                         f"{r['change']:+.2%}{flag}{fell}")
        if self.volume_splits:
            lines.append("random volume halves, against each other")
            for k, s in enumerate(self.volume_splits):
                fell = "  fell back to white noise" if s["fallback"] else ""
                lines.append(f"  split {k}  {s['n_vols'][0]} / {s['n_vols'][1]} volumes  "
                             f"{s['change'][0]:+.2%}, {s['change'][1]:+.2%}{fell}")
        else:
            lines.append(f"volume halves skipped: {self.provenance['volume_splits_skipped']}")
        return lines


@limit_blas
def check_noise_model(dwi, model, *, grid=(2, 2, 2), random_splits=5, n_noise=2000,
                      seed=0, workers=None):
    """Refit the noise model on regions and on volume halves, and price the differences.

    Every refit costs about one loop of the estimate. The default runs 23 of them.
    Volume halves need :data:`~camppca.noise_model.MIN_VOLUMES` volumes in all, and are
    skipped with a warning below that.

    Args:
        dwi: the (X, Y, Z, N) data the model was estimated on. Complex data are checked
            on their real part, as they were estimated.
        model: a :class:`~camppca.noise_model.NoiseModel` from
            :func:`~camppca.estimate_noise_model`. Its ``fitted_to`` gives the patch,
            the MP rule and whether rows were divided by ``d``. The refits use the same.
        grid: regions per axis. The default 2 x 2 x 2 gives 8.
        random_splits: how many random subsets of the centers the regions are judged
            against, and how many random volume halves are compared.
        n_noise: synthetic noise patches per comparison.
        seed: for the random subsets, the volume halves and the noise.
        workers: CPU processes for each refit's split and for the synthetic noise, see
            :mod:`camppca.pool`.

    Returns:
        A :class:`NoiseModelCheck`.

    Raises:
        ValueError: if the model records no fitted patch, or ``dwi`` is not on its grid.
    """
    fitted = model.fitted_to
    if "patch" not in fitted:
        raise ValueError("the noise model records no patch it was fitted to, so its fit "
                         "cannot be repeated. Check a model from estimate_noise_model.")
    grid = tuple(int(g) for g in np.broadcast_to(grid, 3))
    if min(grid) < 1:
        raise ValueError(f"grid must have at least 1 region per axis, got {grid}")
    random_splits = int(random_splits)
    if random_splits < 1:
        raise ValueError(f"random_splits must be >= 1, got {random_splits}")
    dwi = _real_data(dwi, model.mask.shape)[0]
    patch = np.asarray(fitted["patch"], int)
    check_range(dwi, patch)
    algorithm = fitted.get("algorithm", "cordero-grande")
    noise_field = fitted.get("noise_field", True)
    n_vols = dwi.shape[3]
    rng = np.random.default_rng(seed)
    noise_seed = int(rng.integers(2 ** 32))

    # The tiling, never overlapping: see _sample_centers.
    centers, _ = _sample_centers(_good_level(model.d, model.mask), patch, min_centers=0)
    n_centers = centers[0].size
    C_model = model.patch_correlation(patch)

    def refit(pick, data=dwi):
        rho, info = fit_rho(data, model.mask, model.d, _white_rho(model.rho.shape), patch,
                            model.lag, algorithm=algorithm, noise_field=noise_field,
                            workers=workers, centers=tuple(c[pick] for c in centers))
        return _build_C(rho, patch, model.lag), info["fallback"]

    # Regions at the quantiles of the centers, so each axis splits into equal counts.
    bins = []
    for axis, g in enumerate(grid):
        edges = np.quantile(centers[axis], np.arange(1, g) / g)
        bins.append(np.searchsorted(edges, centers[axis], side="right"))
    region_of = np.ravel_multi_index(bins, grid)
    regions, pairs = [], []        # pairs: (C_true, C_used), priced after the fits
    for index in np.ndindex(*grid):
        pick = region_of == np.ravel_multi_index(index, grid)
        if not pick.any():
            continue
        C, fallback = refit(pick)
        pairs.append((C, C_model))
        regions.append(dict(region=" ".join(f"{a}{i}" for a, i in zip("xyz", index,
                                                                         strict=True)),
                            n_centers=int(pick.sum()), change=None,
                            fallback=fallback))

    size = int(np.median([r["n_centers"] for r in regions]))
    reference = []
    for _ in range(random_splits):
        pick = np.zeros(n_centers, bool)
        pick[rng.choice(n_centers, size, replace=False)] = True
        C, fallback = refit(pick)
        pairs.append((C, C_model))
        reference.append(dict(region="random", n_centers=size, change=None,
                              fallback=fallback))

    volume_splits, skipped = [], None
    if n_vols < MIN_VOLUMES:
        skipped = (f"{n_vols} volumes, fewer than {MIN_VOLUMES}, where the "
                   f"Marchenko-Pastur fit degenerates")
        warn(f"volume halves skipped: {skipped}")
    else:
        every = np.ones(n_centers, bool)
        for _ in range(random_splits):
            order = rng.permutation(n_vols)
            halves = [np.sort(order[:n_vols // 2]), np.sort(order[n_vols // 2:])]
            fits = [refit(every, np.ascontiguousarray(dwi[..., h])) for h in halves]
            (C_a, fell_a), (C_b, fell_b) = fits
            pairs += [(C_a, C_b), (C_b, C_a)]
            volume_splits.append(dict(n_vols=(halves[0].size, halves[1].size),
                                      change=None,
                                      fallback=fell_a or fell_b))

    # The synthetic noise in one pool after the fits: serial it took 80 s of 264 at 0.9 mm.
    changes = iter(_sigma_changes(pairs, n_vols, algorithm, n_noise, noise_seed, workers))
    for r in regions + reference:
        r["change"] = next(changes)
    for s in volume_splits:
        s["change"] = (next(changes), next(changes))

    prov = dict(grid=grid, random_splits=random_splits, n_noise=int(n_noise), seed=seed,
                patch=tuple(int(v) for v in patch), algorithm=algorithm,
                noise_field=bool(noise_field), n_centers=int(n_centers),
                volume_splits_skipped=skipped)
    return NoiseModelCheck(regions=regions, reference=reference,
                           volume_splits=volume_splits, provenance=prov)


def _sigma_changes(pairs, n_vols, algorithm, n_noise, seed, workers):
    """:func:`_sigma_change` for each ``(C_true, C_used)`` pair, one task each."""
    workers = resolve_workers(workers)
    if workers == 1:
        return [_sigma_change(*pair, n_vols, algorithm, n_noise, seed) for pair in pairs]
    with worker_pool(workers) as parallel:
        return parallel(delayed(_sigma_change)(*pair, n_vols, algorithm, n_noise, seed)
                        for pair in pairs)


def _sigma_change(C_true, C_used, n_vols, algorithm, n_noise, seed):
    """Median MP level of noise drawn with ``C_true`` and whitened with ``C_used``, over
    the level whitened with ``C_true`` itself, minus one. One seed gives the same draws
    for every pair."""
    rng = np.random.default_rng(seed)
    n_vox = C_true.shape[0]
    n_small, n_large = min(n_vox, n_vols), max(n_vox, n_vols)
    root = sqrtm_psd(C_true, 0.5).astype(np.float32)
    whiteners = [sqrtm_psd(C, -0.5).astype(np.float32) for C in (C_true, C_used)]
    ratios = []
    for start in range(0, n_noise, NOISE_BATCH):
        draws = rng.standard_normal((min(NOISE_BATCH, n_noise - start), n_vox, n_vols),
                                    dtype=np.float32)
        noise = np.matmul(root, draws)
        own, other = (_whiten_split(noise, W, n_small, n_large, algorithm,
                                    values_only=True)[4] for W in whiteners)
        ratios.append(other / own)
    return float(np.median(np.concatenate(ratios)) - 1)
