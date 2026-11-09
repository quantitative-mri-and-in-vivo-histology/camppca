"""CPU worker processes for the passes that touch every voxel, through joblib.

Each patch is split on its own, so the work divides over voxels. Threads do not help:
scipy's eigendecomposition holds the GIL, and OpenBLAS called from several threads at
once queues them. Separate processes, each with OpenBLAS limited to one thread, scale
across the cores.

The processes are joblib's reusable loky workers: started on first use, kept warm
between calls, and able to run functions defined in a user's script. A worker that dies
raises an error instead of hanging, and the next call starts fresh ones. They are not
pinned to cores, which cost nothing.

Large arrays, the volume and the relative noise map, reach the workers as joblib memory
maps: written once per pass to a file the workers map read-only. The file goes to the
temp directory, so ``TMPDIR`` or ``JOBLIB_TEMP_FOLDER`` moves it, to local scratch on a
cluster say. joblib's own default, ``/dev/shm``, would pin the whole volume in RAM for no
gain in speed.

Each task returns only its own voxels' results, so no worker holds a full-slab output.

BLAS THREADS. The workers run OpenBLAS with one thread. This process sets its own only
inside camppca's calls, through :func:`blas_threads`, and restores it after, so a script
that imports camppca keeps its settings. OpenBLAS's default is one thread per hardware
thread, for numpy's and scipy's copies alike.

- Per-patch work run in this process, every pass with ``workers=1``, gets one thread,
  as in the workers. Small matrices on many threads were ten times slower.
- The rest, mostly the loop's fit, gets :func:`main_threads`, at most
  :data:`MAIN_THREADS`, where the fit was fastest.
- Neither raises the count above what the caller set: an ``OPENBLAS_NUM_THREADS`` or an
  outer limit stays in force.

The fitted table moves with the thread count, by up to 1e-4. So the count follows
neither ``workers`` nor the pool: the loop stays the same for any number of workers.
The measurements are in CLAUDE.md, under Gotchas.
"""

from __future__ import annotations

import functools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_info, threadpool_limits

__all__ = ["default_workers", "resolve_workers", "worker_pool", "voxel_chunks",
           "run_voxels", "run_boxes", "blas_threads", "main_threads", "limit_blas"]

#: BLAS threads for camppca's own work in this process, at most.
MAIN_THREADS = 8


def _physical_cores():
    """Physical cores among the CPUs this process may run on."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or 1))
    cores = set()
    for cpu in allowed:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(((topo / "physical_package_id").read_text(),
                       (topo / "core_id").read_text()))
        except OSError:
            cores.add(cpu)
    return len(cores)


def default_workers() -> int:
    """``CAMPPCA_WORKERS`` if set, else the physical cores this process may run on, at
    most its CPU quota. One per hardware thread gained only a few percent.

    joblib's count reads the container's CPU quota, so a job limited to 4 CPUs on a
    32-core host starts 4 workers, not 16 throttled ones.
    """
    if os.environ.get("CAMPPCA_WORKERS"):
        return max(1, int(os.environ["CAMPPCA_WORKERS"]))
    return max(1, min(_physical_cores(), joblib.cpu_count()))


def _caller_threads() -> int:
    """The fewest BLAS threads any loaded BLAS library is set to now."""
    counts = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
    return min(counts) if counts else 1


def main_threads() -> int:
    """BLAS threads for this process's own work: :data:`MAIN_THREADS`, at most the CPUs
    this process may use and at most what the caller has set."""
    return max(1, min(MAIN_THREADS, joblib.cpu_count(), _caller_threads()))


@contextmanager
def blas_threads(n: int):
    """Hold every loaded BLAS library at ``n`` threads, at most the caller's count,
    restored on exit."""
    with threadpool_limits(limits=max(1, min(int(n), _caller_threads())), user_api="blas"):
        yield


def limit_blas(fn):
    """Run ``fn`` with :func:`main_threads` BLAS threads, the caller's setting restored
    after. For the public entry points; nested calls keep the outer limit."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with blas_threads(main_threads()):
            return fn(*args, **kwargs)
    return wrapper


def resolve_workers(workers) -> int:
    """``None`` means :func:`default_workers`; otherwise a positive count."""
    if workers is None:
        return default_workers()
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return workers


@contextmanager
def worker_pool(workers: int, return_as: str = "list"):
    """A joblib ``Parallel`` over ``workers`` processes, kept open for several calls.

    Within one ``with`` block an array passed to several calls is memory-mapped once.
    ``return_as`` is joblib's: "list", "generator" (in order) or "generator_unordered".
    A generator dispatches at most twice ``workers`` tasks ahead of the one consumed.
    """
    temp = os.environ.get("JOBLIB_TEMP_FOLDER") or tempfile.gettempdir()
    with parallel_config(backend="loky", inner_max_num_threads=1), \
            Parallel(n_jobs=workers, temp_folder=temp, max_nbytes="1M", mmap_mode="r",
                     return_as=return_as) as parallel:
        yield parallel


def voxel_chunks(voxels, n_chunks: int, multiple: int) -> list[np.ndarray]:
    """``voxels`` cut into about ``n_chunks`` runs, each but the last a whole number of
    ``multiple``. With ``multiple`` the engine's tile size, every tile is full, so the
    chunks see the same tiles as one pass over all voxels."""
    voxels = np.asarray(voxels)
    if voxels.size == 0:
        return []
    size = max(1, -(-voxels.size // (n_chunks * multiple))) * multiple
    return [voxels[start:start + size] for start in range(0, voxels.size, size)]


def _chunk_task(fn, i, voxels, arrays):
    return i, fn(*arrays, voxels)


def run_voxels(fn, arrays, voxels, write, workers: int, tile: int):
    """Run a per-voxel pass in chunks, over the worker pool when ``workers`` > 1.

    Args:
        fn: picklable ``fn(*arrays, voxels)`` returning a tuple of arrays whose first
            axis runs over the voxels it was given.
        arrays: the large inputs, the volume and the relative noise map say. In the pool they
            are memory-mapped once for all chunks.
        voxels: flat voxel indices, C order.
        write: ``write(sl, outputs)`` stores one chunk's outputs; ``sl`` slices
            ``voxels`` to that chunk. Called in this process, in any chunk order.
        workers: processes; 1 runs every chunk here.
        tile: the engine's tile size; chunks hold whole tiles, so the result is the
            single pass, bit for bit.
    """
    chunks = voxel_chunks(voxels, 8 * workers, int(tile))
    starts = np.cumsum([0] + [c.size for c in chunks])
    if workers == 1:
        with blas_threads(1):              # per-patch work, as in the workers
            for i, chunk in enumerate(chunks):
                write(slice(starts[i], starts[i + 1]), fn(*arrays, chunk))
        return
    with worker_pool(workers, return_as="generator_unordered") as parallel:
        for i, outputs in parallel(delayed(_chunk_task)(fn, i, chunk, arrays)
                                   for i, chunk in enumerate(chunks)):
            write(slice(starts[i], starts[i + 1]), outputs)


def run_boxes(fn, arrays, chunks, add, workers: int):
    """Run a per-chunk pass whose results are summed, in chunk order.

    Like :func:`run_voxels`, but ``add(outputs)`` receives the chunks' outputs strictly
    in the order of ``chunks``, so sums come out the same for any number of workers.
    """
    if workers == 1:
        with blas_threads(1):              # per-patch work, as in the workers
            for chunk in chunks:
                add(fn(*arrays, chunk))
        return
    with worker_pool(workers, return_as="generator") as parallel:
        for _, outputs in parallel(delayed(_chunk_task)(fn, i, chunk, arrays)
                                   for i, chunk in enumerate(chunks)):
            add(outputs)
