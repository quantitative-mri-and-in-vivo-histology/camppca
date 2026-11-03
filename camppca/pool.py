"""CPU worker processes for the passes that touch every voxel, through joblib.

Each patch is split on its own, so the work divides over voxels. Threads do not help:
scipy's eigendecomposition holds the GIL, and OpenBLAS called from several threads at
once queues them (5.9x at 8 threads, 2.7x at 16, through ctypes). Separate processes,
each with OpenBLAS limited to one thread, scale about 12x on 16 cores.

The processes are joblib's reusable loky workers: started on first use, kept warm
between calls, and able to run functions defined in a user's script. A worker that dies
raises an error instead of hanging, and the next call starts fresh ones. They are not
pinned to cores; on the EDDEN 0.9 mm image that cost nothing against the former pinned
pool (151.5 s against 154.1 s for the MP-PCA pass).

Large arrays, the volume and the level field, reach the workers as joblib memory maps:
written once per pass to a file the workers map read-only. The file goes to the temp
directory, so ``TMPDIR`` or ``JOBLIB_TEMP_FOLDER`` moves it, to local scratch on a
cluster say. joblib's own default would be ``/dev/shm``, which pins the whole volume in
RAM: 10.8 GB at the peak against 5.7 GB on disk for that pass, at the same speed.

Each task returns only its own voxels' results, so no worker holds a full-slab output.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config

__all__ = ["default_workers", "resolve_workers", "worker_pool", "voxel_chunks",
           "run_voxels"]


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
    """Physical cores this process may run on. 32 workers on 16 cores were 4% faster."""
    return _physical_cores()


def resolve_workers(workers) -> int:
    """``None`` means :func:`default_workers`; otherwise a positive count."""
    if workers is None:
        return default_workers()
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return workers


@contextmanager
def worker_pool(workers: int, ordered: bool = True):
    """A joblib ``Parallel`` over ``workers`` processes, kept open for several calls.

    Within one ``with`` block an array passed to several calls is memory-mapped once.
    ``ordered`` False yields results as they finish.
    """
    temp = os.environ.get("JOBLIB_TEMP_FOLDER") or tempfile.gettempdir()
    with parallel_config(backend="loky", inner_max_num_threads=1), \
            Parallel(n_jobs=workers, temp_folder=temp, max_nbytes="1M", mmap_mode="r",
                     return_as="list" if ordered else "generator_unordered") as parallel:
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
        arrays: the large inputs, the volume and the level field say. In the pool they
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
        for i, chunk in enumerate(chunks):
            write(slice(starts[i], starts[i + 1]), fn(*arrays, chunk))
        return
    with worker_pool(workers, ordered=False) as parallel:
        for i, outputs in parallel(delayed(_chunk_task)(fn, i, chunk, arrays)
                                   for i, chunk in enumerate(chunks)):
            write(slice(starts[i], starts[i + 1]), outputs)
