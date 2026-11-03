"""A pool of pinned CPU processes for the passes that touch every voxel.

Each patch is split on its own, so the work divides over voxels. Threads do not help:
calls into OpenBLAS from several threads of one process queue on a shared lock. 16
threads calling its eigendecomposition kept 14.6 cores busy for a quarter of the useful
work, the rest in the kernel. Separate processes each load their own OpenBLAS, limited
to one thread, and scale about 12x on 16 cores. On a 16-core host a full MP-PCA pass
over 297 volumes ran faster this way than on an RTX 4060 Ti.

Every worker is pinned to one core, so no two share a core while another idles. This was
essential while the workers ran JAX, whose CPU thread pool spins when idle.

Workers are spawned, and they do NOT re-import the caller's main module, which spawn
does by default. They only run camppca's own functions, so a script without the
``if __name__ == "__main__":`` guard, or one read from stdin, works unchanged.

A worker that dies, killed for memory say, raises ``BrokenProcessPool`` rather than
hanging, and the next call starts a fresh pool.

The volume reaches the workers as a memory-mapped ``.npy`` file in the temp directory,
not through pickling. Each worker reads its slab straight from the mapping, without a
copy of its own. On the full EDDEN image the file in ``/dev/shm`` instead was no faster
and pinned its 2.2 GB in RAM: the system used 14.6 GB at the peak against 12.1 GB with
the file on disk, whose cached pages the kernel can drop.
"""

from __future__ import annotations

import atexit
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from functools import partial
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path

import numpy as np

__all__ = ["default_workers", "resolve_workers", "get_pool", "shared_volume",
           "voxel_chunks", "run_slabs"]

# Set while the workers start, so they load numpy and OpenBLAS with it.
_WORKER_ENV = dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

_POOL = None
_POOL_WORKERS = 0


def _cpus():
    """This process's CPUs, one per physical core first, then their hyperthread siblings."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(os.cpu_count() or 1))
    first, siblings, seen = [], [], set()
    for cpu in allowed:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topo / "physical_package_id").read_text(), (topo / "core_id").read_text())
        except OSError:
            key = cpu
        (siblings if key in seen else first).append(cpu)
        seen.add(key)
    return first, siblings


def default_workers() -> int:
    """Physical cores this process may run on. 32 workers on 16 cores were no faster."""
    return len(_cpus()[0])


def resolve_workers(workers) -> int:
    """``None`` means :func:`default_workers`; otherwise a positive count."""
    if workers is None:
        return default_workers()
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return workers


def _pin(queue):
    cpu = queue.get()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})


@contextmanager
def _launch_settings():
    """While one worker starts: its environment, and no main module to re-import.

    Spawn records ``__main__``'s ``__spec__`` and ``__file__`` and makes the child run
    that module first. Hidden here, so the child starts clean.
    """
    saved_env = {k: os.environ.get(k) for k in _WORKER_ENV}
    main = sys.modules.get("__main__")
    saved_main = {k: main.__dict__[k] for k in ("__spec__", "__file__")
                  if main is not None and k in main.__dict__}
    os.environ.update(_WORKER_ENV)
    if main is not None:
        main.__dict__.pop("__file__", None)
        main.__spec__ = None
    try:
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v
        if main is not None:
            main.__dict__.update(saved_main)


class _WorkerProcess(SpawnProcess):
    @staticmethod
    def _Popen(process_obj):
        with _launch_settings():
            return SpawnProcess._Popen(process_obj)


class _WorkerContext(SpawnContext):
    Process = _WorkerProcess


def get_pool(workers: int):
    """The shared executor of ``workers`` pinned processes, started on first use.

    Kept across calls, so each worker's compiled functions stay warm from the noise
    model estimate to the denoise. Started again when ``workers`` changes or a worker
    died.
    """
    global _POOL, _POOL_WORKERS
    if _POOL is not None and _POOL_WORKERS == workers and not _POOL._broken:
        return _POOL
    _shutdown()
    ctx = _WorkerContext()
    first, siblings = _cpus()
    order = first + siblings
    queue = ctx.SimpleQueue()
    for k in range(workers):
        queue.put(order[k % len(order)])
    _POOL = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_pin,
                                initargs=(queue,))
    _POOL_WORKERS = workers
    return _POOL


@atexit.register
def _shutdown():
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL, _POOL_WORKERS = None, 0


@contextmanager
def shared_volume(dwi, d_field=None, pad=(0, 0, 0)):
    """Write ``dwi`` to a temporary ``.npy`` file the workers can memory-map.

    The level field, if given, rides along as one extra column, as the engine carries
    it. ``pad`` wrap-pads each spatial axis, as the engine pads, so a slab cut from the
    file with a half-patch border holds every patch of its inner voxels. Written one x
    slice at a time, so no padded copy is held in memory.

    Yields:
        The file's path. The file is deleted on exit.
    """
    shape = dwi.shape[:3]
    n_cols = dwi.shape[3] + (d_field is not None)
    yz = np.ix_(*((np.arange(-p, s + p) % s) for s, p in zip(shape[1:], pad[1:], strict=True)))
    fd, name = tempfile.mkstemp(suffix=".npy", prefix="camppca_")
    os.close(fd)
    try:
        out = np.lib.format.open_memmap(
            name, mode="w+", dtype=np.float32,
            shape=tuple(s + 2 * p for s, p in zip(shape, pad, strict=True)) + (n_cols,))
        for i in range(out.shape[0]):
            x = (i - pad[0]) % shape[0]
            out[i, :, :, :dwi.shape[3]] = dwi[x][yz]
            if d_field is not None:
                out[i, :, :, -1] = np.asarray(d_field, np.float32)[x][yz]
        out.flush()
        del out
        yield name
    finally:
        os.unlink(name)


def voxel_chunks(mask, n_chunks: int, multiple: int) -> list[np.ndarray]:
    """The mask voxels in C order, cut into about ``n_chunks`` runs.

    Each run but the last holds a whole number of ``multiple`` voxels. With
    ``multiple`` the engine's tile size, every tile a worker computes is full.

    Returns:
        Flat voxel indices, one array per chunk, covering every mask voxel once.
    """
    idx = np.flatnonzero(np.asarray(mask, bool))
    if idx.size == 0:
        return []
    size = max(1, -(-idx.size // (n_chunks * multiple))) * multiple
    return [idx[start:start + size] for start in range(0, idx.size, size)]


def _slab_task(args):
    fn, path, lo, hi, pad, n_vols, has_d, inner = args
    vol = np.load(path, mmap_mode="r")
    block = vol[lo:hi + 2 * pad[0]]                # a view of the mapping, not a copy
    if has_d:
        outputs = fn(block[..., :n_vols], inner, d_field=block[..., n_vols])
    else:
        outputs = fn(block, inner)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    return tuple(np.asarray(o[inner]) for o in outputs)


def run_slabs(fn, dwi, mask, patch, outputs, workers: int, tile: int, d_field=None):
    """Run a per-voxel pass over the worker pool in equal voxel chunks.

    Each chunk is a run of mask voxels in C order. Its worker gets the x slab those
    voxels span, with a half-patch border, so every voxel sees its whole patch and the
    result is the serial one. About 8 chunks per worker keep the workers evenly busy.

    Args:
        fn: picklable ``fn(dwi, mask[, d_field=...])`` returning one (X, Y, Z, ...) array
            or a tuple of them, computed in-process. Usually a ``functools.partial`` of an
            engine function with ``workers=1``.
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) bool, the voxels to compute.
        patch: the patch shape; its half sets the slab border.
        outputs: the full-size arrays to write into, holding what voxels outside the
            mask must keep. Only mask voxels are written.
        workers: pool size.
        tile: patches per tile in the workers; chunks hold whole tiles.
        d_field: (X, Y, Z) level field passed on to ``fn``, or None.

    Returns:
        ``outputs``, filled.
    """
    pad = tuple(int(p) // 2 for p in patch)
    shape = mask.shape
    fn = partial(fn, tile=int(tile))
    pool = get_pool(workers)
    futures = {}
    with shared_volume(dwi, d_field, pad) as path:
        for chunk in voxel_chunks(mask, 8 * workers, int(tile)):
            ix, iy, iz = np.unravel_index(chunk, shape)
            lo, hi = int(ix[0]), int(ix[-1]) + 1
            inner = np.zeros((hi - lo + 2 * pad[0], shape[1] + 2 * pad[1],
                              shape[2] + 2 * pad[2]), bool)
            inner[ix - lo + pad[0], iy + pad[1], iz + pad[2]] = True
            task = (fn, path, lo, hi, pad, dwi.shape[3], d_field is not None, inner)
            futures[pool.submit(_slab_task, task)] = (ix, iy, iz)
        for future in as_completed(futures):
            where = futures.pop(future)            # drop the chunk's result once written
            for out, part in zip(outputs, future.result(), strict=True):
                out[where] = part
    return outputs
