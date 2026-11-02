"""A pool of pinned CPU processes for the passes that touch every voxel.

Each patch is split on its own, so the work divides over voxels. Threads do not help:
OpenBLAS serializes concurrent eigh calls, from JAX and numpy alike. Separate processes
each load their own OpenBLAS, limited to one thread, and scale about linearly. On a
16-core host a full MP-PCA pass over 297 volumes ran 2.7x faster than on an RTX 4060 Ti.

⚠ Every worker is pinned to one core. XLA sizes its CPU thread pool from the affinity
mask and its idle threads spin, so an unpinned worker takes cores from every other one.
XLA_FLAGS did not stop it. The pin happens in the initializer, before the worker's
first JAX call creates that pool.

Workers are spawned, and they do NOT re-import the caller's main module, which spawn
does by default. They only run camppca's own functions, so a script without the
``if __name__ == "__main__":`` guard, or one read from stdin, works unchanged.

A worker that dies, killed for memory say, raises ``BrokenProcessPool`` rather than
hanging, and the next call starts a fresh pool.

The volume reaches the workers as a memory-mapped ``.npy`` file, not through pickling.
"""

from __future__ import annotations

import atexit
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from multiprocessing.context import SpawnContext, SpawnProcess
from pathlib import Path

import numpy as np

__all__ = ["default_workers", "resolve_workers", "get_pool", "shared_volume",
           "slab_edges", "run_slabs"]

# Set while the workers start, so they load numpy, OpenBLAS and JAX with it.
_WORKER_ENV = dict(JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")

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


def slab_edges(mask, n_slabs: int) -> list[tuple[int, int]]:
    """x ranges holding about equal numbers of mask voxels, skipping empty x slices.

    Returns:
        Up to ``n_slabs`` ``(lo, hi)`` pairs. Together they cover every x slice that
        holds a mask voxel exactly once.
    """
    per_x = np.asarray(mask, bool).sum((1, 2))
    filled = np.flatnonzero(per_x)
    if filled.size == 0:
        return []
    cut = np.searchsorted(np.cumsum(per_x[filled]),
                          np.arange(1, n_slabs) * per_x.sum() / n_slabs, side="right")
    edges = []
    for part in np.split(filled, np.unique(cut)):
        if part.size == 0:
            continue
        # Split again wherever empty slices interrupt a part, so no slab spans them.
        for run in np.split(part, np.flatnonzero(np.diff(part) > 1) + 1):
            edges.append((int(run[0]), int(run[-1]) + 1))
    return edges


def _slab_task(args):
    fn, path, lo, hi, pad, n_vols, has_d, inner = args
    vol = np.load(path, mmap_mode="r")
    block = np.array(vol[lo:hi + 2 * pad[0]])
    if has_d:
        outputs = fn(block[..., :n_vols], inner, d_field=block[..., n_vols])
    else:
        outputs = fn(block, inner)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    region = tuple(slice(p, p + n) for p, n in zip(pad, inner.shape - 2 * np.asarray(pad), strict=True))
    return lo, hi, tuple(np.asarray(o[region]) for o in outputs)


def run_slabs(fn, dwi, mask, patch, outputs, workers: int, d_field=None):
    """Run a per-voxel pass on x slabs in the worker pool and stitch the results.

    Args:
        fn: picklable ``fn(dwi, mask[, d_field=...])`` returning one (X, Y, Z, ...) array
            or a tuple of them, computed in-process. Usually a ``functools.partial`` of an
            engine function with ``workers=1``.
        dwi: (X, Y, Z, N) data.
        mask: (X, Y, Z) bool, the voxels to compute.
        patch: the patch shape; its half sets the slab border.
        outputs: the full-size arrays to write into, holding what voxels outside the
            mask must keep. Slabs overwrite their whole inner region, which ``fn`` leaves
            at those same values outside the mask.
        workers: pool size.
        d_field: (X, Y, Z) level field passed on to ``fn``, or None.

    Returns:
        ``outputs``, filled.
    """
    pad = tuple(int(p) // 2 for p in patch)
    pool = get_pool(workers)
    tasks = []
    with shared_volume(dwi, d_field, pad) as path:
        for lo, hi in slab_edges(mask, 2 * workers):
            inner = np.zeros((hi - lo + 2 * pad[0],) + tuple(
                s + 2 * p for s, p in zip(mask.shape[1:], pad[1:], strict=True)), bool)
            inner[pad[0]:pad[0] + hi - lo, pad[1]:pad[1] + mask.shape[1],
                  pad[2]:pad[2] + mask.shape[2]] = mask[lo:hi]
            tasks.append((fn, path, lo, hi, pad, dwi.shape[3], d_field is not None, inner))
        for future in as_completed([pool.submit(_slab_task, task) for task in tasks]):
            lo, hi, parts = future.result()
            for out, part in zip(outputs, parts, strict=True):
                out[lo:hi] = part
    return outputs
