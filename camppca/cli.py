"""Command line interface.

    camppca denoise  in.nii out.nii [--mask PATH] [-fslgrad BVEC BVAL] [--noise PATH] [--qc DIR]
    camppca estimate in.nii [--mask PATH] [-fslgrad BVEC BVAL] [--qc DIR]
    camppca phase    mag.nii phase.nii real.nii [--imag PATH] [--phase-range LOW HIGH]

Outputs are split by audience:

    positional out   the denoised image
    <out>_mask       the brain mask, written next to the denoised image only when no
                     --mask was given. A mask the user supplies is the recommended way.
                     The estimated one is a fallback, written so it can be checked and
                     reused.
    --noise PATH     the sigma map, a downstream product for SNR maps or weighting, as
                     `dwidenoise -noise` is
    --qc DIR         things to look at: rho as NIfTI and a plot, the relative noise map d,
                     rank, provenance.json, and the check of one rho across regions and
                     volume halves (noise_model_check.txt and .json)

Status lines go to stderr, each stamped with the time since the start, and stdout stays
empty. ``-q`` keeps only warnings and errors, ``-v`` adds a line per loop round. The
library logs through :mod:`logging`, and only this command configures it. Its warnings
come out as stamped lines too, and every line clears the progress bar before it prints.

The output paths are checked before any data is read. An error in the inputs is one line
and exit code 1, with the traceback under ``-v``.

Everything to look at goes into the ``--qc`` directory. The check
(:func:`camppca.check_noise_model`) runs right after the estimate, before the denoise
holds the output in memory. It refits the loop 23 times.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import io
from .denoiser.aggregate import AGGREGATES, check_aggregate
from .denoiser.mp_threshold import ALGORITHMS
from .denoiser.shrink import SHRINKERS
from .noise_model import estimate_noise_model
from .noise_model.qc import check_noise_model
from .phase import RADIANS, phase_scale, stabilize_phase
from .pool import resolve_workers
from .progress import elapsed

__all__ = ["main"]

# Named, not __name__: run as ``python -m camppca.cli`` that is "__main__", outside camppca.
log = logging.getLogger("camppca.cli")


def _add_common(p):
    p.add_argument("-fslgrad", nargs=2, metavar=("BVEC", "BVAL"),
                   help="FSL gradient table. Only used to find the b0s the default "
                        "mask is drawn from. Without it the mask comes from the mean of "
                        "every volume.")
    p.add_argument("--mask", metavar="PATH",
                   help="brain mask (recommended), where the noise model is estimated. "
                        "Without it one is estimated from the mean b0, the mean of "
                        "every volume without -fslgrad, and denoise writes it next to "
                        "the output as "
                        "<output>_mask.nii.gz")
    p.add_argument("--patch", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="patch shape, odd in every dimension (default: dwidenoise's rule, "
                        "the smallest odd cube with at least as many voxels as volumes)")
    p.add_argument("--qc", metavar="DIR",
                   help="write things to look at here, and check that one correlation "
                        "pattern holds across regions and volume halves. The check "
                        "refits the noise model 23 times.")
    p.add_argument("--lowb", type=float, default=100.0,
                   help="b-value at or below which a volume counts as a b0 (default: 100)")
    p.add_argument("--level-rounds", type=int, default=0, metavar="N",
                   help="re-estimate the relative noise map d from the whitened patches up to N "
                        "times (default: 0, keep standard MPPCA's level)")
    p.add_argument("--algorithm", choices=ALGORITHMS, default="cordero-grande",
                   help="Marchenko-Pastur rule, for the noise model and the denoise "
                        "(default: cordero-grande)")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="CPU processes to split the work over (default: one per "
                        "physical core; 1 runs in one process)")
    _add_verbosity(p)


def _add_verbosity(p):
    talk = p.add_mutually_exclusive_group()
    talk.add_argument("-q", "--quiet", action="store_true",
                      help="show only warnings and errors")
    talk.add_argument("-v", "--verbose", action="store_true",
                      help="also show each round of the correlation loop")


def _mask_path(output):
    """``out.nii.gz`` -> ``out_mask.nii.gz``; ``out.nii`` -> ``out_mask.nii``."""
    out = Path(output)
    for suffix in (".nii.gz", ".nii"):
        if out.name.endswith(suffix):
            return out.with_name(out.name[:-len(suffix)] + "_mask" + suffix)
    return out.with_name(out.name + "_mask.nii.gz")


def _parser():
    p = argparse.ArgumentParser(
        prog="camppca", description="Covariance-aware MPPCA denoising for diffusion MRI.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("denoise", help="estimate the noise model, then denoise")
    d.add_argument("input", help="4D input image, real or complex")
    d.add_argument("output", help="denoised 4D image, complex if the input is")
    d.add_argument("--noise", metavar="PATH", help="write the sigma map here")
    d.add_argument("--no-noise-field", action="store_true",
                   help="ignore the per-voxel relative noise map d (whiten with the pattern only)")
    d.add_argument("--aggregate", choices=AGGREGATES, default="center",
                   help="write back each patch's center voxel (default), or the mean of "
                        "the overlapping patch reconstructions")
    d.add_argument("--stride", type=int, default=1, metavar="N",
                   help="with --aggregate mean, a patch center every N voxels per axis "
                        "(default: 1)")
    d.add_argument("--shrink", choices=SHRINKERS, default="hard",
                   help="weight on the kept components: hard threshold (default), or "
                        "gd or lr shrinkage")
    d.add_argument("--mask-only", action="store_true",
                   help="denoise only inside the mask and pass the rest through "
                        "(default: denoise every voxel)")
    _add_common(d)

    e = sub.add_parser("estimate", help="estimate the noise model only, no denoised image")
    e.add_argument("input", help="4D input image")
    _add_common(e)

    s = sub.add_parser("phase", help="rotate complex data onto the real axis; the real "
                                     "part then denoises like a magnitude image")
    s.add_argument("magnitude", help="4D magnitude image")
    s.add_argument("phase", help="4D phase image, in radians unless --phase-range says "
                                 "otherwise")
    s.add_argument("output", help="4D real part")
    s.add_argument("--phase-range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   default=None, help="the phase values that span one full turn, e.g. "
                                      "-4096 4096 for Siemens integers or -180 180 for "
                                      "degrees (default: radians, -pi pi)")
    s.add_argument("--imag", metavar="PATH",
                   help="also write the imaginary part, which should hold noise only")
    s.add_argument("--workers", type=int, default=None, metavar="N",
                   help="CPU processes to split the volumes over (default: one per "
                        "physical core; 1 runs in one process)")
    _add_verbosity(s)
    return p


def _load(args):
    dwi, affine, header = io.load_dwi(args.input)
    bvals = io.load_bvals(args.fslgrad[1]) if args.fslgrad else None
    mask = None
    if args.mask:
        mask = io.load_mask(args.mask, affine)
    return dwi, bvals, mask, affine, header


def _write_qc(directory, model, affine, header, result=None):
    """Free byproducts, as things to look at."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    io.save_like(out / "rho.nii.gz", model.rho, np.eye(4))
    io.save_like(out / "noise_field_d.nii.gz", model.d, affine, header)
    io.save_like(out / "mask.nii.gz", model.mask.astype(np.uint8), affine, header,
                 dtype=np.uint8)

    prov = dict(model=model.provenance)
    if result is not None:
        io.save_like(out / "sigma.nii.gz", result.sigma, affine, header)
        io.save_like(out / "rank.nii.gz", result.rank, affine, header)
        prov["denoise"] = result.provenance
    io.save_json(out / "provenance.json", prov)

    return _plot_rho(out / "rho.png", model)


def _write_check(directory, check):
    """The check of the noise model, as text to read and as JSON."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise_model_check.txt").write_text("\n".join(check.report()) + "\n")
    io.save_json(out / "noise_model_check.json",
                 dict(regions=check.regions, reference=check.reference,
                      volume_splits=check.volume_splits,
                      flagged=check.flagged(),
                      provenance=check.provenance))


def _plot_rho(path, model):
    """Lag profiles along each axis. Optional: skipped, loudly, without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    lag, rho = model.lag, model.rho
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    k = np.arange(-lag, lag + 1)
    ax.plot(k, rho[:, lag, 0], "o-", label="along x")
    ax.plot(k, rho[lag, :, 0], "s-", label="along y")
    ax.plot(np.arange(rho.shape[2]), rho[lag, lag, :], "^-", label="along z")
    ax.axhline(0.0, lw=0.8, color="0.6")
    ax.set_xlabel("lag (voxels)")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


class _Lines(logging.StreamHandler):
    """Writes each line through tqdm, which moves a progress bar out of its way."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record), file=self.stream)
        except Exception:
            self.handleError(record)


def _log_warning(message, category, filename, lineno, file=None, line=None):
    """``warnings.showwarning`` for the run: a warning is one stamped line."""
    log.warning("%s", message)


class _Stamp(logging.Formatter):
    """``[m:ss] message``, the time since ``start``. Warnings and errors say so."""

    def __init__(self, start):
        super().__init__()
        self.start = start

    def format(self, record):
        minutes, seconds = divmod(int(record.created - self.start), 60)
        kind = f"{record.levelname.lower()}: " if record.levelno >= logging.WARNING else ""
        line = f"[{minutes}:{seconds:02d}] {kind}{record.getMessage()}"
        return f"{line}\n{self.formatException(record.exc_info)}" if record.exc_info else line


def _lag1(rho):
    """rho one voxel along x, y and z, each where the table reaches."""
    c = rho.shape[0] // 2                       # the zero lag
    steps = {"x": (c + 1, c, 0), "y": (c, c + 1, 0), "z": (c, c, 1)}
    # A patch one voxel wide on an axis has no lag along it.
    return "".join(f", rho({axis}=1) {rho[at]:+.3f}" for axis, at in steps.items()
                   if all(i < n for i, n in zip(at, rho.shape, strict=True)))


def _shape(dwi):
    return f"{' x '.join(str(n) for n in dwi.shape[:3])}, {dwi.shape[3]} volumes"


def _workers(workers):
    n = resolve_workers(workers)
    return f"{n} worker{'s' if n > 1 else ''}"


def _check_paths(inputs, outputs, qc=None):
    """Refuse output paths that would fail at the write, after all the work.

    Each output must be NIfTI, in a folder that exists and can be written, writable if it
    exists already, and no input. ``qc`` must be a folder or not exist yet, under a folder
    that can be written.
    """
    given = [Path(p).resolve() for p in inputs if p]
    seen = []
    for path in filter(None, outputs):
        out = Path(path)
        if not out.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError(f"{path}: outputs are NIfTI, so the name must end in .nii "
                             f"or .nii.gz")
        folder = out.parent
        if not folder.is_dir():
            raise ValueError(f"{path}: the folder {folder} does not exist")
        if not os.access(folder, os.W_OK):
            raise ValueError(f"{path}: the folder {folder} cannot be written to")
        if out.exists() and not os.access(out, os.W_OK):
            raise ValueError(f"{path}: the file exists and cannot be written to")
        if out.resolve() in given:
            raise ValueError(f"{path}: this is also an input, and writing would overwrite it")
        if out.resolve() in seen:
            raise ValueError(f"{path}: two outputs would be written to this file")
        seen.append(out.resolve())
    if qc:
        folder = Path(qc)
        if folder.exists() and not folder.is_dir():
            raise ValueError(f"--qc {qc}: this is a file, and --qc takes a folder")
        existing = next(p for p in (folder, *folder.parents) if p.exists())
        if not existing.is_dir():
            raise ValueError(f"--qc {qc}: {existing} is a file, so the folder cannot be made")
        if not os.access(existing, os.W_OK):
            raise ValueError(f"--qc {qc}: the folder {existing} cannot be written to")


def _phase(args):
    _check_paths([args.magnitude, args.phase], [args.output, args.imag])
    magnitude, affine, header = io.load_dwi(args.magnitude)
    phase, phase_affine, _ = io.load_dwi(args.phase)
    io.check_same_space(phase_affine, affine, args.phase, args.magnitude)
    if phase.shape != magnitude.shape:
        raise ValueError(f"{args.phase}: shape {phase.shape}, and the magnitude's is "
                         f"{magnitude.shape}")
    phase_range = RADIANS if args.phase_range is None else tuple(args.phase_range)
    scale = phase_scale(phase, phase_range)    # checked before any work is done
    slice_axis = header.get_dim_info()[2]
    slice_axis = 2 if slice_axis is None else slice_axis
    log.info("%s: %s, phase x %.6g to radians, slices along axis %d, %s",
             Path(args.magnitude).name, _shape(magnitude), scale, slice_axis,
             _workers(args.workers))
    out = stabilize_phase(magnitude, phase, slice_axis=slice_axis, imag=bool(args.imag),
                          workers=args.workers, phase_range=phase_range)
    real, imag = out if args.imag else (out, None)
    del magnitude, phase
    io.save_like(args.output, real, affine, header)
    log.info("wrote %s", args.output)
    if imag is not None:
        io.save_like(args.imag, imag, affine, header)
        log.info("wrote %s", args.imag)
    return 0


def _run(args):
    if args.command == "phase":
        return _phase(args)
    if args.command == "estimate" and not args.qc:     # before the minutes of work
        log.error("camppca estimate: nothing written. Pass --qc DIR to keep the estimate.")
        return 1
    inputs = [args.input, args.mask, *(args.fslgrad or [])]
    if args.command == "denoise":
        check_aggregate(args.aggregate, args.stride)    # before the estimate, too
        written = [args.output, args.noise, None if args.mask else _mask_path(args.output)]
    else:
        written = []
    _check_paths(inputs, written, args.qc)
    dwi, bvals, mask, affine, header = _load(args)
    patch = None if args.patch is None else tuple(args.patch)

    log.info("%s: %s, %s", Path(args.input).name, _shape(dwi), _workers(args.workers))
    noise_field = not getattr(args, "no_noise_field", False)   # estimate has no such flag
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch,
                                 lowb=args.lowb, noise_field=noise_field,
                                 algorithm=args.algorithm, level_rounds=args.level_rounds,
                                 workers=args.workers)
    log.info("noise model: patch %s, lag %d%s, sphericity %.3f",
             "x".join(map(str, model.provenance["patch"])), model.lag, _lag1(model.rho),
             model.provenance["sphericity"])

    if args.qc:                    # before the denoise, which holds the output in memory
        log.info("checking the noise model across regions and volume halves")
        check = check_noise_model(dwi, model, workers=args.workers)
        _write_check(args.qc, check)
        for line in check.report():
            log.info("  %s", line)

    result = None
    if args.command == "denoise":
        from .denoiser import denoise_with_noise_model
        result = denoise_with_noise_model(dwi, model, patch=patch,
                                          noise_field=noise_field, workers=args.workers,
                                          algorithm=args.algorithm, shrink=args.shrink,
                                          aggregate=args.aggregate, stride=args.stride,
                                          mask_only=args.mask_only)
        io.save_like(args.output, result.denoised, affine, header,
                     dtype=result.denoised.dtype)
        log.info("wrote %s", args.output)
        if mask is None:
            mask_path = _mask_path(args.output)
            io.save_like(mask_path, model.mask.astype(np.uint8), affine, header,
                         dtype=np.uint8)
            log.info("wrote %s: no --mask given, so one was estimated from the mean %s. "
                     "Check it; supplying your own brain mask is recommended.", mask_path,
                     "b0" if bvals is not None else "of every volume")
        if args.noise:
            io.save_like(args.noise, result.sigma, affine, header)
            log.info("wrote %s", args.noise)

    if args.qc:
        plotted = _write_qc(args.qc, model, affine, header, result)
        log.info("wrote QC to %s/", args.qc)
        if not plotted:
            log.warning("rho.png skipped (matplotlib not installed)")
    return 0


def _expand_home(args):
    """``~`` in every path, for a quoted path or one from a script, which no shell expands."""
    for name in ("input", "output", "mask", "noise", "qc", "magnitude", "phase", "imag"):
        if getattr(args, name, None):
            setattr(args, name, os.path.expanduser(getattr(args, name)))
    if getattr(args, "fslgrad", None):
        args.fslgrad = [os.path.expanduser(p) for p in args.fslgrad]


def main(argv=None):
    args = _parser().parse_args(argv)
    _expand_home(args)
    # The camppca logger and the warning display are set up for this call and restored
    # after, so a script or test calling main keeps its own.
    package = logging.getLogger("camppca")
    saved = package.level, package.propagate, warnings.showwarning
    start = time.time()
    handler = _Lines(sys.stderr)
    handler.setFormatter(_Stamp(start))
    package.addHandler(handler)
    package.setLevel(logging.WARNING if args.quiet else
                     logging.DEBUG if args.verbose else logging.INFO)
    package.propagate = False
    warnings.showwarning = _log_warning
    try:
        status = _run(args)
        if status == 0:
            log.info("done in %s", elapsed(time.time() - start))
    except (ValueError, OSError) as err:       # the inputs, a missing file, a full disk
        # A bug can raise a ValueError too, so the traceback is one flag away.
        log.error("%s%s", err, "" if args.verbose else " (run with -v for the traceback)",
                  exc_info=args.verbose)
        status = 1
    finally:
        package.removeHandler(handler)
        package.setLevel(saved[0])
        package.propagate = saved[1]
        warnings.showwarning = saved[2]
    return status


if __name__ == "__main__":                    # pragma: no cover
    raise SystemExit(main())
