"""Command line interface.

    camppca denoise  in.nii out.nii [--noise n.nii] [--qc DIR] -fslgrad in.bvec in.bval
    camppca estimate in.nii --qc DIR                           -fslgrad in.bvec in.bval
    camppca phase    mag.nii phase.nii real.nii [--imag PATH] [--phase-range LOW HIGH]

Outputs split by AUDIENCE, not by kind:

    positional out   the denoised image
    <out>_mask       the brain mask, written next to the denoised image ONLY when no
                     --mask was given. A mask the user supplies is the recommended way;
                     the estimated one is a fallback, written so it can be checked and
                     reused.
    --noise PATH     the sigma map -- a downstream product (SNR maps, weighting),
                     first-class, as `dwidenoise -noise` is
    --qc DIR         things to look at: rho as NIfTI + a plot, the relative noise map d,
                     rank, provenance.json

⚠ `--qc` is deliberately a SUPERSET. Today it dumps the free byproducts. When the real QC
feature lands (whiteness before/after, noise-only spectrum against the MP edge, an
index.html) it writes into the SAME directory -- no second flag, no rename.

`--model` (a reloadable NoiseModel npz) is deliberately NOT here. Estimation is seconds
against a minutes-long denoise and is deterministic, so re-estimating is
identical; the only strong reason is cross-dataset rho transfer, which is a research
activity for now. Purely additive whenever that changes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import io
from .denoiser.aggregate import check_aggregate
from .denoiser.mp_threshold import ALGORITHMS
from .denoiser.shrink import RULES
from .noise_model import estimate_noise_model
from .phase import RADIANS, phase_scale, stabilize_phase

__all__ = ["main"]


def _add_common(p):
    p.add_argument("-fslgrad", nargs=2, metavar=("BVEC", "BVAL"),
                   help="FSL gradient table. Only used to find the b0s the default "
                        "mask is drawn from. Without it the mask comes from the mean of "
                        "every volume.")
    p.add_argument("--mask", metavar="PATH",
                   help="brain mask (recommended), where the noise model is estimated. "
                        "Without it one is estimated from the "
                        "mean b0, and denoise writes it next to the output as "
                        "<output>_mask.nii.gz")
    p.add_argument("--patch", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="patch shape, odd in every dimension (default: dwidenoise's rule, "
                        "the smallest odd cube with at least as many voxels as volumes)")
    p.add_argument("--qc", metavar="DIR", help="write QC byproducts here")
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
    p.add_argument("-q", "--quiet", action="store_true")


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
    d.add_argument("--aggregate", choices=("center", "mean"), default="center",
                   help="write back each patch's center voxel (default), or the mean of "
                        "the overlapping patch reconstructions")
    d.add_argument("--stride", type=int, default=1, metavar="N",
                   help="with --aggregate mean, a patch center every N voxels per axis "
                        "(default: 1)")
    d.add_argument("--shrink", choices=tuple(RULES), default="hard",
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
    s.add_argument("-q", "--quiet", action="store_true")
    return p


def _load(args):
    dwi, affine, header = io.load_dwi(args.input)
    bvals = io.load_bvals(args.fslgrad[1]) if args.fslgrad else None
    mask = None
    if args.mask:
        mask = io.load_mask(args.mask, affine)
    return dwi, bvals, mask, affine, header


def _write_qc(directory, model, affine, header, result=None):
    """Free byproducts, as things to look at. Not a reload unit -- see the module docstring."""
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


def _phase(args, say):
    magnitude, affine, header = io.load_dwi(args.magnitude)
    phase, phase_affine, _ = io.load_dwi(args.phase)
    io.check_same_space(phase_affine, affine, args.phase, args.magnitude)
    phase_range = RADIANS if args.phase_range is None else tuple(args.phase_range)
    scale = phase_scale(phase, phase_range)    # checked before any work is done
    slice_axis = header.get_dim_info()[2]
    slice_axis = 2 if slice_axis is None else slice_axis
    say(f"  {Path(args.magnitude).name}  {magnitude.shape}, phase x {scale:.6g} to radians, "
        f"slices along axis {slice_axis}")
    out = stabilize_phase(magnitude, phase, slice_axis=slice_axis, imag=bool(args.imag),
                          workers=args.workers, phase_range=phase_range)
    real, imag = out if args.imag else (out, None)
    del magnitude, phase
    io.save_like(args.output, real, affine, header)
    say(f"  wrote {args.output}")
    if imag is not None:
        io.save_like(args.imag, imag, affine, header)
        say(f"  wrote {args.imag}")
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    if args.command == "phase":
        return _phase(args, say)
    if args.command == "estimate" and not args.qc:     # before the minutes of work
        print("camppca estimate: nothing written. Pass --qc DIR to keep the estimate.",
              file=sys.stderr)
        return 1
    if args.command == "denoise":
        check_aggregate(args.aggregate, args.stride)    # before the estimate, too
    dwi, bvals, mask, affine, header = _load(args)
    patch = None if args.patch is None else tuple(args.patch)

    say(f"  {Path(args.input).name}  {dwi.shape}")
    noise_field = not getattr(args, "no_noise_field", False)   # estimate has no such flag
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch,
                                 lowb=args.lowb, noise_field=noise_field,
                                 algorithm=args.algorithm, level_rounds=args.level_rounds,
                                 workers=args.workers)
    # A patch one voxel wide in-plane gives lag 0: no x=1 entry to show.
    lag1 = f", rho(x=1) {model.rho[model.lag + 1, model.lag, 0]:+.3f}" if model.lag else ""
    say(f"  noise model: patch {'x'.join(map(str, model.provenance['patch']))}, "
        f"lag {model.lag}{lag1}, sphericity {model.provenance['sphericity']:.3f}")

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
        say(f"  wrote {args.output}")
        if mask is None:
            mask_path = _mask_path(args.output)
            io.save_like(mask_path, model.mask.astype(np.uint8), affine, header,
                         dtype=np.uint8)
            say(f"  wrote {mask_path}: no --mask given, so one was estimated from the "
                "mean b0. Check it; supplying your own brain mask is recommended.")
        if args.noise:
            io.save_like(args.noise, result.sigma, affine, header)
            say(f"  wrote {args.noise}")

    if args.qc:
        plotted = _write_qc(args.qc, model, affine, header, result)
        say(f"  wrote QC to {args.qc}/")
        if not plotted:
            print("  note: rho.png skipped (matplotlib not installed)", file=sys.stderr)
    return 0


if __name__ == "__main__":                    # pragma: no cover
    raise SystemExit(main())
