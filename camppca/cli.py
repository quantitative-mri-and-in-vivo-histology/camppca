"""Command line interface.

    camppca denoise  in.nii out.nii [--noise n.nii] [--qc DIR] -fslgrad in.bvec in.bval
    camppca estimate in.nii --qc DIR                           -fslgrad in.bvec in.bval

Outputs split by AUDIENCE, not by kind:

    positional out   the denoised image
    --noise PATH     the sigma_hat map -- a downstream product (SNR maps, weighting),
                     first-class, as `dwidenoise -noise` is
    --qc DIR         things to look at: rho as NIfTI + a plot, the level field d,
                     sigma_p2s, rank, provenance.json

⚠ `--qc` is deliberately a SUPERSET. Today it dumps the free byproducts. When the real QC
feature lands (whiteness before/after, noise-only spectrum against the MP edge, an
index.html) it writes into the SAME directory -- no second flag, no rename.

`--model` (a reloadable NoiseModel npz) is deliberately NOT here. Estimation is seconds
against a minutes-long denoise and is deterministic given the seed, so re-estimating is
identical; the only strong reason is cross-dataset kernel transfer, which is a research
activity for now. Purely additive whenever that changes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import io
from .noise_model import estimate_noise_model

__all__ = ["main"]


def _add_common(p):
    p.add_argument("-fslgrad", nargs=2, metavar=("BVEC", "BVAL"),
                   help="FSL gradient table. Without it no volume can be recognised as "
                        "a b0, so every volume enters the fit.")
    p.add_argument("--mask", metavar="PATH", help="brain mask; estimated from the mean b0 "
                                                  "when omitted")
    p.add_argument("--patch", type=int, nargs=3, default=(5, 5, 5), metavar=("X", "Y", "Z"),
                   help="patch shape, odd in every dimension (default: 5 5 5)")
    p.add_argument("--qc", metavar="DIR", help="write QC byproducts here")
    p.add_argument("--lowb", type=float, default=100.0,
                   help="b-value at or below which a volume counts as a b0 (default: 100)")
    p.add_argument("-q", "--quiet", action="store_true")


def _parser():
    p = argparse.ArgumentParser(
        prog="camppca", description="Covariance-aware MP-PCA denoising for diffusion MRI.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("denoise", help="estimate the noise model, then denoise")
    d.add_argument("input", help="4D input image")
    d.add_argument("output", help="denoised 4D image")
    d.add_argument("--noise", metavar="PATH", help="write the sigma_hat map here")
    d.add_argument("--no-noise-field", action="store_true",
                   help="ignore the per-voxel level field d (whiten with the pattern only)")
    _add_common(d)

    e = sub.add_parser("estimate", help="estimate the noise model only (fast, no MP-PCA)")
    e.add_argument("input", help="4D input image")
    _add_common(e)
    return p


def _load(args):
    dwi, affine, header = io.load_dwi(args.input)
    bvals = io.load_bvals(args.fslgrad[1]) if args.fslgrad else None
    mask = None
    if args.mask:
        mask = io.load_mask(args.mask)
    return dwi, bvals, mask, affine, header


def _write_qc(directory, model, affine, result=None):
    """Free byproducts, as things to look at. Not a reload unit -- see the module docstring."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    io.save_like(out / "rho.nii.gz", model.rho, np.eye(4))
    io.save_like(out / "noise_field_d.nii.gz", model.d, affine)
    io.save_like(out / "sigma_p2s.nii.gz", model.sigma_p2s, affine)
    io.save_like(out / "mask.nii.gz", model.mask.astype(np.uint8), affine, dtype=np.uint8)

    prov = dict(model=model.provenance)
    if result is not None:
        io.save_like(out / "sigma_hat.nii.gz", result.sigma_hat, affine)
        io.save_like(out / "rank.nii.gz", result.rank, affine)
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


def main(argv=None):
    args = _parser().parse_args(argv)
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    dwi, bvals, mask, affine, header = _load(args)
    patch = tuple(args.patch)

    say(f"  {Path(args.input).name}  {dwi.shape}")
    noise_field = not getattr(args, "no_noise_field", False)   # estimate has no such flag
    model = estimate_noise_model(dwi, bvals=bvals, mask=mask, patch=patch,
                                 lowb=args.lowb, noise_field=noise_field)
    # A patch one voxel wide in-plane gives lag 0: no x=1 entry to show.
    lag1 = f", rho(x=1) {model.rho[model.lag + 1, model.lag, 0]:+.3f}" if model.lag else ""
    say(f"  noise model: lag {model.lag}{lag1}, sigma_p2s median "
        f"{np.median(model.sigma_p2s[model.mask]):.1f}")

    result = None
    if args.command == "denoise":
        from .denoiser import denoise_with_noise_model
        result = denoise_with_noise_model(dwi, model, patch=patch,
                                          noise_field=noise_field)
        io.save_like(args.output, result.denoised, affine, header)
        say(f"  wrote {args.output}")
        if args.noise:
            io.save_like(args.noise, result.sigma_hat, affine)
            say(f"  wrote {args.noise}")

    if args.qc:
        plotted = _write_qc(args.qc, model, affine, result)
        say(f"  wrote QC to {args.qc}/")
        if not plotted:
            print("  note: rho.png skipped (matplotlib not installed)", file=sys.stderr)
    elif args.command == "estimate":
        print("camppca estimate: nothing written. Pass --qc DIR to keep the estimate.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":                    # pragma: no cover
    raise SystemExit(main())
