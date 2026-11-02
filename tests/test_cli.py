"""CLI end to end: files land where the frozen interface says they do."""

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from camppca.cli import main


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    rng = np.random.default_rng(0)
    shape, n_vols = (34, 34, 16), 50
    bvals = np.concatenate([np.zeros(4), np.full(23, 1000.0), np.full(23, 2000.0)])
    signal = rng.standard_normal(shape + (3,)) @ rng.standard_normal((3, n_vols)) * 20
    dwi = np.zeros(shape + (n_vols,), np.float32)
    dwi[4:30, 4:30, 2:14] = (signal + 400)[4:30, 4:30, 2:14]
    dwi += (rng.standard_normal(shape + (n_vols,)) * 10).astype(np.float32)

    nib.save(nib.Nifti1Image(dwi, np.eye(4)), str(d / "in.nii.gz"))
    np.savetxt(d / "in.bval", bvals[None], fmt="%g")
    np.savetxt(d / "in.bvec", rng.standard_normal((3, n_vols)), fmt="%.4f")
    return d


def _grad(d):
    return ["-fslgrad", str(d / "in.bvec"), str(d / "in.bval")]


def test_denoise_writes_image_noise_map_and_qc(dataset, capsys):
    out, noise, qc = dataset / "out.nii.gz", dataset / "noise.nii.gz", dataset / "qc"
    rc = main(["denoise", str(dataset / "in.nii.gz"), str(out), "--noise", str(noise),
               "--qc", str(qc), "--patch", "5", "5", "3", *_grad(dataset)])
    assert rc == 0

    denoised = nib.load(str(out))
    assert denoised.shape == (34, 34, 16, 50)
    assert np.allclose(denoised.affine, np.eye(4)), "geometry is inherited from the input"

    sigma_hat = np.asanyarray(nib.load(str(noise)).dataobj)
    assert sigma_hat.shape == (34, 34, 16), "the noise map is 3D, not 4D with a stale dim4"
    assert np.median(sigma_hat[sigma_hat > 0]) == pytest.approx(10.0, rel=0.2)

    for name in ("rho.nii.gz", "noise_field_d.nii.gz", "mask.nii.gz",
                 "sigma_hat.nii.gz", "rank.nii.gz", "provenance.json"):
        assert (qc / name).exists(), name

    prov = json.loads((qc / "provenance.json").read_text())
    assert prov["model"]["level_rounds"] == 0, "no level rounds by default"


def test_estimate_writes_qc_but_no_image(dataset):
    qc = dataset / "qc_estimate"
    rc = main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(qc),
               "--patch", "5", "5", "3", *_grad(dataset)])
    assert rc == 0
    assert (qc / "provenance.json").exists()
    assert not (qc / "sigma_hat.nii.gz").exists(), "nothing was denoised"
    assert not (qc / "rank.nii.gz").exists()


def test_estimate_without_qc_writes_nothing_and_says_so(dataset, capsys):
    rc = main(["estimate", str(dataset / "in.nii.gz"), "--patch", "5", "5", "3",
               *_grad(dataset)])
    assert rc == 1
    assert "nothing written" in capsys.readouterr().err


def test_denoise_runs_without_a_gradient_table(dataset):
    qc = dataset / "qc_nograd"
    rc = main(["denoise", str(dataset / "in.nii.gz"), str(dataset / "out2.nii.gz"),
               "--qc", str(qc), "--patch", "5", "5", "3", "-q"])
    assert rc == 0
    prov = json.loads((qc / "provenance.json").read_text())
    assert prov["model"]["n_vols"] == 50


def test_level_rounds_reach_the_estimator(dataset):
    qc = dataset / "qc_level"
    rc = main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(qc),
               "--patch", "5", "5", "3", "--level-rounds", "1", *_grad(dataset), "-q"])
    assert rc == 0
    prov = json.loads((qc / "provenance.json").read_text())
    assert prov["model"]["level_rounds"] == 1
    assert prov["model"]["level_rounds_run"] == 1


def test_even_patch_is_rejected(dataset):
    with pytest.raises(ValueError, match="odd"):
        main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(dataset / "qc_even"),
              "--patch", "4", "5", "5", *_grad(dataset)])


def test_a_qc_rerun_overwrites_rather_than_accumulates(dataset):
    """A QC dir must describe ONE model, so a re-run replaces what the last one wrote."""
    qc = dataset / "qc_rerun"
    main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(qc), "--patch", "5", "5", "3",
          "-q", *_grad(dataset)])
    first = json.loads((qc / "provenance.json").read_text())["model"]

    main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(qc), "--patch", "5", "5", "5",
          "-q", *_grad(dataset)])
    second = json.loads((qc / "provenance.json").read_text())["model"]

    assert first["patch"] == [5, 5, 3] and second["patch"] == [5, 5, 5]
    assert second["lag"] == 4


def test_a_patch_one_voxel_wide_in_plane_runs(dataset):
    """Lag 0 has no x=1 entry; the summary line used to index past the table."""
    qc = dataset / "qc_115"
    rc = main(["estimate", str(dataset / "in.nii.gz"), "--qc", str(qc),
               "--patch", "1", "1", "5", *_grad(dataset), "-q"])
    assert rc == 0
