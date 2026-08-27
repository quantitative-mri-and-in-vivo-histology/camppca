"""The engine's three reductions, asserted.

`selftest` returns METRICS, not pass/fail -- upstream it printed them and moved on, and
its S=1 reduction was wrapped in a bare `except` that recorded "skipped" on any failure,
so a broken import degraded it to two reductions silently. Here the thresholds are
explicit and a skip is a failure.

All three metrics are BIMODAL across runs: XLA picks one of two compilation plans and
the float32 rounding follows it. Observed over repeated processes (jax 0.10.2, CUDA),
identical pre- and post-port:

                          plan A     plan B
    identity_vs_engine    0.0        1.26e-04     signal_rel
    global_vs_percol_S1   2.46e-07   5.71e-04     signal_rel
    percol_vs_literal     1.02e-03   1.81e-03     max_rel_diff
    (sigma and rank were exactly 0 in every run of every reduction)

So every bound here is a BOUND, never an equality -- roughly an order of magnitude above
the worse plan, and orders below anything an algebraic error would produce. The exact
zeros for sigma and rank are asserted tightly: those are integers and threshold decisions,
which must not drift at all.
"""

import numpy as np
import pytest

from _reference_denoise_white import denoise_white
from camppca.denoiser.camppca import selftest


@pytest.fixture(scope="module")
def report():
    return selftest(global_whitener=denoise_white, verbose=False)


def test_all_three_reductions_ran(report):
    for name, value in report.items():
        assert not isinstance(value, str), f"{name} did not run: {value}"
    assert set(report) == {"identity_vs_engine", "global_vs_percol_S1", "percol_vs_literal"}


def test_identity_whitener_reproduces_the_plain_engine(report):
    """W = I must be bit-for-bit the standard MP-PCA baseline."""
    r = report["identity_vs_engine"]
    assert r["signal_rel"] < 1e-3
    assert r["sigma"] == 0
    assert r["rank"] == 0


def test_one_shell_collapses_onto_the_global_whitener(report):
    """S = 1: the per-column whitener must reduce to whitening the patch once."""
    r = report["global_vs_percol_S1"]
    assert r["signal_rel"] < 5e-3
    assert r["sigma"] == 0
    assert r["rank"] == 0


def test_multi_shell_matches_the_literal_float64_path(report):
    """S > 1: the fast S-candidate-centre-rows trick vs whiten/denoise/unwhiten in full."""
    r = report["percol_vs_literal"]
    assert r["rank_agree"] == r["rank_total"], "every sampled patch must retain the same rank"
    assert not np.isnan(r["max_rel_diff"])
    assert r["max_rel_diff"] < 5e-3
