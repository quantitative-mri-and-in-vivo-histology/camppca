"""Status output: silent by default, a bar on a terminal, plain lines elsewhere."""

import io
import logging
import sys

import numpy as np
import pytest

from camppca import denoise_mppca
from camppca.progress import elapsed, progress


class _Stream(io.StringIO):
    """A stderr that says whether it is a terminal."""

    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


def _volume():
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 12, 12, 30)).astype(np.float32)


def _count(n=20):
    with progress(n, "counting") as done:
        for _ in range(n):
            done()


@pytest.mark.parametrize("seconds, text", [(1.06, "1.1 s"), (59.9, "59.9 s"),
                                           (542, "9 min 2 s"), (3900, "1 h 5 min")])
def test_elapsed_reads_as_a_duration(seconds, text):
    assert elapsed(seconds) == text


def test_the_library_is_silent_by_default(caplog, monkeypatch):
    terminal = _Stream(tty=True)
    monkeypatch.setattr(sys, "stderr", terminal)
    _count()
    denoise_mppca(_volume(), patch=(3, 3, 3), workers=1)
    assert caplog.records == []
    assert terminal.getvalue() == "", "no bar either"


def test_progress_writes_a_line_every_ten_percent_off_a_terminal(caplog, monkeypatch):
    log_file = _Stream(tty=False)
    monkeypatch.setattr(sys, "stderr", log_file)
    with caplog.at_level(logging.INFO, logger="camppca"):
        _count()
    messages = [r.getMessage() for r in caplog.records]
    assert messages[:-1] == [f"counting {p}%" for p in range(10, 100, 10)]
    assert messages[-1].startswith("counting done in")
    assert log_file.getvalue() == "", "no bar"


def test_progress_draws_a_bar_on_a_terminal(caplog, monkeypatch):
    terminal = _Stream(tty=True)
    monkeypatch.setattr(sys, "stderr", terminal)
    with caplog.at_level(logging.INFO, logger="camppca"):
        _count()
    assert "\rcounting   0%|" in terminal.getvalue(), "the bar, redrawn at most every 0.1 s"
    assert [r.getMessage()[:16] for r in caplog.records] == ["counting done in"]


def test_the_denoise_pass_reports_its_progress(caplog):
    with caplog.at_level(logging.INFO, logger="camppca"):
        denoise_mppca(_volume(), patch=(3, 3, 3), workers=1)
    assert caplog.records[-1].getMessage().startswith("denoising done in")
