"""Status of the long passes: a progress bar on a terminal, plain lines elsewhere.

camppca logs through :mod:`logging` and never configures it. Imported, it says nothing
until the caller turns on INFO for the ``camppca`` logger, as the command line does.
Warnings are separate: they go through :func:`camppca.warn.warn`.
"""

import logging
import sys
import time
from contextlib import contextmanager

from tqdm import tqdm

__all__ = ["progress", "elapsed"]

log = logging.getLogger(__name__)

#: Percent between the plain lines written when stderr is not a terminal.
STEP = 10

BAR = "{desc} {percentage:3.0f}%|{bar:20}| [{elapsed}<{remaining}]"


def elapsed(seconds: float) -> str:
    """A duration for a status line: "1.1 s", "9 min 2 s" or "1 h 5 min"."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {secs} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


@contextmanager
def progress(total: int, label: str | None):
    """Count finished work, yielding ``update(n=1)``.

    Nothing is shown when ``label`` is None or the ``camppca`` logger is not enabled for
    INFO, the default. Otherwise a terminal gets a bar on stderr, and anything else, a
    batch job's log say, an INFO line each time another :data:`STEP` percent is done.
    Either way an INFO line with the time taken follows.
    """
    if label is None or not log.isEnabledFor(logging.INFO):
        yield lambda n=1: None
        return
    start = time.perf_counter()
    if sys.stderr.isatty():
        with tqdm(total=total, desc=label, file=sys.stderr, leave=False,
                  bar_format=BAR) as bar:
            yield bar.update
    else:
        done, shown = 0, 0

        def update(n=1):
            nonlocal done, shown
            done += n
            percent = 100 * done // total
            if percent >= shown + STEP and done < total:
                shown = percent - percent % STEP
                log.info("%s %d%%", label, shown)

        yield update
    log.info("%s done in %s", label, elapsed(time.perf_counter() - start))
