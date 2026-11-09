"""The Marchenko-Pastur threshold: where the noise bulk ends and signal begins.

Both denoisers reduce a patch to the descending eigenvalues of its Gram matrix and ask
this module the same two questions. How many components are signal? What noise level do
the rest imply?

For a matrix of pure i.i.d. noise the eigenvalues follow the Marchenko-Pastur
distribution, whose support has a width fixed by the noise level and the matrix shape.
Walk down the spectrum. At each cut the eigenvalues below it imply a noise level, and
that level predicts how wide their spread should be. The first cut where the observed
spread falls short of the predicted one is the edge of the bulk. Everything above it is
signal.

The three variants differ only in the degrees of freedom they charge for the components
already cut away, so they agree on long spectra and disagree at small N.

References: Veraart 2016, Cordero-Grande 2019, Ades-Aron 2020.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ALGORITHMS", "check_algorithm", "mp_threshold"]

ALGORITHMS = ("veraart", "cordero-grande", "jespersen")


def check_algorithm(algorithm: str) -> str:
    """Validate ``algorithm`` up front, before any work goes to the workers."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")
    return algorithm


def mp_threshold(eigenvalues: np.ndarray, n_small: int, n_large: int,
                 algorithm: str = "cordero-grande") -> tuple[np.ndarray, np.ndarray]:
    """Find the Marchenko-Pastur edge in a batch of spectra.

    Args:
        eigenvalues: (B, K) descending eigenvalues of each patch's Gram matrix, i.e.
            squared singular values. K must be at least ``n_small``.
        n_small: the smaller of the patch matrix dimensions.
        n_large: the larger of them.
        algorithm: one of :data:`ALGORITHMS`.

    Returns:
        ``(rank, sigma_mppca)``: (B,) int32 count of retained components, and (B,) float
        noise level implied by the discarded ones. ``sigma_mppca`` is floored at the dtype
        epsilon so it is always positive and finite.

    Raises:
        ValueError: if ``algorithm`` is not one of :data:`ALGORITHMS`.
    """
    eps = np.finfo(eigenvalues.dtype).eps
    vals = eigenvalues[:, :n_small]                              # (B, n_small)
    cut = np.arange(n_small)                                     # (n_small,)
    # tail[:, i] = sum of the eigenvalues at i and below, the noise the cut would discard
    tail = np.cumsum(vals[:, ::-1], axis=1)[:, ::-1]
    smallest = vals[:, n_small - 1:n_small]                      # (B, 1)

    if algorithm == "veraart":
        predicted = tail / ((n_small - cut) * float(n_large))
        mp_width = 4.0 * np.sqrt((n_small - cut) * float(n_large))
    elif algorithm == "cordero-grande":
        predicted = tail / ((n_small - cut) * (n_large - cut))
        mp_width = 4.0 * np.sqrt((n_small - cut) * (n_large - cut))
    elif algorithm == "jespersen":
        predicted = tail / ((n_small - cut) * (n_large - cut))
        mp_width = 4.0 * np.sqrt(float(n_large) * float(n_small))
    else:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")

    observed = (vals - smallest) / mp_width
    crossed = observed < predicted
    # first crossing, or all but the smallest component if the spread never falls short
    rank = np.where(np.any(crossed, axis=1), np.argmax(crossed, axis=1), n_small - 1)

    sigma_sq = np.take_along_axis(predicted, rank[:, None], axis=1)[:, 0]
    sigma_mppca = np.sqrt(np.maximum(sigma_sq, 0.0))
    sigma_mppca = np.where((sigma_mppca > 0) & np.isfinite(sigma_mppca), sigma_mppca, eps)
    return rank.astype(np.int32), sigma_mppca.astype(eigenvalues.dtype)
