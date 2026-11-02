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

The rule is written once for numpy and JAX arrays alike: pool workers run it in numpy
and never load JAX.

References: Veraart 2016, Cordero-Grande 2019, Ades-Aron 2020.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax

__all__ = ["ALGORITHMS", "mp_threshold"]

ALGORITHMS = ("veraart", "cordero-grande", "jespersen")


def mp_threshold(eigenvalues: jax.Array, n_small: int, n_large: int,
                 algorithm: str = "cordero-grande", xp=None) -> tuple[jax.Array, jax.Array]:
    """Find the Marchenko-Pastur edge in a batch of spectra.

    Args:
        eigenvalues: (B, K) descending eigenvalues of each patch's Gram matrix, i.e.
            squared singular values. K must be at least ``n_small``.
        n_small: the smaller of the patch matrix dimensions.
        n_large: the larger of them.
        algorithm: one of :data:`ALGORITHMS`.
        xp: the array module, ``numpy`` or ``jax.numpy``. Defaults to ``jax.numpy``.

    Returns:
        ``(rank, sigma_hat)``: (B,) int32 count of retained components, and (B,) float
        noise level implied by the discarded ones. ``sigma_hat`` is floored at the dtype
        epsilon so it is always positive and finite.

    Raises:
        ValueError: if ``algorithm`` is not one of :data:`ALGORITHMS`.
    """
    if xp is None:
        import jax.numpy as xp
    eps = xp.finfo(eigenvalues.dtype).eps
    vals = eigenvalues[:, :n_small]                              # (B, n_small)
    cut = xp.arange(n_small)                                     # (n_small,)
    # tail[:, i] = sum of the eigenvalues at i and below, the noise the cut would discard
    tail = xp.cumsum(vals[:, ::-1], axis=1)[:, ::-1]
    smallest = vals[:, n_small - 1:n_small]                      # (B, 1)

    if algorithm == "veraart":
        predicted = tail / ((n_small - cut) * float(n_large))
        mp_width = 4.0 * xp.sqrt((n_small - cut) * float(n_large))
    elif algorithm == "cordero-grande":
        predicted = tail / ((n_small - cut) * (n_large - cut))
        mp_width = 4.0 * xp.sqrt((n_small - cut) * (n_large - cut))
    elif algorithm == "jespersen":
        predicted = tail / ((n_small - cut) * (n_large - cut))
        mp_width = 4.0 * xp.sqrt(float(n_large) * float(n_small))
    else:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}")

    observed = (vals - smallest) / mp_width
    crossed = observed < predicted
    # first crossing, or everything below the top component if the spread never falls short
    rank = xp.where(xp.any(crossed, axis=1), xp.argmax(crossed, axis=1), n_small - 1)

    sigma_sq = xp.take_along_axis(predicted, rank[:, None], axis=1)[:, 0]
    sigma_hat = xp.sqrt(xp.maximum(sigma_sq, 0.0))
    sigma_hat = xp.where((sigma_hat > 0) & xp.isfinite(sigma_hat), sigma_hat, eps)
    return rank.astype(xp.int32), sigma_hat.astype(eigenvalues.dtype)
