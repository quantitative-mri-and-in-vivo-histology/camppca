"""Shrinkage of the kept components: an opt-in alternative to the hard threshold.

The hard threshold keeps every component above the MP edge at full weight. A shrinker
weights each one instead, by how much of its singular value is signal. Both run on the
whitened patch at the patch's own ``sigma_mppca``, and both keep the MP rank: a component
the rule called noise was counted in ``sigma_mppca`` as noise, so it stays dropped.

``"gd"``  Gavish & Donoho's Frobenius-optimal shrinker. Optimal for the error in the
          whitened patch.
``"lr"``  Leeb & Romanov's correction of it, their Algorithm 1. Unwhitening
          stretches a component's true part by how loud the noise is along that signal
          direction, and its noise part by the average noise loudness, so the error that
          matters, in data units, wants a different weight. With white noise the two agree.

For a patch of ``n_vox`` voxels and ``n_vols`` volumes, ``gamma = n_vox / n_vols``, and
``sw_k = s_k / (sigma_mppca sqrt(n_vols))`` the k-th whitened singular value in units of the
noise. Above the edge ``1 + sqrt(gamma)``::

    l_k    = [sw^2 - 1 - gamma + sqrt((sw^2 - 1 - gamma)^2 - 4 gamma)] / 2
    c_k^2  = (1 - gamma / l_k^2) / (1 + gamma / l_k)       voxel-side cosine
    ct_k^2 = (1 - gamma / l_k^2) / (1 + 1 / l_k)           volume-side cosine
    t_k    = sqrt(l_k) c_k ct_k                            "gd"
    t_k    = sqrt(l_k) c_k ct_k / (c_k^2 + sn_k^2 mu_tau_k) "lr"

with the sine ``sn_k^2 = 1 - c_k^2``, ``mu_tau_k = c_k^2 / (q_k - sn_k^2)`` and
``q_k = u_k^T C u_k / (tr C / n_vox)``, ``u_k`` the k-th left singular vector of the
whitened patch. Where ``q_k <= sn_k^2`` there is no valid ``tau_k`` and "lr" falls back to
"gd". The weight on the component is ``t_k / sw_k``, and 0 below the edge.

With the center voxel as output, ``q_k`` measures noise loudness with ``C`` alone. The
loss that matters is the error at the center voxel, whose level ``d_c`` scales every
component equally and so drops out of the weights; the other voxels' levels never enter.
With patch averaging every voxel of a patch reaches the output, and the loss is Leeb &
Romanov's own, the error over the whole patch in data units. The way back is then
``D C^(1/2)``, and :func:`signal_alignment` given the patch's levels measures
``q_k = |D C^(1/2) u_k|^2 / mean(d^2)``. With equal levels the two agree.

References:
    Gavish M, Donoho DL. Optimal shrinkage of singular values. IEEE Trans Inf
    Theory 2017;63(4):2137-2152. The "gd" shrinker, for Frobenius loss.

    Leeb W, Romanov E. Optimal spectral shrinkage and PCA with heteroscedastic
    noise. IEEE Trans Inf Theory 2021;67(5):3009-3037. The "lr" shrinker.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RULES", "SHRINKERS", "check_shrink", "rule_text", "shrink_weight",
           "signal_alignment"]

SHRINKERS = ("hard", "gd", "lr")

#: How each shrinker reads in a result's provenance.
RULES = {"hard": "hard threshold, no shrinkage",
         "gd": "MP rank, Gavish-Donoho shrinkage",
         "lr": "MP rank, Leeb-Romanov shrinkage"}


def rule_text(algorithm, shrink, aggregate, stride):
    """The rule a denoiser applied, in words, for provenance."""
    where = "center-voxel only" if aggregate == "center" else \
        f"mean over overlapping patches, stride {stride}"
    return f"MP cut {algorithm}, {RULES[shrink]}, {where}"


def check_shrink(shrink: str) -> str:
    """Validate a shrinker name; returns it.

    Raises:
        ValueError: if ``shrink`` is not one of :data:`SHRINKERS`.
    """
    if shrink not in SHRINKERS:
        raise ValueError(f"shrink must be one of {SHRINKERS}, got {shrink!r}")
    return shrink


def shrink_weight(evals: np.ndarray, sigma_mppca: np.ndarray, n_vox: int, n_vols: int,
                  alignment: np.ndarray | None = None) -> np.ndarray:
    """Per component, the shrunk singular value over the observed one.

    Args:
        evals: (B, n_vols) descending eigenvalues of the whitened Gram.
        sigma_mppca: (B,) the patch's MP noise level.
        n_vox: voxels in the patch.
        n_vols: volumes in the patch.
        alignment: (B, n_vols) ``q_k`` from :func:`signal_alignment` for "lr"; None
            for "gd".

    Returns:
        (B, n_vols) weights, 0 below the edge. The caller zeroes ``k >= rank``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):   # both branches of each where
        return _shrink_weight(evals, sigma_mppca, n_vox, n_vols, alignment)


def _shrink_weight(evals, sigma_mppca, n_vox, n_vols, alignment):
    gamma = n_vox / n_vols
    sw = np.sqrt(evals) / (sigma_mppca[:, None] * np.sqrt(n_vols))
    above = sw > 1 + np.sqrt(gamma)
    x = sw ** 2 - 1 - gamma
    ell = np.where(above, (x + np.sqrt(np.maximum(x ** 2 - 4 * gamma, 0.0))) / 2, 1.0)
    c2 = np.clip((1 - gamma / ell ** 2) / (1 + gamma / ell), 0.0, 1.0)
    ct2 = np.clip((1 - gamma / ell ** 2) / (1 + 1 / ell), 0.0, 1.0)
    t = np.sqrt(ell * c2 * ct2)
    if alignment is not None:
        s2 = 1 - c2
        valid = alignment > s2
        mu_tau = c2 / np.where(valid, alignment - s2, 1.0)
        t = np.where(valid, t / (c2 + s2 * mu_tau), t)
    return np.where(above, t / np.maximum(sw, 1e-12), 0.0)


def signal_alignment(whitened: np.ndarray, evecs: np.ndarray, evals: np.ndarray,
                     C: np.ndarray, levels: np.ndarray | None = None,
                     root: np.ndarray | None = None, r: int | None = None) -> np.ndarray:
    """How loud the noise is along each component, for "lr".

    Without ``levels``, for the center-voxel output: ``q_k = u_k^T C u_k / (tr C / n_vox)``.
    With them, for the whole patch: ``q_k = |D C^(1/2) u_k|^2 / mean(d^2 diag C)``, with
    ``D`` the patch's own levels.

    Args:
        whitened: (B, n_vox, n_vols) whitened patches.
        evecs: (B, n_vols, n_vols) eigenvectors of their Gram, columns descending.
        evals: (B, n_vols) the matching eigenvalues.
        C: (n_vox, n_vox) the patch correlation the patches were whitened with.
        levels: (B, n_vox) each patch voxel's level ``d``, or None.
        root: (n_vox, n_vox) ``C^(1/2)``; needed with ``levels``.
        r: compute only the first ``r`` components, the most any patch keeps; the rest
            get 1. Their weight is zero anyway, and at 7^3 and 297 volumes all of them
            cost twice the whitening and the Gram together.

    Returns:
        (B, n_vols). Above 1 where the component lies where the noise is loud.
    """
    n_vols = evecs.shape[2]
    if r is not None and r < n_vols:
        out = np.ones(evals.shape, np.result_type(evals.dtype, np.float32))
        out[:, :r] = signal_alignment(whitened, evecs[:, :, :r], evals[:, :r], C, levels,
                                      root)
        return out
    n_vox = whitened.shape[1]
    U = np.einsum("bpn,bnk->bpk", whitened, evecs)
    U = U / np.sqrt(np.maximum(evals, 1e-12))[:, None, :]
    if levels is None:
        q = np.einsum("bpk,bpk->bk", U.conj(), np.einsum("pq,bqk->bpk", C, U)).real
        return q / (np.trace(C) / n_vox)
    back = levels[:, :, None] * np.einsum("pq,bqk->bpk", root, U)        # D C^(1/2) u_k
    q = np.einsum("bpk,bpk->bk", back.conj(), back).real
    return q / np.mean(levels ** 2 * np.diag(C)[None, :], axis=1)[:, None]
