"""Shrinkage of the kept components: an opt-in alternative to the hard threshold.

The hard threshold keeps every component above the MP edge at full weight. A shrinker
weights each one instead, by how much of its singular value is signal. Both run on the
whitened patch at the patch's own ``sigma_hat``, and both keep the MP rank: a component
the rule called noise was counted in ``sigma_hat`` as noise, so it stays dropped.

``"gd"``  Gavish & Donoho's Frobenius-optimal shrinker. Optimal for the error in the
          WHITENED patch.
``"lr"``  Leeb & Romanov's correction of it (arXiv 1811.02201, Algorithm 1). Unwhitening
          stretches a component's true part by how loud the noise is along that signal
          direction, and its noise part by the average noise loudness, so the error that
          matters, in data units, wants a different weight. With white noise the two agree.

For a patch of ``n_vox`` voxels and ``n_vols`` volumes, ``gamma = n_vox / n_vols``, and
``sw_k = s_k / (sigma_hat sqrt(n_vols))`` the k-th whitened singular value in units of the
noise. Above the edge ``1 + sqrt(gamma)``::

    l_k    = [sw^2 - 1 - gamma + sqrt((sw^2 - 1 - gamma)^2 - 4 gamma)] / 2
    c_k^2  = (1 - gamma / l_k^2) / (1 + gamma / l_k)       voxel-side cosine
    ct_k^2 = (1 - gamma / l_k^2) / (1 + 1 / l_k)           volume-side cosine
    t_k    = sqrt(l_k) c_k ct_k                            "gd"
    t_k    = sqrt(l_k) c_k ct_k / (c_k^2 + s_k^2 mu_tau_k)  "lr"

with ``s_k^2 = 1 - c_k^2``, ``mu_tau_k = c_k^2 / (q_k - s_k^2)`` and
``q_k = u_k^T C u_k / (tr C / n_vox)``, ``u_k`` the k-th left singular vector of the
whitened patch. Where ``q_k <= s_k^2`` there is no valid ``tau_k`` and "lr" falls back to
"gd". The weight on the component is ``t_k / sw_k``, and 0 below the edge.

``q_k`` measures noise loudness with ``C`` alone. With ``noise_field=True`` the rows were
divided by ``d`` first, so the level variation inside the patch is not part of it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["RULES", "SHRINKERS", "check_shrink", "shrink_weight", "signal_alignment"]

SHRINKERS = ("hard", "gd", "lr")

#: How each shrinker reads in a result's provenance.
RULES = {"hard": "hard threshold, no shrinkage",
         "gd": "MP rank, Gavish-Donoho shrinkage",
         "lr": "MP rank, Leeb-Romanov shrinkage"}


def check_shrink(shrink: str) -> str:
    """Validate a shrinker name; returns it.

    Raises:
        ValueError: if ``shrink`` is not one of :data:`SHRINKERS`.
    """
    if shrink not in SHRINKERS:
        raise ValueError(f"shrink must be one of {SHRINKERS}, got {shrink!r}")
    return shrink


def shrink_weight(evals: jax.Array, sigma_hat: jax.Array, n_vox: int, n_vols: int,
                  alignment: jax.Array | None = None) -> jax.Array:
    """Per component, the shrunk singular value over the observed one.

    Args:
        evals: (B, n_vols) descending eigenvalues of the whitened Gram.
        sigma_hat: (B,) the patch's MP noise level.
        n_vox: voxels in the patch.
        n_vols: volumes in the patch.
        alignment: (B, n_vols) ``q_k`` from :func:`signal_alignment` for "lr"; None
            for "gd".

    Returns:
        (B, n_vols) weights, 0 below the edge. The caller zeroes ``k >= rank``.
    """
    gamma = n_vox / n_vols
    sw = jnp.sqrt(evals) / (sigma_hat[:, None] * np.sqrt(n_vols))
    above = sw > 1 + np.sqrt(gamma)
    x = sw ** 2 - 1 - gamma
    ell = jnp.where(above, (x + jnp.sqrt(jnp.maximum(x ** 2 - 4 * gamma, 0.0))) / 2, 1.0)
    c2 = jnp.clip((1 - gamma / ell ** 2) / (1 + gamma / ell), 0.0, 1.0)
    ct2 = jnp.clip((1 - gamma / ell ** 2) / (1 + 1 / ell), 0.0, 1.0)
    t = jnp.sqrt(ell * c2 * ct2)
    if alignment is not None:
        s2 = 1 - c2
        valid = alignment > s2
        mu_tau = c2 / jnp.where(valid, alignment - s2, 1.0)
        t = jnp.where(valid, t / (c2 + s2 * mu_tau), t)
    return jnp.where(above, t / jnp.maximum(sw, 1e-12), 0.0)


def signal_alignment(whitened: jax.Array, evecs: jax.Array, evals: jax.Array,
                     C: jax.Array) -> jax.Array:
    """``q_k = u_k^T C u_k / (tr C / n_vox)``: how loud the noise is along each component.

    Args:
        whitened: (B, n_vox, n_vols) whitened patches.
        evecs: (B, n_vols, n_vols) eigenvectors of their Gram, columns descending.
        evals: (B, n_vols) the matching eigenvalues.
        C: (n_vox, n_vox) the patch correlation the patches were whitened with.

    Returns:
        (B, n_vols). Above 1 where the component lies where the noise is loud.
    """
    n_vox = whitened.shape[1]
    U = jnp.einsum("bpn,bnk->bpk", whitened, evecs)
    U = U / jnp.sqrt(jnp.maximum(evals, 1e-12))[:, None, :]
    q = jnp.einsum("bpk,bpk->bk", U, jnp.einsum("pq,bqk->bpk", C, U))
    return q / (jnp.trace(C) / n_vox)
