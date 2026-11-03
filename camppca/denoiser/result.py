"""What the denoisers return."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..noise_model import NoiseModel

__all__ = ["DenoiseResult"]


@dataclass
class DenoiseResult:
    """Denoised data plus the noise level the Marchenko-Pastur fit found.

    ``sigma_hat`` is an OUTPUT. It is what the fit measured, in the input's units.

    Voxels outside the mask are passed through unchanged. Both denoisers write back the
    centre voxel of each patch by default; ``aggregate="mean"`` averages the overlapping
    patch reconstructions instead (:mod:`camppca.denoiser.aggregate`).
    """

    denoised: np.ndarray
    """(X, Y, Z, N) denoised data."""

    sigma_hat: np.ndarray
    """(X, Y, Z) fitted noise level, zero outside the mask. With ``aggregate="mean"``, the
    root of the mean sigma_hat^2 over the patches covering each voxel."""

    rank: np.ndarray
    """(X, Y, Z) retained component count, zero outside the mask. With
    ``aggregate="mean"``, the mean rank over the covering patches, as float32."""

    model: NoiseModel | None = None
    """The noise model used, or None for standard MP-PCA, which does not use one."""

    provenance: dict = field(default_factory=dict)
    """What was run, for the record: patch, algorithm, and the rule applied."""
