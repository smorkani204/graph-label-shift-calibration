"""Utilities for controlled probability miscalibration."""

from __future__ import annotations

import numpy as np


def apply_temperature_to_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling directly to class probabilities.

    Parameters
    ----------
    probabilities:
        Array of class probabilities with shape
        (n_samples, n_classes).

    temperature:
        Positive temperature value.

        temperature < 1 sharpens probabilities.
        temperature = 1 leaves probabilities unchanged.
        temperature > 1 softens probabilities.

    Returns
    -------
    scaled_probabilities:
        Temperature-scaled probability vectors whose rows
        sum to one.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes)."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be strictly positive."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "probabilities cannot contain negative values."
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(row_sums, 1.0):
        raise ValueError(
            "each probability vector must sum to one."
        )

    epsilon = np.finfo(float).eps

    clipped = np.clip(
        probabilities,
        epsilon,
        1.0,
    )

    logits = np.log(clipped)

    scaled_logits = logits / temperature

    scaled_logits -= np.max(
        scaled_logits,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(scaled_logits)

    scaled_probabilities = (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )

    return scaled_probabilities