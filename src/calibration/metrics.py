"""Calibration metrics for classification experiments."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute top-label Expected Calibration Error (ECE).

    This metric uses true labels and is therefore intended
    for researcher-side evaluation only.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels)

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape (n_samples, n_classes)."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape (n_samples,)."
        )

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            "probabilities and labels must contain the same number of samples."
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    correctness = (
        predictions == labels
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == 0:
            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            in_bin = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if not np.any(in_bin):
            continue

        bin_accuracy = np.mean(
            correctness[in_bin]
        )

        bin_confidence = np.mean(
            confidences[in_bin]
        )

        bin_weight = np.mean(in_bin)

        ece += (
            bin_weight
            * abs(bin_accuracy - bin_confidence)
        )

    return float(ece)