"""Calibration-related probabilistic losses."""

from __future__ import annotations

import numpy as np


def negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute multiclass negative log-likelihood.

    Parameters
    ----------
    probabilities:
        Array with shape (n_samples, n_classes).

    labels:
        Integer class labels with shape (n_samples,).

    Returns
    -------
    nll:
        Mean negative log-likelihood.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    labels = np.asarray(
        labels,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape "
            "(n_samples, n_classes)."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape (n_samples,)."
        )

    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            "probabilities and labels must contain "
            "the same number of samples."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "probabilities cannot contain negative values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise ValueError(
            "each probability vector must sum to one."
        )

    if np.any(labels < 0) or np.any(
        labels >= probabilities.shape[1]
    ):
        raise ValueError(
            "labels contain an invalid class index."
        )

    epsilon = np.finfo(float).eps

    true_class_probabilities = probabilities[
        np.arange(labels.shape[0]),
        labels,
    ]

    true_class_probabilities = np.clip(
        true_class_probabilities,
        epsilon,
        1.0,
    )

    nll = -np.mean(
        np.log(true_class_probabilities)
    )

    return float(nll)