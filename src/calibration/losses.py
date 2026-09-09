"""Calibration-related probabilistic losses."""

from __future__ import annotations

import numpy as np


def _validate_probabilities_and_labels(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate multiclass probabilities and integer labels.

    Parameters
    ----------
    probabilities:
        Array with shape (n_samples, n_classes).

    labels:
        Integer class labels with shape (n_samples,).

    Returns
    -------
    probabilities:
        Validated floating-point probability array.

    labels:
        Validated label array.
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

    if probabilities.shape[0] == 0:
        raise ValueError(
            "probabilities and labels cannot be empty."
        )

    if probabilities.shape[1] == 0:
        raise ValueError(
            "probabilities must contain at least one class."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            "probabilities must contain only finite values."
        )

    if np.any(probabilities < 0):
        raise ValueError(
            "probabilities cannot contain negative values."
        )

    if np.any(probabilities > 1):
        raise ValueError(
            "probabilities cannot contain values greater than one."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise ValueError(
            "each probability vector must sum to one."
        )

    if not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "labels must contain integer class indices."
        )

    if np.any(labels < 0) or np.any(
        labels >= probabilities.shape[1]
    ):
        raise ValueError(
            "labels contain an invalid class index."
        )

    return probabilities, labels


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

    probabilities, labels = (
        _validate_probabilities_and_labels(
            probabilities,
            labels,
        )
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
        np.log(
            true_class_probabilities
        )
    )

    return float(nll)


def brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the multiclass Brier score.

    The score is defined as

        (1 / N) * sum_i sum_k
        (p_ik - 1[y_i = k])^2

    where N is the number of samples, p_ik is the predicted
    probability for class k, and 1[y_i = k] is the one-hot
    representation of the true class.

    This implementation does not divide the classwise squared
    error by the number of classes.

    Lower values indicate better probabilistic predictions.
    A score of zero corresponds to perfect probabilistic
    predictions.

    Parameters
    ----------
    probabilities:
        Array with shape (n_samples, n_classes).

    labels:
        Integer class labels with shape (n_samples,).

    Returns
    -------
    score:
        Mean multiclass Brier score.
    """

    probabilities, labels = (
        _validate_probabilities_and_labels(
            probabilities,
            labels,
        )
    )

    n_samples, n_classes = (
        probabilities.shape
    )

    one_hot_labels = np.zeros(
        (
            n_samples,
            n_classes,
        ),
        dtype=float,
    )

    one_hot_labels[
        np.arange(n_samples),
        labels,
    ] = 1.0

    squared_errors = (
        probabilities
        - one_hot_labels
    ) ** 2

    score = np.mean(
        np.sum(
            squared_errors,
            axis=1,
        )
    )

    return float(score)