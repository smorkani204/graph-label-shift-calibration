"""Black Box Shift Estimation (BBSE) for label shift."""

from __future__ import annotations

import numpy as np


def estimate_target_prediction_distribution(
    predicted_labels: np.ndarray,
    n_classes: int = 2,
) -> np.ndarray:
    """
    Estimate the target distribution of predicted labels.

    Parameters
    ----------
    predicted_labels:
        Hard classifier predictions on unlabeled target samples.

    n_classes:
        Number of classes.

    Returns
    -------
    mu_hat:
        Estimated target predicted-label distribution with shape
        (n_classes,).
    """
    counts = np.bincount(
        predicted_labels,
        minlength=n_classes,
    ).astype(float)

    mu_hat = counts / counts.sum()

    return mu_hat


def estimate_target_priors_bbse(
    confusion_matrix: np.ndarray,
    target_prediction_distribution: np.ndarray,
) -> np.ndarray:
    """
    Estimate target class priors using BBSE.

    Solves

        C q_t = mu_t

    where

        C[i, j] = P(hat{Y}=i | Y=j)
        q_t[j]  = P_t(Y=j)
        mu_t[i] = P_t(hat{Y}=i)

    Parameters
    ----------
    confusion_matrix:
        BBSE-oriented source confusion matrix with shape
        (n_classes, n_classes).

        Rows correspond to predicted classes.
        Columns correspond to true classes.

    target_prediction_distribution:
        Estimated distribution of predicted labels on the
        unlabeled target data.

    Returns
    -------
    q_hat:
        Estimated target class-prior vector.
    """
    q_hat = np.linalg.solve(
        confusion_matrix,
        target_prediction_distribution,
    )

    return q_hat


def is_valid_probability_vector(
    probabilities: np.ndarray,
    atol: float = 1e-8,
) -> bool:
    """
    Check whether a vector is a valid probability distribution.

    A valid probability vector must:

    - contain only finite values,
    - have no negative entries,
    - have no entries greater than 1,
    - sum to 1 within numerical tolerance.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if not np.all(np.isfinite(probabilities)):
        return False

    if np.any(probabilities < -atol):
        return False

    if np.any(probabilities > 1.0 + atol):
        return False

    if not np.isclose(probabilities.sum(), 1.0, atol=atol):
        return False

    return True

def compute_importance_weights(
    target_priors: np.ndarray,
    source_priors: np.ndarray,
) -> np.ndarray:
    """
    Compute label-shift importance weights.

    For each class y,

        w(y) = P_t(Y=y) / P_s(Y=y)

    Parameters
    ----------
    target_priors:
        Estimated target class-prior probabilities.

    source_priors:
        Source class-prior probabilities.

    Returns
    -------
    weights:
        Class-specific importance weights.
    """
    target_priors = np.asarray(target_priors, dtype=float)
    source_priors = np.asarray(source_priors, dtype=float)

    if target_priors.shape != source_priors.shape:
        raise ValueError(
            "target_priors and source_priors must have the same shape."
        )

    if np.any(source_priors <= 0):
        raise ValueError(
            "source_priors must be strictly positive."
        )

    weights = target_priors / source_priors

    return weights