"""Simple non-graph baseline classifier for label-shift experiments."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic-regression classifier on labeled source data.

    Parameters
    ----------
    X_train:
        Source training features with shape (n_samples, n_features).

    y_train:
        Source training labels with shape (n_samples,).

    random_state:
        Random seed for reproducibility.

    Returns
    -------
    model:
        Fitted scikit-learn LogisticRegression classifier.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model


def predict_labels_and_probabilities(
    model: LogisticRegression,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate hard class predictions and class probabilities.

    Returns
    -------
    predictions:
        Predicted class labels with shape (n_samples,).

    probabilities:
        Predicted class probabilities with shape (n_samples, 2).
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return predictions, probabilities