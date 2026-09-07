"""Synthetic data generation for controlled label-shift experiments."""

from __future__ import annotations

import numpy as np


def generate_gaussian_label_shift_data(
    n_samples: int,
    class_priors: tuple[float, float],
    n_features: int = 10,
    mean_separation: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a binary classification dataset with controllable class priors.

    The class-conditional feature distributions are:

        X | Y=0 ~ N(mu_0, I)
        X | Y=1 ~ N(mu_1, I)

    where:
        mu_0 = (-mean_separation, 0, ..., 0)
        mu_1 = ( mean_separation, 0, ..., 0)

    Changing ``class_priors`` while keeping the class-conditional
    distributions fixed creates pure label shift.

    Parameters
    ----------
    n_samples:
        Number of observations to generate.

    class_priors:
        Probabilities for classes 0 and 1.

    n_features:
        Number of input features.

    mean_separation:
        Controls how separable the two classes are.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    X:
        Feature matrix with shape (n_samples, n_features).

    y:
        Binary labels with shape (n_samples,).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    priors = np.asarray(class_priors, dtype=float)

    if priors.shape != (2,):
        raise ValueError("class_priors must contain exactly two probabilities.")

    if np.any(priors < 0):
        raise ValueError("class probabilities cannot be negative.")

    if not np.isclose(priors.sum(), 1.0):
        raise ValueError("class_priors must sum to 1.")

    rng = np.random.default_rng(seed)

    y = rng.choice(
        [0, 1],
        size=n_samples,
        p=priors,
    )

    mu_0 = np.zeros(n_features)
    mu_1 = np.zeros(n_features)

    mu_0[0] = -mean_separation
    mu_1[0] = mean_separation

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, n_features),
    )

    X[y == 0] += mu_0
    X[y == 1] += mu_1

    return X, y

def generate_source_target_label_shift(
    n_source: int = 2000,
    n_target: int = 2000,
    source_priors: tuple[float, float] = (0.5, 0.5),
    target_priors: tuple[float, float] = (0.7, 0.3),
    n_features: int = 10,
    mean_separation: float = 1.0,
    source_seed: int = 42,
    target_seed: int = 43,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate source and target datasets under pure label shift.

    Source and target share the same class-conditional feature
    distributions P(X | Y), while their class priors P(Y) may differ.
    """

    X_source, y_source = generate_gaussian_label_shift_data(
        n_samples=n_source,
        class_priors=source_priors,
        n_features=n_features,
        mean_separation=mean_separation,
        seed=source_seed,
    )

    X_target, y_target = generate_gaussian_label_shift_data(
        n_samples=n_target,
        class_priors=target_priors,
        n_features=n_features,
        mean_separation=mean_separation,
        seed=target_seed,
    )

    return X_source, y_source, X_target, y_target