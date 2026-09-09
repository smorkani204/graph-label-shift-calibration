"""LaSCal-compatible label-free calibration error estimation."""

from __future__ import annotations

import numpy as np


def one_hot_encode_labels(
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoded labels.

    Parameters
    ----------
    labels:
        Integer class labels with shape (n_samples,).

    n_classes:
        Total number of classes.

    Returns
    -------
    one_hot:
        One-hot encoded label matrix with shape
        (n_samples, n_classes).
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(
            "labels must have shape (n_samples,)."
        )

    if n_classes <= 0:
        raise ValueError(
            "n_classes must be positive."
        )

    if np.any(labels < 0) or np.any(
        labels >= n_classes
    ):
        raise ValueError(
            "labels contain an invalid class index."
        )

    one_hot = np.zeros(
        (labels.shape[0], n_classes),
        dtype=float,
    )

    one_hot[
        np.arange(labels.shape[0]),
        labels,
    ] = 1.0

    return one_hot


def assign_equal_width_bins(
    values: np.ndarray,
    n_bins: int = 15,
) -> np.ndarray:
    """
    Assign values in [0, 1] to equal-width bins.

    Parameters
    ----------
    values:
        One-dimensional array of probabilities or confidence values.

    n_bins:
        Number of equal-width bins.

    Returns
    -------
    bin_indices:
        Integer bin index for each value, ranging from
        0 to n_bins - 1.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must have shape (n_samples,)."
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            "values must lie in the interval [0, 1]."
        )

    bin_indices = np.floor(
        values * n_bins
    ).astype(int)

    bin_indices = np.clip(
        bin_indices,
        0,
        n_bins - 1,
    )

    return bin_indices


def estimate_target_class_probability_in_bin(
    source_class_probabilities: np.ndarray,
    source_class_indicators: np.ndarray,
    target_class_probabilities: np.ndarray,
    class_weight: float,
    bin_index: int,
    n_bins: int = 15,
) -> float:
    """
    Estimate the target conditional class probability inside one bin.

    This implements the simple equal-width binning-kernel version
    of the importance-weighted label-free estimator for one class.

    This helper is retained as a transparent validation
    implementation and is separate from the official-style
    LaSCal estimator below.
    """
    source_class_probabilities = np.asarray(
        source_class_probabilities,
        dtype=float,
    )

    source_class_indicators = np.asarray(
        source_class_indicators,
        dtype=float,
    )

    target_class_probabilities = np.asarray(
        target_class_probabilities,
        dtype=float,
    )

    if source_class_probabilities.ndim != 1:
        raise ValueError(
            "source_class_probabilities must be one-dimensional."
        )

    if source_class_indicators.ndim != 1:
        raise ValueError(
            "source_class_indicators must be one-dimensional."
        )

    if target_class_probabilities.ndim != 1:
        raise ValueError(
            "target_class_probabilities must be one-dimensional."
        )

    if (
        source_class_probabilities.shape[0]
        != source_class_indicators.shape[0]
    ):
        raise ValueError(
            "source probabilities and indicators must contain "
            "the same number of samples."
        )

    if class_weight < 0:
        raise ValueError(
            "class_weight cannot be negative."
        )

    if bin_index < 0 or bin_index >= n_bins:
        raise ValueError(
            "bin_index is outside the valid range."
        )

    source_bins = assign_equal_width_bins(
        source_class_probabilities,
        n_bins=n_bins,
    )

    target_bins = assign_equal_width_bins(
        target_class_probabilities,
        n_bins=n_bins,
    )

    source_mask = source_bins == bin_index
    target_mask = target_bins == bin_index

    target_bin_count = np.sum(
        target_mask
    )

    if target_bin_count == 0:
        raise ValueError(
            "cannot estimate calibration for an empty target bin."
        )

    source_numerator = (
        class_weight
        * np.sum(
            source_class_indicators[source_mask]
        )
        / source_class_indicators.shape[0]
    )

    target_denominator = (
        target_bin_count
        / target_class_probabilities.shape[0]
    )

    estimate = (
        source_numerator
        / target_denominator
    )

    return float(estimate)


def estimate_target_class_calibration_curve(
    source_class_probabilities: np.ndarray,
    source_class_indicators: np.ndarray,
    target_class_probabilities: np.ndarray,
    class_weight: float,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the target class-wise calibration curve across
    occupied equal-width bins.

    This function belongs to the simple validation
    implementation rather than the exact-reference path.
    """
    target_class_probabilities = np.asarray(
        target_class_probabilities,
        dtype=float,
    )

    if target_class_probabilities.ndim != 1:
        raise ValueError(
            "target_class_probabilities must be one-dimensional."
        )

    target_bins = assign_equal_width_bins(
        target_class_probabilities,
        n_bins=n_bins,
    )

    occupied_bins = np.unique(
        target_bins
    )

    mean_target_probabilities = []
    estimated_target_class_frequencies = []

    for bin_index in occupied_bins:
        target_mask = (
            target_bins == bin_index
        )

        mean_probability = np.mean(
            target_class_probabilities[
                target_mask
            ]
        )

        estimated_frequency = (
            estimate_target_class_probability_in_bin(
                source_class_probabilities=(
                    source_class_probabilities
                ),
                source_class_indicators=(
                    source_class_indicators
                ),
                target_class_probabilities=(
                    target_class_probabilities
                ),
                class_weight=class_weight,
                bin_index=int(bin_index),
                n_bins=n_bins,
            )
        )

        mean_target_probabilities.append(
            mean_probability
        )

        estimated_target_class_frequencies.append(
            estimated_frequency
        )

    return (
        occupied_bins.astype(int),
        np.asarray(
            mean_target_probabilities,
            dtype=float,
        ),
        np.asarray(
            estimated_target_class_frequencies,
            dtype=float,
        ),
    )


def estimate_classwise_l1_calibration_error(
    source_class_probabilities: np.ndarray,
    source_class_indicators: np.ndarray,
    target_class_probabilities: np.ndarray,
    class_weight: float,
    n_bins: int = 15,
) -> float:
    """
    Estimate the simple label-free L1 calibration error
    for one class.

    This is a transparent equal-width-bin validation
    implementation. It is not the exact official LaSCal
    estimator used below.
    """
    target_class_probabilities = np.asarray(
        target_class_probabilities,
        dtype=float,
    )

    if target_class_probabilities.ndim != 1:
        raise ValueError(
            "target_class_probabilities must be one-dimensional."
        )

    if target_class_probabilities.shape[0] == 0:
        raise ValueError(
            "target_class_probabilities cannot be empty."
        )

    (
        occupied_bins,
        mean_target_probabilities,
        estimated_target_class_frequencies,
    ) = estimate_target_class_calibration_curve(
        source_class_probabilities=(
            source_class_probabilities
        ),
        source_class_indicators=(
            source_class_indicators
        ),
        target_class_probabilities=(
            target_class_probabilities
        ),
        class_weight=class_weight,
        n_bins=n_bins,
    )

    target_bins = assign_equal_width_bins(
        target_class_probabilities,
        n_bins=n_bins,
    )

    calibration_error = 0.0

    for position, bin_index in enumerate(
        occupied_bins
    ):
        target_bin_count = np.sum(
            target_bins == bin_index
        )

        target_bin_mass = (
            target_bin_count
            / target_class_probabilities.shape[0]
        )

        calibration_gap = abs(
            estimated_target_class_frequencies[position]
            - mean_target_probabilities[position]
        )

        calibration_error += (
            target_bin_mass
            * calibration_gap
        )

    return float(calibration_error)


def compute_adaptive_bin_boundaries(
    values: np.ndarray,
    n_bins: int = 15,
) -> np.ndarray:
    """
    Compute approximately equal-count adaptive bin boundaries.

    This mirrors the boundary construction used by the
    official LaSCal implementation.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.shape[0] == 0:
        raise ValueError(
            "values cannot be empty."
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(
            "values must lie in the interval [0, 1]."
        )

    n_points = values.shape[0]

    sorted_values = np.sort(
        values
    )

    boundaries = np.interp(
        np.linspace(
            0,
            n_points,
            n_bins + 1,
        ),
        np.arange(n_points),
        sorted_values,
    )

    return boundaries


def estimate_lascal_classwise_error(
    source_class_probabilities: np.ndarray,
    source_class_indicators: np.ndarray,
    target_class_probabilities: np.ndarray,
    class_weight: float,
    p: int = 2,
    n_bins: int = 15,
    adaptive_bins: bool = True,
) -> float:
    """
    Estimate the official-style LaSCal class-wise
    calibration error.

    This follows the structure of the reference
    EceLabelShift.get_ece implementation:

    - class-wise probabilities;
    - target-based adaptive binning when requested;
    - intervals of the form (lower, upper];
    - importance weighting under label shift;
    - leave-one-out target denominator;
    - bins with at most one target point are skipped;
    - per-target-point |z - eta_hat|^p contributions;
    - division by the total target sample size.

    The returned value is the mean p-th-power
    calibration error. No p-th root is applied.
    """
    source_class_probabilities = np.asarray(
        source_class_probabilities,
        dtype=float,
    )

    source_class_indicators = np.asarray(
        source_class_indicators,
        dtype=float,
    )

    target_class_probabilities = np.asarray(
        target_class_probabilities,
        dtype=float,
    )

    if source_class_probabilities.ndim != 1:
        raise ValueError(
            "source_class_probabilities must be one-dimensional."
        )

    if source_class_indicators.ndim != 1:
        raise ValueError(
            "source_class_indicators must be one-dimensional."
        )

    if target_class_probabilities.ndim != 1:
        raise ValueError(
            "target_class_probabilities must be one-dimensional."
        )

    if (
        source_class_probabilities.shape[0]
        != source_class_indicators.shape[0]
    ):
        raise ValueError(
            "source probabilities and indicators must contain "
            "the same number of samples."
        )

    if source_class_probabilities.shape[0] == 0:
        raise ValueError(
            "source data cannot be empty."
        )

    if target_class_probabilities.shape[0] == 0:
        raise ValueError(
            "target data cannot be empty."
        )

    if class_weight < 0:
        raise ValueError(
            "class_weight cannot be negative."
        )

    if p <= 0:
        raise ValueError(
            "p must be positive."
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    n_source = (
        source_class_probabilities.shape[0]
    )

    n_target = (
        target_class_probabilities.shape[0]
    )

    if adaptive_bins:
        boundaries = compute_adaptive_bin_boundaries(
            target_class_probabilities,
            n_bins=n_bins,
        )
    else:
        boundaries = np.linspace(
            0.0,
            1.0,
            n_bins + 1,
        )

    total_error = 0.0

    for bin_index in range(n_bins):
        lower = boundaries[bin_index]
        upper = boundaries[bin_index + 1]

        # Match the official LaSCal implementation exactly:
        # confidence > lower and confidence <= upper.
        source_mask = (
            (source_class_probabilities > lower)
            & (source_class_probabilities <= upper)
        )

        target_mask = (
            (target_class_probabilities > lower)
            & (target_class_probabilities <= upper)
        )

        target_bin_count = int(
            np.sum(target_mask)
        )

        # The official estimator evaluates only bins
        # containing more than one target observation.
        if target_bin_count <= 1:
            continue

        weighted_source_mass = (
            class_weight
            * np.sum(
                source_class_indicators[
                    source_mask
                ]
            )
        )

        normalizer = (
            (n_target - 1)
            / n_source
        )

        estimated_frequency = (
            normalizer
            * weighted_source_mass
            / (target_bin_count - 1)
        )

        target_values = (
            target_class_probabilities[
                target_mask
            ]
        )

        total_error += np.sum(
            np.abs(
                target_values
                - estimated_frequency
            )
            ** p
        )

    return float(
        total_error / n_target
    )