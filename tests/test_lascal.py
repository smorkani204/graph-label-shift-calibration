import numpy as np
import pytest

from src.calibration.lascal import (
    assign_equal_width_bins,
    compute_adaptive_bin_boundaries,
    estimate_classwise_l1_calibration_error,
    estimate_lascal_classwise_error,
    estimate_target_class_calibration_curve,
    estimate_target_class_probability_in_bin,
    one_hot_encode_labels,
)


def test_one_hot_encode_binary_labels():
    labels = np.array([0, 1, 0])

    encoded = one_hot_encode_labels(
        labels,
        n_classes=2,
    )

    expected = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    assert np.array_equal(
        encoded,
        expected,
    )


def test_one_hot_encode_multiclass_labels():
    labels = np.array([0, 2, 1])

    encoded = one_hot_encode_labels(
        labels,
        n_classes=3,
    )

    expected = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    assert np.array_equal(
        encoded,
        expected,
    )


def test_one_hot_rows_sum_to_one():
    labels = np.array([0, 1, 1, 0])

    encoded = one_hot_encode_labels(
        labels,
        n_classes=2,
    )

    assert np.allclose(
        encoded.sum(axis=1),
        1.0,
    )


def test_one_hot_rejects_invalid_class():
    labels = np.array([0, 2])

    with pytest.raises(ValueError):
        one_hot_encode_labels(
            labels,
            n_classes=2,
        )


def test_one_hot_rejects_nonpositive_number_of_classes():
    labels = np.array([0, 1])

    with pytest.raises(ValueError):
        one_hot_encode_labels(
            labels,
            n_classes=0,
        )


def test_equal_width_bin_assignment():
    values = np.array([
        0.05,
        0.25,
        0.55,
        0.85,
    ])

    bins = assign_equal_width_bins(
        values,
        n_bins=5,
    )

    expected = np.array([
        0,
        1,
        2,
        4,
    ])

    assert np.array_equal(
        bins,
        expected,
    )


def test_equal_width_bins_handle_boundaries():
    values = np.array([
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ])

    bins = assign_equal_width_bins(
        values,
        n_bins=5,
    )

    expected = np.array([
        0,
        1,
        2,
        3,
        4,
        4,
    ])

    assert np.array_equal(
        bins,
        expected,
    )


def test_equal_width_bins_reject_values_below_zero():
    values = np.array([
        -0.1,
        0.5,
    ])

    with pytest.raises(ValueError):
        assign_equal_width_bins(
            values,
            n_bins=5,
        )


def test_equal_width_bins_reject_values_above_one():
    values = np.array([
        0.5,
        1.1,
    ])

    with pytest.raises(ValueError):
        assign_equal_width_bins(
            values,
            n_bins=5,
        )


def test_equal_width_bins_reject_nonpositive_bins():
    values = np.array([
        0.2,
        0.8,
    ])

    with pytest.raises(ValueError):
        assign_equal_width_bins(
            values,
            n_bins=0,
        )


def test_estimate_target_class_probability_in_bin_known_value():
    source_probs = np.array([
        0.10,
        0.30,
        0.60,
        0.90,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.15,
        0.35,
        0.80,
        0.90,
    ])

    estimate = estimate_target_class_probability_in_bin(
        source_class_probabilities=source_probs,
        source_class_indicators=source_indicators,
        target_class_probabilities=target_probs,
        class_weight=1.0,
        bin_index=3,
        n_bins=4,
    )

    expected = 0.5

    assert np.isclose(
        estimate,
        expected,
    )


def test_estimate_target_class_probability_scales_with_weight():
    source_probs = np.array([
        0.10,
        0.30,
        0.60,
        0.90,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.15,
        0.35,
        0.80,
        0.90,
    ])

    estimate = estimate_target_class_probability_in_bin(
        source_class_probabilities=source_probs,
        source_class_indicators=source_indicators,
        target_class_probabilities=target_probs,
        class_weight=2.0,
        bin_index=3,
        n_bins=4,
    )

    expected = 1.0

    assert np.isclose(
        estimate,
        expected,
    )


def test_estimate_target_class_probability_rejects_empty_target_bin():
    source_probs = np.array([
        0.10,
        0.20,
    ])

    source_indicators = np.array([
        1.0,
        0.0,
    ])

    target_probs = np.array([
        0.10,
        0.20,
    ])

    with pytest.raises(ValueError):
        estimate_target_class_probability_in_bin(
            source_class_probabilities=source_probs,
            source_class_indicators=source_indicators,
            target_class_probabilities=target_probs,
            class_weight=1.0,
            bin_index=3,
            n_bins=4,
        )


def test_estimate_target_class_probability_rejects_length_mismatch():
    source_probs = np.array([
        0.10,
        0.20,
    ])

    source_indicators = np.array([
        1.0,
    ])

    target_probs = np.array([
        0.10,
        0.20,
    ])

    with pytest.raises(ValueError):
        estimate_target_class_probability_in_bin(
            source_class_probabilities=source_probs,
            source_class_indicators=source_indicators,
            target_class_probabilities=target_probs,
            class_weight=1.0,
            bin_index=0,
            n_bins=4,
        )

def test_estimate_target_class_calibration_curve_known_values():
    source_probs = np.array([
        0.10,
        0.30,
        0.60,
        0.90,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    (
        occupied_bins,
        mean_probabilities,
        estimated_frequencies,
    ) = estimate_target_class_calibration_curve(
        source_class_probabilities=source_probs,
        source_class_indicators=source_indicators,
        target_class_probabilities=target_probs,
        class_weight=1.0,
        n_bins=4,
    )

    expected_bins = np.array([
        0,
        3,
    ])

    expected_mean_probabilities = np.array([
        0.15,
        0.85,
    ])

    expected_frequencies = np.array([
        0.0,
        0.5,
    ])

    assert np.array_equal(
        occupied_bins,
        expected_bins,
    )

    assert np.allclose(
        mean_probabilities,
        expected_mean_probabilities,
    )

    assert np.allclose(
        estimated_frequencies,
        expected_frequencies,
    )

def test_estimate_classwise_l1_calibration_error_known_value():
    source_probs = np.array([
        0.10,
        0.30,
        0.60,
        0.90,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    calibration_error = (
        estimate_classwise_l1_calibration_error(
            source_class_probabilities=source_probs,
            source_class_indicators=source_indicators,
            target_class_probabilities=target_probs,
            class_weight=1.0,
            n_bins=4,
        )
    )

    expected = 0.25

    assert np.isclose(
        calibration_error,
        expected,
    )

def test_compute_adaptive_bin_boundaries_known_values():
    values = np.array([
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
    ])

    boundaries = compute_adaptive_bin_boundaries(
        values,
        n_bins=3,
    )

    expected = np.array([
        0.10,
        0.30,
        0.50,
        0.60,
    ])

    assert np.allclose(
        boundaries,
        expected,
    )

def test_estimate_lascal_classwise_error_known_value():
    source_probs = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.10,
        0.20,
        0.80,
        0.90,
    ])

    error = estimate_lascal_classwise_error(
        source_class_probabilities=source_probs,
        source_class_indicators=source_indicators,
        target_class_probabilities=target_probs,
        class_weight=1.0,
        p=2,
        n_bins=2,
        adaptive_bins=False,
    )

    expected = 0.225

    assert np.isclose(
        error,
        expected,
    )

def test_estimate_lascal_classwise_error_adaptive_is_finite():
    source_probs = np.array([
        0.05,
        0.10,
        0.20,
        0.35,
        0.55,
        0.70,
        0.85,
        0.95,
    ])

    source_indicators = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    target_probs = np.array([
        0.08,
        0.15,
        0.25,
        0.40,
        0.60,
        0.75,
        0.88,
        0.93,
    ])

    error = estimate_lascal_classwise_error(
        source_class_probabilities=source_probs,
        source_class_indicators=source_indicators,
        target_class_probabilities=target_probs,
        class_weight=1.0,
        p=2,
        n_bins=2,
        adaptive_bins=True,
    )

    assert np.isfinite(error)
    assert error >= 0.0