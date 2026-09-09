"""Tests for the multiclass Brier score."""

import numpy as np
import pytest

from src.calibration.losses import brier_score


def test_brier_score_perfect_predictions_is_zero():
    """Perfect deterministic predictions must have Brier score zero."""

    probabilities = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 1, 0, 1],
        dtype=int,
    )

    score = brier_score(
        probabilities,
        labels,
    )

    assert np.isclose(
        score,
        0.0,
    )


def test_brier_score_known_binary_value():
    """
    Check the score against a hand-computable binary example.

    For:
        p1 = [0.8, 0.2], y1 = 0
        p2 = [0.3, 0.7], y2 = 1

    squared errors are:
        (0.8 - 1)^2 + (0.2 - 0)^2 = 0.08
        (0.3 - 0)^2 + (0.7 - 1)^2 = 0.18

    mean multiclass Brier score:
        (0.08 + 0.18) / 2 = 0.13
    """

    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 1],
        dtype=int,
    )

    score = brier_score(
        probabilities,
        labels,
    )

    assert np.isclose(
        score,
        0.13,
    )


def test_brier_score_known_multiclass_value():
    """
    Check the definition explicitly in a three-class setting.

    Sample 1:
        p = [0.7, 0.2, 0.1], y = 0

        squared error =
        (0.7 - 1)^2 + 0.2^2 + 0.1^2
        = 0.14

    Sample 2:
        p = [0.1, 0.3, 0.6], y = 2

        squared error =
        0.1^2 + 0.3^2 + (0.6 - 1)^2
        = 0.26

    mean =
        (0.14 + 0.26) / 2
        = 0.20
    """

    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 2],
        dtype=int,
    )

    score = brier_score(
        probabilities,
        labels,
    )

    assert np.isclose(
        score,
        0.20,
    )


def test_brier_score_better_predictions_have_lower_score():
    """Better probabilistic predictions should receive a lower score."""

    labels = np.asarray(
        [0, 1, 0, 1],
        dtype=int,
    )

    good_probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=float,
    )

    poor_probabilities = np.asarray(
        [
            [0.55, 0.45],
            [0.45, 0.55],
            [0.55, 0.45],
            [0.45, 0.55],
        ],
        dtype=float,
    )

    good_score = brier_score(
        good_probabilities,
        labels,
    )

    poor_score = brier_score(
        poor_probabilities,
        labels,
    )

    assert good_score < poor_score


def test_brier_score_worst_binary_deterministic_prediction_is_two():
    """
    Under our unnormalized multiclass definition, a completely wrong
    deterministic binary prediction has score 2.
    """

    probabilities = np.asarray(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 1],
        dtype=int,
    )

    score = brier_score(
        probabilities,
        labels,
    )

    assert np.isclose(
        score,
        2.0,
    )


def test_brier_score_rejects_mismatched_sample_counts():
    """Probability rows and labels must have matching sample counts."""

    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0],
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        brier_score(
            probabilities,
            labels,
        )


def test_brier_score_rejects_probabilities_that_do_not_sum_to_one():
    """Each probability vector must define a valid distribution."""

    probabilities = np.asarray(
        [
            [0.8, 0.4],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 1],
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        brier_score(
            probabilities,
            labels,
        )


def test_brier_score_rejects_negative_probabilities():
    """Negative probabilities must be rejected."""

    probabilities = np.asarray(
        [
            [1.1, -0.1],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 1],
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        brier_score(
            probabilities,
            labels,
        )


def test_brier_score_rejects_invalid_class_label():
    """Labels outside the available class range must be rejected."""

    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0, 2],
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="invalid class index",
    ):
        brier_score(
            probabilities,
            labels,
        )


def test_brier_score_rejects_non_integer_labels():
    """Class labels must be represented as integer class indices."""

    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=float,
    )

    labels = np.asarray(
        [0.0, 1.0],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="integer class indices",
    ):
        brier_score(
            probabilities,
            labels,
        )


def test_brier_score_rejects_empty_input():
    """The metric must not silently return NaN for an empty dataset."""

    probabilities = np.empty(
        (0, 2),
        dtype=float,
    )

    labels = np.empty(
        (0,),
        dtype=int,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        brier_score(
            probabilities,
            labels,
        )