import numpy as np

from src.calibration.losses import negative_log_likelihood


def test_nll_perfect_predictions_is_near_zero():
    probabilities = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = np.array([0, 1])

    nll = negative_log_likelihood(
        probabilities,
        labels,
    )

    assert np.isclose(
        nll,
        0.0,
    )


def test_nll_known_value():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    labels = np.array([0, 1])

    nll = negative_log_likelihood(
        probabilities,
        labels,
    )

    expected = -np.mean(
        [
            np.log(0.8),
            np.log(0.7),
        ]
    )

    assert np.isclose(
        nll,
        expected,
    )


def test_nll_worse_predictions_have_higher_loss():
    good_probabilities = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    bad_probabilities = np.array([
        [0.6, 0.4],
        [0.4, 0.6],
    ])

    labels = np.array([0, 1])

    good_nll = negative_log_likelihood(
        good_probabilities,
        labels,
    )

    bad_nll = negative_log_likelihood(
        bad_probabilities,
        labels,
    )

    assert good_nll < bad_nll


def test_nll_rejects_mismatched_sample_counts():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    labels = np.array([0])

    try:
        negative_log_likelihood(
            probabilities,
            labels,
        )
        assert False
    except ValueError:
        assert True


def test_nll_rejects_invalid_class_label():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    labels = np.array([2])

    try:
        negative_log_likelihood(
            probabilities,
            labels,
        )
        assert False
    except ValueError:
        assert True