import numpy as np

from src.calibration.metrics import expected_calibration_error


def test_ece_perfect_calibration_zero_error():
    probabilities = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    labels = np.array([0, 1])

    ece = expected_calibration_error(
        probabilities,
        labels,
        n_bins=10,
    )

    assert np.isclose(ece, 0.0)


def test_ece_known_value():
    probabilities = np.array([
        [0.9, 0.1],
        [0.9, 0.1],
    ])

    labels = np.array([0, 1])

    ece = expected_calibration_error(
        probabilities,
        labels,
        n_bins=10,
    )

    expected = 0.4

    assert np.isclose(ece, expected)


def test_ece_rejects_mismatched_lengths():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    labels = np.array([0])

    try:
        expected_calibration_error(
            probabilities,
            labels,
        )
        assert False
    except ValueError:
        assert True


def test_ece_rejects_invalid_bin_count():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    labels = np.array([0])

    try:
        expected_calibration_error(
            probabilities,
            labels,
            n_bins=0,
        )
        assert False
    except ValueError:
        assert True