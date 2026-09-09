import numpy as np

from src.calibration.temperature import (
    apply_temperature_to_probabilities,
)


def test_temperature_one_preserves_probabilities():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    scaled = apply_temperature_to_probabilities(
        probabilities,
        temperature=1.0,
    )

    assert np.allclose(
        scaled,
        probabilities,
    )


def test_lower_temperature_sharpens_probabilities():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    scaled = apply_temperature_to_probabilities(
        probabilities,
        temperature=0.5,
    )

    assert scaled[0, 0] > probabilities[0, 0]
    assert scaled[0, 1] < probabilities[0, 1]


def test_higher_temperature_softens_probabilities():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    scaled = apply_temperature_to_probabilities(
        probabilities,
        temperature=2.0,
    )

    assert scaled[0, 0] < probabilities[0, 0]
    assert scaled[0, 0] > 0.5


def test_scaled_probabilities_sum_to_one():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
    ])

    scaled = apply_temperature_to_probabilities(
        probabilities,
        temperature=0.5,
    )

    assert np.allclose(
        scaled.sum(axis=1),
        1.0,
    )


def test_temperature_preserves_predicted_class():
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    original_predictions = np.argmax(
        probabilities,
        axis=1,
    )

    scaled = apply_temperature_to_probabilities(
        probabilities,
        temperature=2.0,
    )

    scaled_predictions = np.argmax(
        scaled,
        axis=1,
    )

    assert np.array_equal(
        original_predictions,
        scaled_predictions,
    )


def test_rejects_nonpositive_temperature():
    probabilities = np.array([
        [0.8, 0.2],
    ])

    try:
        apply_temperature_to_probabilities(
            probabilities,
            temperature=0.0,
        )
        assert False
    except ValueError:
        assert True