import numpy as np

from src.label_shift.bbse import (
    estimate_target_prediction_distribution,
    estimate_target_priors_bbse,
    is_valid_probability_vector,
    compute_importance_weights,
)


def test_target_prediction_distribution():
    predictions = np.array([0, 0, 0, 1])

    mu_hat = estimate_target_prediction_distribution(
        predictions,
        n_classes=2,
    )

    expected = np.array([0.75, 0.25])

    assert np.allclose(mu_hat, expected)


def test_bbse_with_perfect_classifier():
    C = np.eye(2)

    mu_hat = np.array([0.7, 0.3])

    q_hat = estimate_target_priors_bbse(
        C,
        mu_hat,
    )

    assert np.allclose(
        q_hat,
        np.array([0.7, 0.3]),
    )


def test_bbse_recovers_known_priors():
    C = np.array([
        [0.8, 0.2],
        [0.2, 0.8],
    ])

    q_true = np.array([0.7, 0.3])

    mu = C @ q_true

    q_hat = estimate_target_priors_bbse(
        C,
        mu,
    )

    assert np.allclose(q_hat, q_true)


def test_bbse_solution_satisfies_linear_system():
    C = np.array([
        [0.85, 0.15],
        [0.15, 0.85],
    ])

    mu_hat = np.array([0.64, 0.36])

    q_hat = estimate_target_priors_bbse(
        C,
        mu_hat,
    )

    assert np.allclose(
        C @ q_hat,
        mu_hat,
    )


def test_valid_probability_vector():
    probabilities = np.array([0.7, 0.3])

    assert is_valid_probability_vector(probabilities)


def test_invalid_probability_vector():
    probabilities = np.array([1.05, -0.05])

    assert not is_valid_probability_vector(probabilities)

def test_compute_importance_weights():
    target_priors = np.array([0.7, 0.3])
    source_priors = np.array([0.5, 0.5])

    weights = compute_importance_weights(
        target_priors,
        source_priors,
    )

    expected = np.array([1.4, 0.6])

    assert np.allclose(weights, expected)


def test_importance_weights_shape_mismatch():
    target_priors = np.array([0.7, 0.3])
    source_priors = np.array([0.5, 0.3, 0.2])

    try:
        compute_importance_weights(
            target_priors,
            source_priors,
        )
        assert False
    except ValueError:
        assert True


def test_importance_weights_zero_source_prior():
    target_priors = np.array([0.7, 0.3])
    source_priors = np.array([1.0, 0.0])

    try:
        compute_importance_weights(
            target_priors,
            source_priors,
        )
        assert False
    except ValueError:
        assert True