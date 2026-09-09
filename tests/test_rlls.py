import numpy as np
import pytest

from src.label_shift.rlls import compute_rho


def test_compute_rho_matches_reference_formula():
    n_class = 2
    n_train = 400
    delta = 0.05

    expected = 3 * (
        2 * np.log(2 * n_class / delta) / (3 * n_train)
        + np.sqrt(2 * np.log(2 * n_class / delta) / n_train)
    )

    result = compute_rho(
        n_class=n_class,
        n_train=n_train,
        delta=delta,
    )

    assert np.isclose(result, expected)


def test_compute_rho_is_positive():
    assert compute_rho(n_class=2, n_train=400) > 0


@pytest.mark.parametrize(
    "n_class,n_train,delta",
    [
        (0, 400, 0.05),
        (2, 0, 0.05),
        (2, 400, 0.0),
        (2, 400, 1.0),
    ],
)
def test_compute_rho_rejects_invalid_inputs(n_class, n_train, delta):
    with pytest.raises(ValueError):
        compute_rho(n_class, n_train, delta)


from src.label_shift.rlls import compute_rlls_weights


def test_rlls_no_shift_returns_unit_weights():
    C_yy = np.array([
        [0.45, 0.05],
        [0.05, 0.45],
    ])

    mu_train_y = np.array([0.50, 0.50])
    mu_y = np.array([0.50, 0.50])

    weights = compute_rlls_weights(
        C_yy=C_yy,
        mu_y=mu_y,
        mu_train_y=mu_train_y,
        rho=0.01,
    )

    assert np.allclose(weights, np.ones(2), atol=1e-6)


def test_rlls_weights_are_nonnegative():
    C_yy = np.array([
        [0.45, 0.05],
        [0.05, 0.45],
    ])

    mu_train_y = np.array([0.50, 0.50])
    mu_y = np.array([0.65, 0.35])

    weights = compute_rlls_weights(
        C_yy=C_yy,
        mu_y=mu_y,
        mu_train_y=mu_train_y,
        rho=0.01,
    )

    assert np.all(weights >= -1e-8)


from src.label_shift.rlls import compute_hard_rlls_moments


def test_hard_rlls_moments_known_example():
    source_probabilities = np.array([
        [0.9, 0.1],  # pred 0, true 0
        [0.8, 0.2],  # pred 0, true 1
        [0.3, 0.7],  # pred 1, true 1
        [0.2, 0.8],  # pred 1, true 1
    ])

    source_labels = np.array([0, 1, 1, 1])

    target_probabilities = np.array([
        [0.9, 0.1],  # pred 0
        [0.7, 0.3],  # pred 0
        [0.6, 0.4],  # pred 0
        [0.2, 0.8],  # pred 1
    ])

    C_yy, mu_target, mu_source = compute_hard_rlls_moments(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        target_probabilities=target_probabilities,
    )

    expected_C_yy = np.array([
        [0.25, 0.25],
        [0.00, 0.50],
    ])

    expected_mu_source = np.array([0.50, 0.50])
    expected_mu_target = np.array([0.75, 0.25])

    assert np.allclose(C_yy, expected_C_yy)
    assert np.allclose(mu_source, expected_mu_source)
    assert np.allclose(mu_target, expected_mu_target)


from src.label_shift.rlls import estimate_rlls_hard_weights


def test_estimate_rlls_hard_weights_no_shift():
    source_probabilities = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])

    source_labels = np.array([0, 0, 1, 1])

    target_probabilities = np.array([
        [0.85, 0.15],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.15, 0.85],
    ])

    weights = estimate_rlls_hard_weights(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        target_probabilities=target_probabilities,
    )

    assert np.allclose(weights, np.ones(2), atol=1e-6)
