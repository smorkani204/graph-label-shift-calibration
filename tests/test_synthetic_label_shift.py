import numpy as np
import pytest

from src.data.synthetic_label_shift import (
    generate_gaussian_label_shift_data,
    generate_source_target_label_shift,
)


def test_output_shapes():
    X, y = generate_gaussian_label_shift_data(
        n_samples=2000,
        class_priors=(0.5, 0.5),
        n_features=10,
        seed=42,
    )

    assert X.shape == (2000, 10)
    assert y.shape == (2000,)


def test_reproducibility():
    X1, y1 = generate_gaussian_label_shift_data(
        n_samples=1000,
        class_priors=(0.5, 0.5),
        seed=123,
    )

    X2, y2 = generate_gaussian_label_shift_data(
        n_samples=1000,
        class_priors=(0.5, 0.5),
        seed=123,
    )

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_label_shift_priors_are_approximately_correct():
    _, y_s, _, y_t = generate_source_target_label_shift(
        n_source=10000,
        n_target=10000,
        source_priors=(0.5, 0.5),
        target_priors=(0.7, 0.3),
        source_seed=10,
        target_seed=11,
    )

    source_class0 = (y_s == 0).mean()
    target_class0 = (y_t == 0).mean()

    assert abs(source_class0 - 0.5) < 0.03
    assert abs(target_class0 - 0.7) < 0.03


def test_class_conditional_means_are_stable():
    X_s, y_s, X_t, y_t = generate_source_target_label_shift(
        n_source=10000,
        n_target=10000,
        source_priors=(0.5, 0.5),
        target_priors=(0.7, 0.3),
        n_features=10,
        mean_separation=1.0,
        source_seed=100,
        target_seed=101,
    )

    for cls, expected_mean in [(0, -1.0), (1, 1.0)]:
        source_mean = X_s[y_s == cls, 0].mean()
        target_mean = X_t[y_t == cls, 0].mean()

        assert abs(source_mean - expected_mean) < 0.1
        assert abs(target_mean - expected_mean) < 0.1
        assert abs(source_mean - target_mean) < 0.1


def test_invalid_priors_raise_error():
    with pytest.raises(ValueError):
        generate_gaussian_label_shift_data(
            n_samples=100,
            class_priors=(0.8, 0.8),
        )