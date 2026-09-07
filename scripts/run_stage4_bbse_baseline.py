"""Run the Stage 4 non-graph BBSE baseline experiment."""

from __future__ import annotations
from src.utils.results import save_json_results

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.synthetic_label_shift import (
    generate_source_target_label_shift,
    split_source_data,
)
from src.models.baseline import (
    train_logistic_regression,
    predict_labels_and_probabilities,
)
from src.label_shift.bbse import (
    estimate_target_prediction_distribution,
    estimate_target_priors_bbse,
    compute_importance_weights,
    is_valid_probability_vector,
)


def main() -> None:
    # Generate source and target data under pure label shift.
    X_s, y_s, X_t, y_t = generate_source_target_label_shift()

    # Split source data into train, validation, and test sets.
    X_train, y_train, X_val, y_val, X_test, y_test = split_source_data(
        X_s,
        y_s,
    )

    # Train the classifier using source training data only.
    model = train_logistic_regression(
        X_train,
        y_train,
    )

    # Predictions needed for BBSE.
    pred_test, _ = predict_labels_and_probabilities(
        model,
        X_test,
    )

    pred_target, _ = predict_labels_and_probabilities(
        model,
        X_t,
    )

    # Build source confusion matrix.
    # sklearn orientation:
    # rows = true class
    # columns = predicted class
    cm_true_pred = confusion_matrix(
        y_test,
        pred_test,
        labels=[0, 1],
        normalize="true",
    )

    # BBSE orientation:
    # rows = predicted class
    # columns = true class
    C = cm_true_pred.T

    # Observable target predicted-label distribution.
    mu_hat = estimate_target_prediction_distribution(
        pred_target,
        n_classes=2,
    )

    # Estimate hidden target class priors with BBSE.
    q_hat_target = estimate_target_priors_bbse(
        C,
        mu_hat,
    )

    # Empirical source priors from held-out source data.
    q_hat_source = np.array([
        (y_test == 0).mean(),
        (y_test == 1).mean(),
    ])

    # Estimated importance weights.
    estimated_weights = compute_importance_weights(
        q_hat_target,
        q_hat_source,
    )

    # ---------------------------------------------------------
    # Researcher-side ground truth
    # These values must NEVER be used by BBSE itself.
    # ---------------------------------------------------------

    q_true_target = np.array([
        (y_t == 0).mean(),
        (y_t == 1).mean(),
    ])

    oracle_weights = compute_importance_weights(
        q_true_target,
        q_hat_source,
    )

    # Main diagnostic errors.
    prior_error = np.linalg.norm(
        q_hat_target - q_true_target
    )

    weight_error = np.linalg.norm(
        estimated_weights - oracle_weights
    )

    condition_number = np.linalg.cond(C)

    priors_valid = is_valid_probability_vector(
        q_hat_target
    )
    # Store the experiment configuration and numerical results.
    results = {
        "experiment": "stage4_bbse_baseline",
        "source_priors_empirical": q_hat_source.tolist(),
        "target_priors_estimated": q_hat_target.tolist(),
        "target_priors_true": q_true_target.tolist(),
        "target_prediction_distribution": mu_hat.tolist(),
        "importance_weights_estimated": estimated_weights.tolist(),
        "importance_weights_oracle": oracle_weights.tolist(),
        "prior_l2_error": float(prior_error),
        "weight_l2_error": float(weight_error),
        "confusion_matrix": C.tolist(),
        "confusion_matrix_condition_number": float(condition_number),
        "estimated_priors_valid": bool(priors_valid),
    }

    save_json_results(
        results,
        "results/stage4/bbse_baseline.json",
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("=== STAGE 4 BBSE BASELINE ===")
    print()

    print("Source confusion matrix C:")
    print(np.round(C, 3))
    print()

    print("Condition number:")
    print(round(condition_number, 4))
    print()

    print("Observed target prediction distribution:")
    print(np.round(mu_hat, 3))
    print()

    print("Estimated target priors:")
    print(np.round(q_hat_target, 3))
    print()

    print("True target priors (researcher-side only):")
    print(np.round(q_true_target, 3))
    print()

    print("Prior-estimation L2 error:")
    print(round(prior_error, 4))
    print()

    print("Estimated importance weights:")
    print(np.round(estimated_weights, 3))
    print()

    print("Oracle importance weights:")
    print(np.round(oracle_weights, 3))
    print()

    print("Importance-weight L2 error:")
    print(round(weight_error, 4))
    print()

    print("Estimated priors form valid probability vector:")
    print(priors_valid)


if __name__ == "__main__":
    main()
