"""Run the Stage 4 non-graph BBSE and RLLS baseline experiment."""

from __future__ import annotations

import numpy as np

from sklearn.metrics import confusion_matrix

from src.calibration.losses import negative_log_likelihood
from src.calibration.metrics import expected_calibration_error
from src.calibration.temperature import (
    apply_temperature_to_probabilities,
)
from src.data.synthetic_label_shift import (
    generate_source_target_label_shift,
    split_source_data,
)
from src.label_shift.bbse import (
    compute_importance_weights,
    estimate_target_prediction_distribution,
    estimate_target_priors_bbse,
    is_valid_probability_vector,
)
from src.label_shift.rlls import estimate_rlls_hard_weights
from src.models.baseline import (
    predict_labels_and_probabilities,
    train_logistic_regression,
)
from src.utils.results import save_json_results


def main() -> None:
    # ---------------------------------------------------------
    # 1. Generate source and target data under pure label shift
    # ---------------------------------------------------------

    X_s, y_s, X_t, y_t = generate_source_target_label_shift()

    # ---------------------------------------------------------
    # 2. Split labeled source data
    # ---------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_source_data(
        X_s,
        y_s,
    )

    # ---------------------------------------------------------
    # 3. Train classifier using source training data only
    # ---------------------------------------------------------

    model = train_logistic_regression(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 4. Generate validation, test, and target predictions
    # ---------------------------------------------------------

    pred_val, prob_val = predict_labels_and_probabilities(
        model,
        X_val,
    )

    pred_test, prob_test = predict_labels_and_probabilities(
        model,
        X_test,
    )

    pred_target, prob_target = predict_labels_and_probabilities(
        model,
        X_t,
    )

    # ---------------------------------------------------------
    # 5. Researcher-side calibration evaluation
    # ---------------------------------------------------------

    source_ece = expected_calibration_error(
        prob_test,
        y_test,
    )

    target_ece = expected_calibration_error(
        prob_target,
        y_t,
    )

    source_nll = negative_log_likelihood(
        prob_test,
        y_test,
    )

    target_nll = negative_log_likelihood(
        prob_target,
        y_t,
    )

    # ---------------------------------------------------------
    # 6. Controlled temperature stress test
    # ---------------------------------------------------------

    temperatures = [0.5, 1.0, 2.0]

    temperature_results = []

    for temperature in temperatures:
        scaled_target_probabilities = (
            apply_temperature_to_probabilities(
                prob_target,
                temperature=temperature,
            )
        )

        scaled_target_ece = expected_calibration_error(
            scaled_target_probabilities,
            y_t,
        )

        scaled_target_nll = negative_log_likelihood(
            scaled_target_probabilities,
            y_t,
        )

        temperature_results.append(
            {
                "temperature": temperature,
                "target_ece": scaled_target_ece,
                "target_nll": scaled_target_nll,
            }
        )

    # ---------------------------------------------------------
    # 7. Build source-validation confusion matrix for BBSE
    # ---------------------------------------------------------

    # sklearn orientation:
    #
    # rows    = true class
    # columns = predicted class

    cm_true_pred = confusion_matrix(
        y_val,
        pred_val,
        labels=[0, 1],
        normalize="true",
    )

    # BBSE orientation:
    #
    # C_ij = P_s(predicted class = i | true class = j)
    #
    # rows    = predicted class
    # columns = true class

    C = cm_true_pred.T

    # ---------------------------------------------------------
    # 8. Estimate observable target prediction distribution
    # ---------------------------------------------------------

    mu_hat = estimate_target_prediction_distribution(
        pred_target,
        n_classes=2,
    )

    # ---------------------------------------------------------
    # 9. Estimate hidden target priors using BBSE
    # ---------------------------------------------------------

    q_hat_target_bbse = estimate_target_priors_bbse(
        C,
        mu_hat,
    )

    # ---------------------------------------------------------
    # 10. Estimate source priors from validation labels
    # ---------------------------------------------------------

    q_hat_source = np.array(
        [
            (y_val == 0).mean(),
            (y_val == 1).mean(),
        ]
    )

    # ---------------------------------------------------------
    # 11. Compute BBSE importance weights
    # ---------------------------------------------------------

    bbse_weights = compute_importance_weights(
        q_hat_target_bbse,
        q_hat_source,
    )

    # ---------------------------------------------------------
    # 12. Compute RLLS-hard importance weights
    # ---------------------------------------------------------

    # RLLS uses:
    #
    # - source-validation posterior probabilities,
    # - source-validation labels,
    # - unlabeled target posterior probabilities.
    #
    # Target labels are NOT used here.
    #
    # alpha=0.01 and delta=0.05 match the reference
    # RLLS-hard configuration used in our implementation.

    rlls_weights = estimate_rlls_hard_weights(
        source_probabilities=prob_val,
        source_labels=y_val,
        target_probabilities=prob_target,
        alpha=0.01,
        delta=0.05,
    )

    # ---------------------------------------------------------
    # 13. Researcher-side ground truth
    #
    # Target labels below are used ONLY for evaluation.
    # They must never be used by BBSE or RLLS.
    # ---------------------------------------------------------

    q_true_target = np.array(
        [
            (y_t == 0).mean(),
            (y_t == 1).mean(),
        ]
    )

    oracle_weights = compute_importance_weights(
        q_true_target,
        q_hat_source,
    )

    # ---------------------------------------------------------
    # 14. Diagnostic errors
    # ---------------------------------------------------------

    bbse_prior_error = np.linalg.norm(
        q_hat_target_bbse - q_true_target
    )

    bbse_weight_error = np.linalg.norm(
        bbse_weights - oracle_weights
    )

    rlls_weight_error = np.linalg.norm(
        rlls_weights - oracle_weights
    )

    condition_number = np.linalg.cond(C)

    bbse_priors_valid = is_valid_probability_vector(
        q_hat_target_bbse
    )

    # ---------------------------------------------------------
    # 15. Store experiment results
    # ---------------------------------------------------------

    results = {
        "experiment": "stage4_bbse_rlls_baseline",
        "source_priors_empirical": q_hat_source.tolist(),
        "target_priors_bbse": q_hat_target_bbse.tolist(),
        "target_priors_true": q_true_target.tolist(),
        "target_prediction_distribution": mu_hat.tolist(),
        "importance_weights_bbse": bbse_weights.tolist(),
        "importance_weights_rlls": rlls_weights.tolist(),
        "importance_weights_oracle": oracle_weights.tolist(),
        "bbse_prior_l2_error": float(bbse_prior_error),
        "bbse_weight_l2_error": float(bbse_weight_error),
        "rlls_weight_l2_error": float(rlls_weight_error),
        "confusion_matrix": C.tolist(),
        "confusion_matrix_condition_number": float(
            condition_number
        ),
        "bbse_estimated_priors_valid": bool(
            bbse_priors_valid
        ),
        "rlls_alpha": 0.01,
        "rlls_delta": 0.05,
        "source_ece": float(source_ece),
        "target_ece": float(target_ece),
        "source_nll": float(source_nll),
        "target_nll": float(target_nll),
        "temperature_results": [
            {
                "temperature": float(
                    result["temperature"]
                ),
                "target_ece": float(
                    result["target_ece"]
                ),
                "target_nll": float(
                    result["target_nll"]
                ),
            }
            for result in temperature_results
        ],
    }

    # ---------------------------------------------------------
    # 16. Save results to disk
    # ---------------------------------------------------------

    save_json_results(
        results,
        "results/stage4/bbse_rlls_baseline.json",
    )

    # ---------------------------------------------------------
    # 17. Print experiment diagnostics
    # ---------------------------------------------------------

    print("=== STAGE 4 BBSE + RLLS BASELINE ===")
    print()

    print("Source-validation confusion matrix C:")
    print(np.round(C, 3))
    print()

    print("Condition number:")
    print(round(condition_number, 4))
    print()

    print("Source-validation empirical priors:")
    print(np.round(q_hat_source, 3))
    print()

    print("Observed target prediction distribution:")
    print(np.round(mu_hat, 3))
    print()

    print("BBSE estimated target priors:")
    print(np.round(q_hat_target_bbse, 3))
    print()

    print("True target priors (researcher-side only):")
    print(np.round(q_true_target, 3))
    print()

    print("BBSE prior-estimation L2 error:")
    print(round(bbse_prior_error, 4))
    print()

    print("BBSE importance weights:")
    print(np.round(bbse_weights, 3))
    print()

    print("RLLS importance weights:")
    print(np.round(rlls_weights, 3))
    print()

    print("Oracle importance weights:")
    print(np.round(oracle_weights, 3))
    print()

    print("BBSE importance-weight L2 error:")
    print(round(bbse_weight_error, 4))
    print()

    print("RLLS importance-weight L2 error:")
    print(round(rlls_weight_error, 4))
    print()

    print("BBSE estimated priors form valid probability vector:")
    print(bbse_priors_valid)
    print()

    print("Source test ECE:")
    print(round(source_ece, 4))
    print()

    print("Target ECE (researcher-side only):")
    print(round(target_ece, 4))
    print()

    print("Source test NLL:")
    print(round(source_nll, 4))
    print()

    print("Target NLL (researcher-side only):")
    print(round(target_nll, 4))
    print()

    print("Target calibration metrics across temperatures:")

    for result in temperature_results:
        print(
            f"T={result['temperature']}: "
            f"ECE={result['target_ece']:.4f}, "
            f"NLL={result['target_nll']:.4f}"
        )


if __name__ == "__main__":
    main()
