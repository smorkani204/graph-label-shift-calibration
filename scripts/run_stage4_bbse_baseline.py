"""Run the Stage 4 non-graph BBSE, RLLS, and LaSCal baseline experiment."""

from __future__ import annotations

import argparse

import numpy as np

from sklearn.metrics import confusion_matrix

from src.calibration.lascal import (
    compute_supervised_classwise_error,
    estimate_lascal_classwise_error,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 4 non-graph "
            "BBSE/RLLS/LaSCal baseline."
        )
    )

    parser.add_argument(
        "--target-prior-0",
        type=float,
        default=0.7,
        help=(
            "Configured target probability for class 0. "
            "Class-1 probability is 1 - target-prior-0."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for the experiment.",
    )

    parser.add_argument(
        "--n-source",
        type=int,
        default=2000,
        help=(
            "Number of source samples generated before the "
            "train/validation/test split."
        ),
    )

    parser.add_argument(
        "--n-target",
        type=int,
        default=2000,
        help="Number of unlabeled target samples.",
    )

    return parser.parse_args()


def main() -> None:
    # ---------------------------------------------------------
    # 1. Read experiment configuration
    # ---------------------------------------------------------
    args = parse_args()

    target_prior_0 = args.target_prior_0
    seed = args.seed
    n_source = args.n_source
    n_target = args.n_target

    if not 0.0 <= target_prior_0 <= 1.0:
        raise ValueError(
            "--target-prior-0 must be between 0 and 1."
        )

    if n_source <= 0:
        raise ValueError(
            "--n-source must be a positive integer."
        )

    if n_target <= 0:
        raise ValueError(
            "--n-target must be a positive integer."
        )

    target_prior_1 = 1.0 - target_prior_0

    configured_source_priors = (
        0.5,
        0.5,
    )

    configured_target_priors = (
        target_prior_0,
        target_prior_1,
    )

    # ---------------------------------------------------------
    # 2. Generate source and target data
    #
    # Source and target share P(X | Y).
    #
    # Only the class priors may differ.
    #
    # We use:
    #
    # source seed = seed
    # target seed = seed + 1
    #
    # so source and target samples are generated independently
    # while remaining exactly reproducible.
    # ---------------------------------------------------------
    X_s, y_s, X_t, y_t = generate_source_target_label_shift(
        n_source=n_source,
        n_target=n_target,
        source_priors=configured_source_priors,
        target_priors=configured_target_priors,
        source_seed=seed,
        target_seed=seed + 1,
    )

    # ---------------------------------------------------------
    # 3. Split labeled source data
    #
    # The same base seed is used for the source split so the
    # entire experiment is reproducible from one integer.
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
        seed=seed,
    )

    # ---------------------------------------------------------
    # 4. Train classifier using source training data only
    # ---------------------------------------------------------
    model = train_logistic_regression(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 5. Generate validation, test, and target predictions
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
    # 6. Researcher-side calibration evaluation
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
    # 7. Controlled temperature stress test
    # ---------------------------------------------------------
    temperatures = [
        0.5,
        1.0,
        2.0,
    ]

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
    # 8. Build source-validation confusion matrix for BBSE
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
    # C_ij = P_s(
    #     predicted class = i
    #     |
    #     true class = j
    # )
    #
    # rows    = predicted class
    # columns = true class
    C = cm_true_pred.T

    # ---------------------------------------------------------
    # 9. Estimate observable target prediction distribution
    # ---------------------------------------------------------
    mu_hat = estimate_target_prediction_distribution(
        pred_target,
        n_classes=2,
    )

    # ---------------------------------------------------------
    # 10. Estimate hidden target priors using BBSE
    # ---------------------------------------------------------
    q_hat_target_bbse = estimate_target_priors_bbse(
        C,
        mu_hat,
    )

    # ---------------------------------------------------------
    # 11. Estimate source priors from validation labels
    # ---------------------------------------------------------
    q_hat_source = np.array(
        [
            (y_val == 0).mean(),
            (y_val == 1).mean(),
        ]
    )

    # ---------------------------------------------------------
    # 12. Compute BBSE importance weights
    # ---------------------------------------------------------
    bbse_weights = compute_importance_weights(
        q_hat_target_bbse,
        q_hat_source,
    )

    # ---------------------------------------------------------
    # 13. Compute RLLS-hard importance weights
    # ---------------------------------------------------------
    rlls_weights = estimate_rlls_hard_weights(
        source_probabilities=prob_val,
        source_labels=y_val,
        target_probabilities=prob_target,
        alpha=0.01,
        delta=0.05,
    )

    # ---------------------------------------------------------
    # 14. Researcher-side ground truth
    #
    # Target labels below are used ONLY for evaluation.
    # They are never used by BBSE or RLLS.
    # ---------------------------------------------------------
    q_true_target = np.array(
        [
            (y_t == 0).mean(),
            (y_t == 1).mean(),
        ]
    )

    # Empirical-prior oracle weights:
    #
    # empirical target priors
    # divided by
    # empirical source-validation priors.
    oracle_weights = compute_importance_weights(
        q_true_target,
        q_hat_source,
    )

    # ---------------------------------------------------------
    # 15. Class-wise target calibration-error estimation
    # ---------------------------------------------------------
    lascal_p = 2
    lascal_n_bins = 15

    true_target_classwise_ce = []
    lascal_bbse_classwise_ce = []
    lascal_rlls_classwise_ce = []
    lascal_oracle_classwise_ce = []

    for class_index in range(2):
        source_class_probabilities = (
            prob_val[:, class_index]
        )

        source_class_indicators = (
            y_val == class_index
        ).astype(float)

        target_class_probabilities = (
            prob_target[:, class_index]
        )

        target_class_indicators = (
            y_t == class_index
        ).astype(float)

        # -----------------------------------------------------
        # Supervised target reference
        #
        # Target labels are used only for researcher-side
        # evaluation.
        # -----------------------------------------------------
        true_ce = compute_supervised_classwise_error(
            class_probabilities=target_class_probabilities,
            class_indicators=target_class_indicators,
            p=lascal_p,
            n_bins=lascal_n_bins,
            adaptive_bins=True,
        )

        # -----------------------------------------------------
        # Label-free LaSCal estimate using BBSE weight
        # -----------------------------------------------------
        bbse_ce = estimate_lascal_classwise_error(
            source_class_probabilities=source_class_probabilities,
            source_class_indicators=source_class_indicators,
            target_class_probabilities=target_class_probabilities,
            class_weight=bbse_weights[class_index],
            p=lascal_p,
            n_bins=lascal_n_bins,
            adaptive_bins=True,
        )

        # -----------------------------------------------------
        # Label-free LaSCal estimate using RLLS weight
        # -----------------------------------------------------
        rlls_ce = estimate_lascal_classwise_error(
            source_class_probabilities=source_class_probabilities,
            source_class_indicators=source_class_indicators,
            target_class_probabilities=target_class_probabilities,
            class_weight=rlls_weights[class_index],
            p=lascal_p,
            n_bins=lascal_n_bins,
            adaptive_bins=True,
        )

        # -----------------------------------------------------
        # LaSCal estimate using empirical-prior oracle weight
        # -----------------------------------------------------
        oracle_ce = estimate_lascal_classwise_error(
            source_class_probabilities=source_class_probabilities,
            source_class_indicators=source_class_indicators,
            target_class_probabilities=target_class_probabilities,
            class_weight=oracle_weights[class_index],
            p=lascal_p,
            n_bins=lascal_n_bins,
            adaptive_bins=True,
        )

        true_target_classwise_ce.append(
            true_ce
        )

        lascal_bbse_classwise_ce.append(
            bbse_ce
        )

        lascal_rlls_classwise_ce.append(
            rlls_ce
        )

        lascal_oracle_classwise_ce.append(
            oracle_ce
        )

    true_target_classwise_ce = np.asarray(
        true_target_classwise_ce,
        dtype=float,
    )

    lascal_bbse_classwise_ce = np.asarray(
        lascal_bbse_classwise_ce,
        dtype=float,
    )

    lascal_rlls_classwise_ce = np.asarray(
        lascal_rlls_classwise_ce,
        dtype=float,
    )

    lascal_oracle_classwise_ce = np.asarray(
        lascal_oracle_classwise_ce,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 16. Calibration-error estimation diagnostics
    # ---------------------------------------------------------
    bbse_classwise_ce_error = np.abs(
        lascal_bbse_classwise_ce
        - true_target_classwise_ce
    )

    rlls_classwise_ce_error = np.abs(
        lascal_rlls_classwise_ce
        - true_target_classwise_ce
    )

    oracle_classwise_ce_error = np.abs(
        lascal_oracle_classwise_ce
        - true_target_classwise_ce
    )

    bbse_macro_ce_error = float(
        np.mean(
            bbse_classwise_ce_error
        )
    )

    rlls_macro_ce_error = float(
        np.mean(
            rlls_classwise_ce_error
        )
    )

    oracle_macro_ce_error = float(
        np.mean(
            oracle_classwise_ce_error
        )
    )

    # ---------------------------------------------------------
    # 17. Prior / importance-weight diagnostics
    # ---------------------------------------------------------
    bbse_prior_error = np.linalg.norm(
        q_hat_target_bbse
        - q_true_target
    )

    bbse_weight_error = np.linalg.norm(
        bbse_weights
        - oracle_weights
    )

    rlls_weight_error = np.linalg.norm(
        rlls_weights
        - oracle_weights
    )

    condition_number = np.linalg.cond(
        C
    )

    bbse_priors_valid = (
        is_valid_probability_vector(
            q_hat_target_bbse
        )
    )

    # ---------------------------------------------------------
    # 18. Determine experiment label
    # ---------------------------------------------------------
    if np.isclose(
        target_prior_0,
        0.5,
    ):
        experiment_label = (
            "NO-SHIFT SANITY CONTROL"
        )

        experiment_name = (
            "stage4_no_shift_sanity_control"
        )
    else:
        experiment_label = (
            "LABEL-SHIFT BASELINE"
        )

        experiment_name = (
            "stage4_label_shift_baseline"
        )

    # ---------------------------------------------------------
    # 19. Store experiment results
    # ---------------------------------------------------------
    results = {
        "experiment": experiment_name,
        "seed": seed,
        "source_seed": seed,
        "target_seed": seed + 1,

        # Sample-size configuration
        "n_source": int(n_source),
        "n_target": int(n_target),
        "n_source_train": int(len(y_train)),
        "n_source_validation": int(len(y_val)),
        "n_source_test": int(len(y_test)),

        "source_priors_configured": list(
            configured_source_priors
        ),
        "target_priors_configured": list(
            configured_target_priors
        ),
        "source_priors_empirical": (
            q_hat_source.tolist()
        ),
        "target_priors_bbse": (
            q_hat_target_bbse.tolist()
        ),
        "target_priors_true": (
            q_true_target.tolist()
        ),
        "target_prediction_distribution": (
            mu_hat.tolist()
        ),
        "importance_weights_bbse": (
            bbse_weights.tolist()
        ),
        "importance_weights_rlls": (
            rlls_weights.tolist()
        ),
        "importance_weights_oracle": (
            oracle_weights.tolist()
        ),
        "bbse_prior_l2_error": float(
            bbse_prior_error
        ),
        "bbse_weight_l2_error": float(
            bbse_weight_error
        ),
        "rlls_weight_l2_error": float(
            rlls_weight_error
        ),
        "confusion_matrix": (
            C.tolist()
        ),
        "confusion_matrix_condition_number": float(
            condition_number
        ),
        "bbse_estimated_priors_valid": bool(
            bbse_priors_valid
        ),
        "rlls_alpha": 0.01,
        "rlls_delta": 0.05,
        "lascal_p": lascal_p,
        "lascal_n_bins": lascal_n_bins,
        "true_target_classwise_ce": (
            true_target_classwise_ce.tolist()
        ),
        "lascal_bbse_classwise_ce": (
            lascal_bbse_classwise_ce.tolist()
        ),
        "lascal_rlls_classwise_ce": (
            lascal_rlls_classwise_ce.tolist()
        ),
        "lascal_oracle_classwise_ce": (
            lascal_oracle_classwise_ce.tolist()
        ),
        "bbse_classwise_ce_absolute_error": (
            bbse_classwise_ce_error.tolist()
        ),
        "rlls_classwise_ce_absolute_error": (
            rlls_classwise_ce_error.tolist()
        ),
        "oracle_classwise_ce_absolute_error": (
            oracle_classwise_ce_error.tolist()
        ),
        "bbse_macro_ce_absolute_error": (
            bbse_macro_ce_error
        ),
        "rlls_macro_ce_absolute_error": (
            rlls_macro_ce_error
        ),
        "oracle_macro_ce_absolute_error": (
            oracle_macro_ce_error
        ),
        "source_ece": float(
            source_ece
        ),
        "target_ece": float(
            target_ece
        ),
        "source_nll": float(
            source_nll
        ),
        "target_nll": float(
            target_nll
        ),
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
    # 20. Create unique output filename
    #
    # Including sample sizes is important now because otherwise
    # sample-size experiments would overwrite one another.
    #
    # Example:
    #
    # target prior 0 = 0.70
    # n_source = 4000
    # n_target = 2000
    # seed = 1
    #
    # baseline_target_prior0_0p70_ns4000_nt2000_seed1.json
    # ---------------------------------------------------------
    target_prior_tag = (
        f"{target_prior_0:.2f}"
        .replace(".", "p")
    )

    output_path = (
        "results/stage4/"
        f"baseline_target_prior0_{target_prior_tag}_"
        f"ns{n_source}_"
        f"nt{n_target}_"
        f"seed{seed}.json"
    )

    # ---------------------------------------------------------
    # 21. Save results
    # ---------------------------------------------------------
    save_json_results(
        results,
        output_path,
    )

    # ---------------------------------------------------------
    # 22. Print experiment diagnostics
    # ---------------------------------------------------------
    print(
        f"=== STAGE 4 {experiment_label} ==="
    )

    print()
    print("Base seed:")
    print(seed)

    print()
    print("Source seed:")
    print(seed)

    print()
    print("Target seed:")
    print(seed + 1)

    print()
    print("Source sample size:")
    print(n_source)

    print()
    print("Target sample size:")
    print(n_target)

    print()
    print("Source train / validation / test sizes:")
    print(
        len(y_train),
        len(y_val),
        len(y_test),
    )

    print()
    print("Configured source priors:")
    print(
        np.round(
            configured_source_priors,
            3,
        )
    )

    print()
    print("Configured target priors:")
    print(
        np.round(
            configured_target_priors,
            3,
        )
    )

    print()
    print("Source-validation confusion matrix C:")
    print(
        np.round(
            C,
            3,
        )
    )

    print()
    print("Condition number:")
    print(
        round(
            condition_number,
            4,
        )
    )

    print()
    print("Source-validation empirical priors:")
    print(
        np.round(
            q_hat_source,
            3,
        )
    )

    print()
    print("Observed target prediction distribution:")
    print(
        np.round(
            mu_hat,
            3,
        )
    )

    print()
    print("BBSE estimated target priors:")
    print(
        np.round(
            q_hat_target_bbse,
            3,
        )
    )

    print()
    print(
        "True target priors "
        "(researcher-side only):"
    )
    print(
        np.round(
            q_true_target,
            3,
        )
    )

    print()
    print("BBSE prior-estimation L2 error:")
    print(
        round(
            bbse_prior_error,
            4,
        )
    )

    print()
    print("BBSE importance weights:")
    print(
        np.round(
            bbse_weights,
            3,
        )
    )

    print()
    print("RLLS importance weights:")
    print(
        np.round(
            rlls_weights,
            3,
        )
    )

    print()
    print("Oracle importance weights:")
    print(
        np.round(
            oracle_weights,
            3,
        )
    )

    print()
    print("BBSE importance-weight L2 error:")
    print(
        round(
            bbse_weight_error,
            4,
        )
    )

    print()
    print("RLLS importance-weight L2 error:")
    print(
        round(
            rlls_weight_error,
            4,
        )
    )

    print()
    print(
        "BBSE estimated priors form valid "
        "probability vector:"
    )
    print(
        bbse_priors_valid
    )

    print()
    print(
        "True target classwise CE "
        "(researcher-side only):"
    )
    print(
        np.round(
            true_target_classwise_ce,
            6,
        )
    )

    print()
    print(
        "LaSCal classwise CE "
        "using BBSE weights:"
    )
    print(
        np.round(
            lascal_bbse_classwise_ce,
            6,
        )
    )

    print()
    print(
        "LaSCal classwise CE "
        "using RLLS weights:"
    )
    print(
        np.round(
            lascal_rlls_classwise_ce,
            6,
        )
    )

    print()
    print(
        "LaSCal classwise CE "
        "using oracle weights:"
    )
    print(
        np.round(
            lascal_oracle_classwise_ce,
            6,
        )
    )

    print()
    print(
        "BBSE classwise CE "
        "absolute error:"
    )
    print(
        np.round(
            bbse_classwise_ce_error,
            6,
        )
    )

    print()
    print(
        "RLLS classwise CE "
        "absolute error:"
    )
    print(
        np.round(
            rlls_classwise_ce_error,
            6,
        )
    )

    print()
    print(
        "Oracle-weight classwise CE "
        "absolute error:"
    )
    print(
        np.round(
            oracle_classwise_ce_error,
            6,
        )
    )

    print()
    print(
        "BBSE macro classwise CE "
        "absolute error:"
    )
    print(
        round(
            bbse_macro_ce_error,
            6,
        )
    )

    print()
    print(
        "RLLS macro classwise CE "
        "absolute error:"
    )
    print(
        round(
            rlls_macro_ce_error,
            6,
        )
    )

    print()
    print(
        "Oracle-weight macro classwise CE "
        "absolute error:"
    )
    print(
        round(
            oracle_macro_ce_error,
            6,
        )
    )

    print()
    print("Source test ECE:")
    print(
        round(
            source_ece,
            4,
        )
    )

    print()
    print(
        "Target ECE "
        "(researcher-side only):"
    )
    print(
        round(
            target_ece,
            4,
        )
    )

    print()
    print("Source test NLL:")
    print(
        round(
            source_nll,
            4,
        )
    )

    print()
    print(
        "Target NLL "
        "(researcher-side only):"
    )
    print(
        round(
            target_nll,
            4,
        )
    )

    print()
    print(
        "Target calibration metrics "
        "across temperatures:"
    )

    for result in temperature_results:
        print(
            f"T={result['temperature']}: "
            f"ECE={result['target_ece']:.4f}, "
            f"NLL={result['target_nll']:.4f}"
        )

    print()
    print("Saved results to:")
    print(output_path)


if __name__ == "__main__":
    main()