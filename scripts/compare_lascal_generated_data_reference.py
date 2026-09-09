"""Compare our LaSCal estimator with the official implementation on generated data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.calibration.lascal import (
    estimate_lascal_classwise_error,
)
from src.data.synthetic_label_shift import (
    generate_source_target_label_shift,
    split_source_data,
)
from src.label_shift.bbse import (
    compute_importance_weights,
    estimate_target_prediction_distribution,
    estimate_target_priors_bbse,
)
from src.models.baseline import (
    predict_labels_and_probabilities,
    train_logistic_regression,
)


# ---------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------
SEED = 1

N_SOURCE = 2000
N_TARGET = 2000

SOURCE_PRIORS = (
    0.5,
    0.5,
)

TARGET_PRIORS = (
    0.7,
    0.3,
)

P = 2
N_BINS = 15

RTOL = 1e-7
ATOL = 1e-8


def load_official_ece_label_shift():
    """Load EceLabelShift directly from the official repository."""

    official_path = Path(
        "../label-shift-calibration-main/"
        "src/lascal/calibration_error/"
        "ece_label_shift.py"
    )

    if not official_path.exists():
        raise FileNotFoundError(
            "Official LaSCal implementation was not found at "
            f"{official_path}"
        )

    specification = (
        importlib.util.spec_from_file_location(
            "official_ece_label_shift",
            official_path,
        )
    )

    if specification is None:
        raise RuntimeError(
            "Could not construct import specification "
            "for official LaSCal implementation."
        )

    if specification.loader is None:
        raise RuntimeError(
            "Official LaSCal module has no loader."
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(
        module
    )

    return module.EceLabelShift


def probabilities_to_logits(
    probabilities: np.ndarray,
) -> torch.Tensor:
    """
    Convert probabilities to logits.

    Softmax(log(probabilities)) recovers the original
    probability vector, up to floating-point precision.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    probabilities = (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )

    return torch.tensor(
        np.log(probabilities),
        dtype=torch.float64,
    )


def main() -> None:
    # ---------------------------------------------------------
    # 1. Generate the same pure-label-shift setting used by
    #    the Stage 4 baseline.
    # ---------------------------------------------------------
    (
        X_source,
        y_source,
        X_target,
        y_target,
    ) = generate_source_target_label_shift(
        n_source=N_SOURCE,
        n_target=N_TARGET,
        source_priors=SOURCE_PRIORS,
        target_priors=TARGET_PRIORS,
        source_seed=SEED,
        target_seed=SEED + 1,
    )

    # ---------------------------------------------------------
    # 2. Split source data exactly as in the baseline.
    # ---------------------------------------------------------
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        _X_test,
        _y_test,
    ) = split_source_data(
        X_source,
        y_source,
        seed=SEED,
    )

    # ---------------------------------------------------------
    # 3. Train the same non-graph classifier.
    # ---------------------------------------------------------
    model = train_logistic_regression(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 4. Obtain source-validation and target probabilities.
    # ---------------------------------------------------------
    (
        validation_predictions,
        validation_probabilities,
    ) = predict_labels_and_probabilities(
        model,
        X_validation,
    )

    (
        target_predictions,
        target_probabilities,
    ) = predict_labels_and_probabilities(
        model,
        X_target,
    )

    # ---------------------------------------------------------
    # 5. Estimate BBSE target priors and importance weights.
    # ---------------------------------------------------------
    confusion_true_predicted = confusion_matrix(
        y_validation,
        validation_predictions,
        labels=[0, 1],
        normalize="true",
    )

    confusion_bbse = (
        confusion_true_predicted.T
    )

    target_prediction_distribution = (
        estimate_target_prediction_distribution(
            target_predictions,
            n_classes=2,
        )
    )

    estimated_target_priors = (
        estimate_target_priors_bbse(
            confusion_bbse,
            target_prediction_distribution,
        )
    )

    estimated_source_priors = np.array(
        [
            (y_validation == 0).mean(),
            (y_validation == 1).mean(),
        ],
        dtype=float,
    )

    bbse_weights = compute_importance_weights(
        estimated_target_priors,
        estimated_source_priors,
    )

    # ---------------------------------------------------------
    # 6. Compute OUR classwise LaSCal estimate.
    # ---------------------------------------------------------
    our_classwise_ce = []

    for class_index in range(2):
        class_ce = (
            estimate_lascal_classwise_error(
                source_class_probabilities=(
                    validation_probabilities[
                        :,
                        class_index,
                    ]
                ),
                source_class_indicators=(
                    y_validation
                    == class_index
                ).astype(float),
                target_class_probabilities=(
                    target_probabilities[
                        :,
                        class_index,
                    ]
                ),
                class_weight=(
                    bbse_weights[
                        class_index
                    ]
                ),
                p=P,
                n_bins=N_BINS,
                adaptive_bins=True,
            )
        )

        our_classwise_ce.append(
            class_ce
        )

    our_classwise_ce = np.asarray(
        our_classwise_ce,
        dtype=float,
    )

    # ---------------------------------------------------------
    # 7. Compute OFFICIAL LaSCal estimate using exactly the
    #    same probabilities, labels, weights, p, and bins.
    #
    #    The official API expects logits and applies softmax
    #    internally, so log(probability) is supplied.
    # ---------------------------------------------------------
    OfficialEceLabelShift = (
        load_official_ece_label_shift()
    )

    official_estimator = (
        OfficialEceLabelShift(
            p=P,
            n_bins=N_BINS,
            adaptive_bins=True,
            classwise=True,
        )
    )

    validation_logits = (
        probabilities_to_logits(
            validation_probabilities
        )
    )

    target_logits = (
        probabilities_to_logits(
            target_probabilities
        )
    )

    validation_labels_tensor = (
        torch.tensor(
            y_validation,
            dtype=torch.long,
        )
    )

    weights_tensor = torch.tensor(
        bbse_weights,
        dtype=torch.float64,
    )

    with torch.no_grad():
        official_output = (
            official_estimator(
                logits=target_logits,
                logits_source=validation_logits,
                labels_source=(
                    validation_labels_tensor
                ),
                weights=weights_tensor,
            )
        )

    official_classwise_ce = (
        official_output
        .detach()
        .cpu()
        .numpy()
        .astype(float)
    )

    # ---------------------------------------------------------
    # 8. Compare implementations.
    # ---------------------------------------------------------
    absolute_difference = np.abs(
        our_classwise_ce
        - official_classwise_ce
    )

    maximum_absolute_difference = float(
        np.max(
            absolute_difference
        )
    )

    equivalent = np.allclose(
        our_classwise_ce,
        official_classwise_ce,
        rtol=RTOL,
        atol=ATOL,
    )

    # ---------------------------------------------------------
    # 9. Report diagnostics.
    # ---------------------------------------------------------
    print(
        "=== GENERATED-DATA LASCAL "
        "REFERENCE CHECK ==="
    )

    print()
    print("Seed:")
    print(SEED)

    print()
    print("Source sample size:")
    print(N_SOURCE)

    print()
    print("Source validation size:")
    print(len(y_validation))

    print()
    print("Target sample size:")
    print(N_TARGET)

    print()
    print("Configured target priors:")
    print(
        np.asarray(
            TARGET_PRIORS
        )
    )

    print()
    print("BBSE estimated target priors:")
    print(
        estimated_target_priors
    )

    print()
    print("BBSE weights:")
    print(
        bbse_weights
    )

    print()
    print("Our LaSCal classwise CE:")
    print(
        our_classwise_ce
    )

    print()
    print("Official LaSCal classwise CE:")
    print(
        official_classwise_ce
    )

    print()
    print("Absolute difference:")
    print(
        absolute_difference
    )

    print()
    print("Maximum absolute difference:")
    print(
        maximum_absolute_difference
    )

    print()
    print(
        "Numerically equivalent "
        f"(rtol={RTOL}, atol={ATOL}):"
    )
    print(
        equivalent
    )

    if not equivalent:
        raise AssertionError(
            "Our LaSCal implementation does not "
            "match the official implementation "
            "within the configured tolerances."
        )


if __name__ == "__main__":
    main()