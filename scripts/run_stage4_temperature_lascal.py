"""Stage 4 sanity check: LaSCal under controlled temperature miscalibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.calibration.lascal import (
    compute_supervised_classwise_error,
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


SEED = 1

N_SOURCE = 2000
N_TARGET = 2000

SOURCE_PRIORS = (0.5, 0.5)
TARGET_PRIORS = (0.7, 0.3)

TEMPERATURES = [0.5, 1.0, 2.0]

P = 2
N_BINS = 15

RESULTS_PATH = Path(
    "results/stage4/stage4_temperature_lascal.json"
)


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Apply temperature scaling to probability vectors.

    For positive T:

        p_T(y|x) = softmax(log(p(y|x)) / T)

    The same transformation must be applied to source-validation
    and target probabilities because temperature scaling defines
    a modified classifier.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must have shape "
            "(n_samples, n_classes)."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    clipped = np.clip(
        probabilities,
        1e-12,
        1.0,
    )

    log_probabilities = np.log(
        clipped
    )

    scaled_logits = (
        log_probabilities
        / temperature
    )

    scaled_logits = (
        scaled_logits
        - np.max(
            scaled_logits,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        scaled_logits
    )

    scaled_probabilities = (
        exponentials
        / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
    )

    return scaled_probabilities


def main() -> None:
    # ---------------------------------------------------------
    # 1. Generate the same Stage 4 pure-label-shift baseline.
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
    # 2. Source split.
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
    # 3. Train the non-graph classifier.
    # ---------------------------------------------------------
    model = train_logistic_regression(
        X_train,
        y_train,
    )

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
    # 4. Estimate BBSE weights once.
    #
    # Positive temperature scaling preserves argmax predictions,
    # so the hard-BBSE quantities are unchanged across T.
    # ---------------------------------------------------------
    n_classes = (
        validation_probabilities.shape[1]
    )

    confusion_matrix_bbse = np.zeros(
        (n_classes, n_classes),
        dtype=float,
    )

    for true_class in range(n_classes):
        true_mask = (
            y_validation == true_class
        )

        class_count = np.sum(
            true_mask
        )

        if class_count == 0:
            raise ValueError(
                "A source-validation class is empty."
            )

        for predicted_class in range(
            n_classes
        ):
            confusion_matrix_bbse[
                predicted_class,
                true_class,
            ] = (
                np.sum(
                    validation_predictions[
                        true_mask
                    ]
                    == predicted_class
                )
                / class_count
            )

    target_prediction_distribution = (
        estimate_target_prediction_distribution(
            target_predictions,
            n_classes=n_classes,
        )
    )

    estimated_target_priors = (
        estimate_target_priors_bbse(
            confusion_matrix_bbse,
            target_prediction_distribution,
        )
    )

    source_validation_priors = np.asarray(
        [
            np.mean(
                y_validation
                == class_index
            )
            for class_index in range(
                n_classes
            )
        ],
        dtype=float,
    )

    bbse_weights = compute_importance_weights(
        estimated_target_priors,
        source_validation_priors,
    )

    # ---------------------------------------------------------
    # 5. Evaluate LaSCal and supervised target CE at each T.
    # ---------------------------------------------------------
    temperature_results = {}

    print(
        "=== STAGE 4 LASCAL TEMPERATURE "
        "SANITY CHECK ==="
    )
    print()

    print(
        "BBSE weights:"
    )
    print(
        np.round(
            bbse_weights,
            6,
        )
    )
    print()

    for temperature in TEMPERATURES:
        validation_probabilities_t = (
            apply_temperature(
                validation_probabilities,
                temperature,
            )
        )

        target_probabilities_t = (
            apply_temperature(
                target_probabilities,
                temperature,
            )
        )

        lascal_classwise = []
        supervised_classwise = []

        for class_index in range(
            n_classes
        ):
            source_indicators = (
                y_validation
                == class_index
            ).astype(float)

            target_indicators = (
                y_target
                == class_index
            ).astype(float)

            lascal_value = (
                estimate_lascal_classwise_error(
                    source_class_probabilities=(
                        validation_probabilities_t[
                            :,
                            class_index,
                        ]
                    ),
                    source_class_indicators=(
                        source_indicators
                    ),
                    target_class_probabilities=(
                        target_probabilities_t[
                            :,
                            class_index,
                        ]
                    ),
                    class_weight=float(
                        bbse_weights[
                            class_index
                        ]
                    ),
                    p=P,
                    n_bins=N_BINS,
                    adaptive_bins=True,
                )
            )

            supervised_value = (
                compute_supervised_classwise_error(
                    class_probabilities=(
                        target_probabilities_t[
                            :,
                            class_index,
                        ]
                    ),
                    class_indicators=(
                        target_indicators
                    ),
                    p=P,
                    n_bins=N_BINS,
                    adaptive_bins=True,
                )
            )

            lascal_classwise.append(
                float(lascal_value)
            )

            supervised_classwise.append(
                float(supervised_value)
            )

        lascal_array = np.asarray(
            lascal_classwise,
            dtype=float,
        )

        supervised_array = np.asarray(
            supervised_classwise,
            dtype=float,
        )

        absolute_error = np.abs(
            lascal_array
            - supervised_array
        )

        macro_absolute_error = float(
            np.mean(
                absolute_error
            )
        )

        temperature_key = (
            f"{temperature:.1f}"
        )

        temperature_results[
            temperature_key
        ] = {
            "temperature": (
                float(temperature)
            ),
            "lascal_classwise_ce": (
                lascal_array.tolist()
            ),
            "supervised_target_classwise_ce": (
                supervised_array.tolist()
            ),
            "classwise_absolute_error": (
                absolute_error.tolist()
            ),
            "macro_absolute_error": (
                macro_absolute_error
            ),
        }

        print(
            f"T={temperature:.1f}"
        )

        print(
            "LaSCal classwise CE:"
        )

        print(
            np.round(
                lascal_array,
                6,
            )
        )

        print(
            "Supervised target classwise CE:"
        )

        print(
            np.round(
                supervised_array,
                6,
            )
        )

        print(
            "Absolute estimation error:"
        )

        print(
            np.round(
                absolute_error,
                6,
            )
        )

        print(
            "Macro absolute estimation error:"
        )

        print(
            f"{macro_absolute_error:.6f}"
        )

        print()

    # ---------------------------------------------------------
    # 6. Save reproducible results.
    # ---------------------------------------------------------
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "stage4_lascal_temperature_sanity"
        ),
        "seed": SEED,
        "n_source": N_SOURCE,
        "n_target": N_TARGET,
        "source_priors": (
            list(SOURCE_PRIORS)
        ),
        "target_priors": (
            list(TARGET_PRIORS)
        ),
        "temperatures": (
            TEMPERATURES
        ),
        "lascal_p": P,
        "lascal_n_bins": N_BINS,
        "bbse_weights": (
            bbse_weights.tolist()
        ),
        "temperature_results": (
            temperature_results
        ),
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "Saved results to:"
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()