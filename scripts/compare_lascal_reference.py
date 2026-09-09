"""Numerically compare our LaSCal estimator with the official reference code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------
# Make the project root importable
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.calibration.lascal import (
    estimate_lascal_classwise_error,
)


def load_official_lascal_class():
    """
    Load EceLabelShift directly from the official LaSCal source file.

    We load the file directly instead of importing the full ``lascal``
    package because the package imports optional dependencies that are
    unrelated to this calibration-error comparison.
    """
    official_file = Path(
        "../label-shift-calibration-main/"
        "src/lascal/calibration_error/ece_label_shift.py"
    )

    if not official_file.exists():
        raise FileNotFoundError(
            "Official LaSCal source file was not found at:\n"
            f"{official_file.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        "official_ece_label_shift",
        official_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not create an import specification "
            "for the official LaSCal file."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.EceLabelShift


def main() -> None:
    """
    Compare our class-wise estimator with the official implementation
    on identical controlled inputs.
    """

    # --------------------------------------------------
    # Controlled probabilities and labels
    # --------------------------------------------------

    source_probabilities = np.array(
        [
            [0.90, 0.10],
            [0.75, 0.25],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.25, 0.75],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )

    source_labels = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )

    target_probabilities = np.array(
        [
            [0.85, 0.15],
            [0.70, 0.30],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.30, 0.70],
            [0.15, 0.85],
        ],
        dtype=np.float64,
    )

    weights = np.array(
        [1.20, 0.80],
        dtype=np.float64,
    )

    p = 2
    n_bins = 3

    # --------------------------------------------------
    # Our implementation
    # --------------------------------------------------

    our_results = []

    for class_index in range(2):
        class_result = estimate_lascal_classwise_error(
            source_class_probabilities=(
                source_probabilities[:, class_index]
            ),
            source_class_indicators=(
                source_labels == class_index
            ),
            target_class_probabilities=(
                target_probabilities[:, class_index]
            ),
            class_weight=weights[class_index],
            p=p,
            n_bins=n_bins,
            adaptive_bins=True,
        )

        our_results.append(class_result)

    our_results = np.asarray(
        our_results,
        dtype=np.float64,
    )

    # --------------------------------------------------
    # Official implementation
    # --------------------------------------------------

    EceLabelShift = load_official_lascal_class()

    # The official code expects logits.
    #
    # Because every probability row sums to one:
    #
    #     softmax(log(p)) = p
    #
    # so converting probabilities to log-probabilities gives the
    # official implementation exactly the same probabilities.
    source_logits = torch.log(
        torch.tensor(
            source_probabilities,
            dtype=torch.float64,
        )
    )

    target_logits = torch.log(
        torch.tensor(
            target_probabilities,
            dtype=torch.float64,
        )
    )

    official_labels = torch.tensor(
        source_labels,
        dtype=torch.long,
    )

    official_weights = torch.tensor(
        weights,
        dtype=torch.float64,
    )

    official_estimator = EceLabelShift(
        p=p,
        n_bins=n_bins,
        adaptive_bins=True,
        classwise=True,
    )

    with torch.no_grad():
        official_results = official_estimator(
            logits=target_logits,
            logits_source=source_logits,
            labels_source=official_labels,
            weights=official_weights,
        )

    official_results = (
        official_results
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # --------------------------------------------------
    # Numerical comparison
    # --------------------------------------------------

    absolute_difference = np.abs(
        our_results - official_results
    )

    max_absolute_difference = float(
        np.max(absolute_difference)
    )

    match = np.allclose(
        our_results,
        official_results,
        rtol=1e-7,
        atol=1e-8,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print(
        "=== LASCAL REFERENCE COMPARISON ==="
    )
    print()

    print("Our implementation:")
    print(our_results)
    print()

    print("Official implementation:")
    print(official_results)
    print()

    print("Absolute difference:")
    print(absolute_difference)
    print()

    print(
        "Maximum absolute difference:"
    )
    print(max_absolute_difference)
    print()

    print(
        "Numerically equivalent "
        "(rtol=1e-7, atol=1e-8):"
    )
    print(match)

    # Make failure explicit for reproducibility or CI use.
    if not match:
        raise AssertionError(
            "Our LaSCal estimator does not numerically "
            "match the official implementation within "
            "the specified tolerance."
        )


if __name__ == "__main__":
    main()