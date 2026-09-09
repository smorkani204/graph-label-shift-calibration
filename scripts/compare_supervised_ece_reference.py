"""Numerically compare our supervised classwise CE with official LaSCal Ece."""

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
    compute_supervised_classwise_error,
)


def load_official_ece_class():
    """
    Load Ece directly from the official LaSCal source file.
    """
    official_file = Path(
        "../label-shift-calibration-main/"
        "src/lascal/calibration_error/ece.py"
    )

    if not official_file.exists():
        raise FileNotFoundError(
            "Official LaSCal Ece source file was not found at:\n"
            f"{official_file.resolve()}"
        )

    spec = importlib.util.spec_from_file_location(
        "official_ece",
        official_file,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Could not create an import specification "
            "for the official Ece file."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.Ece


def main() -> None:
    """
    Compare our supervised classwise calibration error with the
    official LaSCal supervised Ece implementation.
    """

    # --------------------------------------------------
    # Controlled probabilities and target labels
    # --------------------------------------------------

    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.40, 0.60],
            [0.30, 0.70],
            [0.20, 0.80],
            [0.10, 0.90],
        ],
        dtype=np.float64,
    )

    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=np.int64,
    )

    p = 2
    n_bins = 2

    # --------------------------------------------------
    # Our implementation
    # --------------------------------------------------

    our_results = []

    for class_index in range(2):
        class_result = compute_supervised_classwise_error(
            class_probabilities=probabilities[:, class_index],
            class_indicators=(labels == class_index),
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

    Ece = load_official_ece_class()

    # The official implementation expects logits.
    #
    # Since each probability row sums to one:
    #
    #     softmax(log(p)) = p
    #
    # so log-probabilities preserve exactly the probabilities
    # used by our implementation.
    logits = torch.log(
        torch.tensor(
            probabilities,
            dtype=torch.float64,
        )
    )

    official_labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    official_estimator = Ece(
        p=p,
        n_bins=n_bins,
        version="our",
        adaptive_bins=True,
        classwise=True,
    )

    with torch.no_grad():
        official_results = official_estimator(
            logits=logits,
            labels=official_labels,
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

    print("=== SUPERVISED ECE REFERENCE COMPARISON ===")
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

    print("Maximum absolute difference:")
    print(max_absolute_difference)
    print()

    print(
        "Numerically equivalent "
        "(rtol=1e-7, atol=1e-8):"
    )
    print(match)

    if not match:
        raise AssertionError(
            "Our supervised classwise calibration error does not "
            "numerically match the official LaSCal Ece "
            "implementation within the specified tolerance."
        )


if __name__ == "__main__":
    main()