"""Run Stage 4 baseline experiments across multiple seeds and aggregate results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


SEEDS = list(range(1, 11))
TARGET_PRIOR_0_VALUES = [0.5, 0.6, 0.7, 0.8]

N_SOURCE = 2000
N_TARGET = 2000

RESULTS_DIR = Path("results/stage4")
SUMMARY_PATH = RESULTS_DIR / "stage4_multiseed_summary.json"


def run_single_experiment(
    target_prior_0: float,
    seed: int,
) -> Path:
    """Run one Stage 4 baseline condition."""

    command = [
        sys.executable,
        "scripts/run_stage4_bbse_baseline.py",
        "--target-prior-0",
        str(target_prior_0),
        "--n-source",
        str(N_SOURCE),
        "--n-target",
        str(N_TARGET),
        "--seed",
        str(seed),
    ]

    print(
        f"Running target_prior_0={target_prior_0}, "
        f"n_source={N_SOURCE}, "
        f"n_target={N_TARGET}, "
        f"seed={seed}"
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."

    subprocess.run(
        command,
        check=True,
        env=environment,
    )

    prior_tag = (
        f"{target_prior_0:.2f}"
        .replace(".", "p")
    )

    result_path = (
        RESULTS_DIR
        / (
            f"baseline_target_prior0_{prior_tag}"
            f"_ns{N_SOURCE}"
            f"_nt{N_TARGET}"
            f"_seed{seed}.json"
        )
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"Expected result file not found: {result_path}"
        )

    return result_path


def load_result(
    result_path: Path,
) -> dict:
    """Load one baseline result JSON file."""

    with result_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def summarize_metric(
    values: list[float],
) -> dict:
    """Compute summary statistics while retaining raw values."""

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(np.mean(array)),
        "std": float(
            np.std(
                array,
                ddof=1,
            )
        ),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values": array.tolist(),
    }


def summarize_condition(
    results: list[dict],
) -> dict:
    """Aggregate metrics for one target-prior condition."""

    bbse_macro_errors = [
        result[
            "bbse_macro_ce_absolute_error"
        ]
        for result in results
    ]

    rlls_macro_errors = [
        result[
            "rlls_macro_ce_absolute_error"
        ]
        for result in results
    ]

    oracle_macro_errors = [
        result[
            "oracle_macro_ce_absolute_error"
        ]
        for result in results
    ]

    bbse_prior_errors = [
        result[
            "bbse_prior_l2_error"
        ]
        for result in results
    ]

    bbse_weight_errors = [
        result[
            "bbse_weight_l2_error"
        ]
        for result in results
    ]

    rlls_weight_errors = [
        result[
            "rlls_weight_l2_error"
        ]
        for result in results
    ]

    condition_numbers = [
        result[
            "confusion_matrix_condition_number"
        ]
        for result in results
    ]

    target_ece_values = [
        result[
            "target_ece"
        ]
        for result in results
    ]

    target_nll_values = [
        result[
            "target_nll"
        ]
        for result in results
    ]

    bbse_valid_values = [
    bool(
        result[
            "bbse_estimated_priors_valid"
        ]
    )
    for result in results
    ]

    bbse_valid_count = int(
        sum(bbse_valid_values)
    )

    bbse_invalid_count = int(
        len(bbse_valid_values)
        - bbse_valid_count
    )

    return {
        "n_runs": len(results),
        "bbse_valid_count": bbse_valid_count,
        "bbse_invalid_count": bbse_invalid_count,
        "bbse_valid_fraction": float(
            bbse_valid_count
            / len(results)
        ),
        "bbse_valid_values": (
            bbse_valid_values
        ),
        "bbse_macro_ce_absolute_error": (
            summarize_metric(
                bbse_macro_errors
            )
        ),
        "rlls_macro_ce_absolute_error": (
            summarize_metric(
                rlls_macro_errors
            )
        ),
        "oracle_macro_ce_absolute_error": (
            summarize_metric(
                oracle_macro_errors
            )
        ),
        "bbse_prior_l2_error": (
            summarize_metric(
                bbse_prior_errors
            )
        ),
        "bbse_weight_l2_error": (
            summarize_metric(
                bbse_weight_errors
            )
        ),
        "rlls_weight_l2_error": (
            summarize_metric(
                rlls_weight_errors
            )
        ),
        "confusion_matrix_condition_number": (
            summarize_metric(
                condition_numbers
            )
        ),
        "target_ece": (
            summarize_metric(
                target_ece_values
            )
        ),
        "target_nll": (
            summarize_metric(
                target_nll_values
            )
        ),
    }


def main() -> None:
    """Run all multi-seed shift-severity conditions."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = {}

    for target_prior_0 in TARGET_PRIOR_0_VALUES:
        condition_results = []

        for seed in SEEDS:
            result_path = run_single_experiment(
                target_prior_0=target_prior_0,
                seed=seed,
            )

            result = load_result(
                result_path
            )

            condition_results.append(
                result
            )

        prior_key = (
            f"{target_prior_0:.2f}"
        )

        all_summaries[
            prior_key
        ] = {
            "target_prior_0": (
                target_prior_0
            ),
            "target_prior_1": (
                1.0 - target_prior_0
            ),
            "n_source": N_SOURCE,
            "n_target": N_TARGET,
            "seeds": SEEDS,
            "summary": summarize_condition(
                condition_results
            ),
        }

    final_summary = {
        "experiment": (
            "stage4_multiseed_baseline"
        ),
        "target_prior_0_values": (
            TARGET_PRIOR_0_VALUES
        ),
        "n_source": N_SOURCE,
        "n_target": N_TARGET,
        "seeds": SEEDS,
        "conditions": (
            all_summaries
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_summary,
            file,
            indent=2,
        )

    print()
    print(
        "=== STAGE 4 MULTI-SEED SUMMARY ==="
    )
    print()

    for condition in all_summaries.values():
        summary = condition[
            "summary"
        ]

        print(
            "Target priors: "
            f"[{condition['target_prior_0']:.2f}, "
            f"{condition['target_prior_1']:.2f}]"
        )

        print(
            "BBSE valid runs:"
        )

        print(
            f"{summary['bbse_valid_count']}"
            f"/{summary['n_runs']}"
        )

        print(
            "BBSE macro CE error:"
        )

        print(
            f"{summary['bbse_macro_ce_absolute_error']['mean']:.6f} "
            f"± "
            f"{summary['bbse_macro_ce_absolute_error']['std']:.6f}"
        )

        print(
            "RLLS macro CE error:"
        )

        print(
            f"{summary['rlls_macro_ce_absolute_error']['mean']:.6f} "
            f"± "
            f"{summary['rlls_macro_ce_absolute_error']['std']:.6f}"
        )

        print(
            "Oracle macro CE error:"
        )

        print(
            f"{summary['oracle_macro_ce_absolute_error']['mean']:.6f} "
            f"± "
            f"{summary['oracle_macro_ce_absolute_error']['std']:.6f}"
        )

        print(
            "BBSE prior L2 error:"
        )

        print(
            f"{summary['bbse_prior_l2_error']['mean']:.6f} "
            f"± "
            f"{summary['bbse_prior_l2_error']['std']:.6f}"
        )

        print()

    print(
        "Saved summary to:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()