"""Run Stage 4 source-sample-size sensitivity experiments."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np


TARGET_PRIOR_0 = 0.7
N_TARGET = 2000

SOURCE_SAMPLE_SIZES = [
    500,
    1000,
    2000,
    4000,
    8000,
]

SEEDS = list(range(1, 11))

RESULTS_DIR = Path("results/stage4")
SUMMARY_PATH = RESULTS_DIR / "stage4_sample_size_summary.json"


def run_experiment(
    n_source: int,
    seed: int,
) -> Path:
    command = [
        "python",
        "scripts/run_stage4_bbse_baseline.py",
        "--target-prior-0",
        str(TARGET_PRIOR_0),
        "--n-source",
        str(n_source),
        "--n-target",
        str(N_TARGET),
        "--seed",
        str(seed),
    ]

    environment = {
        **os.environ,
        "PYTHONPATH": ".",
    }

    print()
    print(
        f"Running n_source={n_source}, "
        f"n_target={N_TARGET}, "
        f"seed={seed}"
    )

    subprocess.run(
        command,
        check=True,
        env=environment,
    )

    prior_tag = (
        f"{TARGET_PRIOR_0:.2f}"
        .replace(".", "p")
    )

    result_path = (
        RESULTS_DIR
        / (
            f"baseline_target_prior0_{prior_tag}_"
            f"ns{n_source}_"
            f"nt{N_TARGET}_"
            f"seed{seed}.json"
        )
    )

    if not result_path.exists():
        raise FileNotFoundError(
            f"Expected result file was not created: "
            f"{result_path}"
        )

    return result_path


def load_result(
    result_path: Path,
) -> dict:
    with result_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def summarize_metric(
    results: list[dict],
    key: str,
) -> dict:
    values = np.asarray(
        [
            float(result[key])
            for result in results
        ],
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(
                values,
                ddof=1,
            )
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
        "values": values.tolist(),
    }


def summarize_condition(
    n_source: int,
    results: list[dict],
) -> dict:
    validation_sizes = np.asarray(
        [
            int(result["n_source_validation"])
            for result in results
        ],
        dtype=int,
    )

    bbse_valid_flags = [
        bool(
            result[
                "bbse_estimated_priors_valid"
            ]
        )
        for result in results
    ]

    return {
        "n_source": int(n_source),
        "n_target": int(N_TARGET),

        "source_validation_size_mean": float(
            np.mean(validation_sizes)
        ),

        "source_validation_sizes": (
            validation_sizes.tolist()
        ),

        "bbse_valid_count": int(
            sum(bbse_valid_flags)
        ),

        "n_runs": int(
            len(results)
        ),

        "bbse_macro_ce_absolute_error": (
            summarize_metric(
                results,
                "bbse_macro_ce_absolute_error",
            )
        ),

        "rlls_macro_ce_absolute_error": (
            summarize_metric(
                results,
                "rlls_macro_ce_absolute_error",
            )
        ),

        "oracle_macro_ce_absolute_error": (
            summarize_metric(
                results,
                "oracle_macro_ce_absolute_error",
            )
        ),

        "bbse_prior_l2_error": (
            summarize_metric(
                results,
                "bbse_prior_l2_error",
            )
        ),

        "bbse_weight_l2_error": (
            summarize_metric(
                results,
                "bbse_weight_l2_error",
            )
        ),

        "rlls_weight_l2_error": (
            summarize_metric(
                results,
                "rlls_weight_l2_error",
            )
        ),

        "confusion_matrix_condition_number": (
            summarize_metric(
                results,
                (
                    "confusion_matrix_"
                    "condition_number"
                ),
            )
        ),

        "target_ece": (
            summarize_metric(
                results,
                "target_ece",
            )
        ),

        "target_nll": (
            summarize_metric(
                results,
                "target_nll",
            )
        ),
    }


def print_summary(
    summary: dict,
) -> None:
    print()
    print(
        "=== STAGE 4 SAMPLE-SIZE SUMMARY ==="
    )

    for condition in summary["conditions"]:
        print()
        print(
            f"Source sample size: "
            f"{condition['n_source']}"
        )

        print(
            f"Mean validation size: "
            f"{condition['source_validation_size_mean']:.1f}"
        )

        print(
            f"BBSE valid runs: "
            f"{condition['bbse_valid_count']}"
            f"/{condition['n_runs']}"
        )

        bbse = condition[
            "bbse_macro_ce_absolute_error"
        ]

        rlls = condition[
            "rlls_macro_ce_absolute_error"
        ]

        oracle = condition[
            "oracle_macro_ce_absolute_error"
        ]

        prior = condition[
            "bbse_prior_l2_error"
        ]

        weight = condition[
            "bbse_weight_l2_error"
        ]

        print()
        print(
            "BBSE macro CE error:"
        )
        print(
            f"{bbse['mean']:.6f} "
            f"± {bbse['std']:.6f}"
        )

        print()
        print(
            "RLLS macro CE error:"
        )
        print(
            f"{rlls['mean']:.6f} "
            f"± {rlls['std']:.6f}"
        )

        print()
        print(
            "Oracle macro CE error:"
        )
        print(
            f"{oracle['mean']:.6f} "
            f"± {oracle['std']:.6f}"
        )

        print()
        print(
            "BBSE prior L2 error:"
        )
        print(
            f"{prior['mean']:.6f} "
            f"± {prior['std']:.6f}"
        )

        print()
        print(
            "BBSE weight L2 error:"
        )
        print(
            f"{weight['mean']:.6f} "
            f"± {weight['std']:.6f}"
        )


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_conditions = []

    for n_source in SOURCE_SAMPLE_SIZES:
        condition_results = []

        for seed in SEEDS:
            result_path = run_experiment(
                n_source=n_source,
                seed=seed,
            )

            result = load_result(
                result_path
            )

            condition_results.append(
                result
            )

        condition_summary = (
            summarize_condition(
                n_source=n_source,
                results=condition_results,
            )
        )

        all_conditions.append(
            condition_summary
        )

    summary = {
        "experiment": (
            "stage4_source_sample_size_sensitivity"
        ),

        "target_prior_0": float(
            TARGET_PRIOR_0
        ),

        "target_priors": [
            float(TARGET_PRIOR_0),
            float(1.0 - TARGET_PRIOR_0),
        ],

        "n_target": int(
            N_TARGET
        ),

        "source_sample_sizes": (
            SOURCE_SAMPLE_SIZES
        ),

        "seeds": (
            SEEDS
        ),

        "n_seeds": int(
            len(SEEDS)
        ),

        "conditions": (
            all_conditions
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print_summary(
        summary
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