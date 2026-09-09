"""Generate research-quality Stage 4 diagnostic figures.

The figures summarize the i.i.d. label-shift baseline before
introducing graph dependence in Stage 5.

Figures:
1. CE estimation error vs label-shift severity.
2. CE estimation error vs source sample size.
3. BBSE target-prior estimation error vs shift severity.
4. Importance-weight estimation error vs source sample size.
5. LaSCal vs supervised target CE under temperature scaling.
6. Classwise LaSCal vs supervised CE under temperature scaling.
7. Paired BBSE-minus-oracle CE error vs shift severity.

Where raw multi-seed values are available, individual seed observations
are shown in addition to mean ± standard deviation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results/stage4")
FIGURES_DIR = Path("figures/stage4")

SHIFT_SUMMARY_PATH = (
    RESULTS_DIR
    / "stage4_multiseed_summary.json"
)

SAMPLE_SIZE_SUMMARY_PATH = (
    RESULTS_DIR
    / "stage4_sample_size_summary.json"
)

TEMPERATURE_RESULTS_PATH = (
    RESULTS_DIR
    / "stage4_temperature_lascal.json"
)


# ============================================================
# General helpers
# ============================================================

def load_json(
    path: Path,
) -> dict:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required results file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_current_figure(
    filename: str,
) -> None:
    """Save the active matplotlib figure."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def deterministic_offsets(
    center: float,
    n_points: int,
    width: float,
) -> np.ndarray:
    """
    Create small deterministic horizontal offsets.

    This lets us display individual seed observations without
    introducing random plotting jitter.
    """

    if n_points <= 1:
        return np.asarray(
            [center],
            dtype=float,
        )

    return (
        center
        + np.linspace(
            -width,
            width,
            n_points,
        )
    )


# ============================================================
# Shift-summary helpers
# ============================================================

def get_shift_conditions(
    summary: dict,
) -> list[dict]:
    """
    Return shift-severity conditions ordered by target class-0 prior.

    The Stage 4 multi-seed summary stores conditions as a dictionary
    whose entries contain a nested "summary" dictionary.
    """

    conditions = summary[
        "conditions"
    ]

    ordered = []

    for key in sorted(
        conditions.keys(),
        key=float,
    ):
        ordered.append(
            conditions[key]
        )

    return ordered


def get_shift_metric(
    condition: dict,
    metric_name: str,
) -> dict:
    """Read one aggregate metric from a shift condition."""

    return condition[
        "summary"
    ][
        metric_name
    ]


# ============================================================
# Sample-size-summary helpers
# ============================================================

def get_sample_size_conditions(
    summary: dict,
) -> list[dict]:
    """
    Return sample-size conditions ordered by source sample size.

    The sample-size summary stores conditions directly as a list.
    """

    return sorted(
        summary[
            "conditions"
        ],
        key=lambda condition: condition[
            "n_source"
        ],
    )


# ============================================================
# Figure 1
# CE estimation error vs shift severity
# ============================================================

def plot_shift_severity_ce_error(
    summary: dict,
) -> None:
    """
    Show label-free CE estimation error as label shift becomes stronger.

    Mean ± standard deviation is shown for BBSE, RLLS, and the
    empirical-prior oracle.

    Individual BBSE seed observations are also shown.
    """

    conditions = get_shift_conditions(
        summary
    )

    x = np.asarray(
        [
            float(
                condition[
                    "target_prior_0"
                ]
            )
            for condition in conditions
        ],
        dtype=float,
    )

    bbse_mean = []
    bbse_std = []

    rlls_mean = []
    rlls_std = []

    oracle_mean = []
    oracle_std = []

    for condition in conditions:
        bbse_metric = get_shift_metric(
            condition,
            "bbse_macro_ce_absolute_error",
        )

        rlls_metric = get_shift_metric(
            condition,
            "rlls_macro_ce_absolute_error",
        )

        oracle_metric = get_shift_metric(
            condition,
            "oracle_macro_ce_absolute_error",
        )

        bbse_mean.append(
            bbse_metric["mean"]
        )

        bbse_std.append(
            bbse_metric["std"]
        )

        rlls_mean.append(
            rlls_metric["mean"]
        )

        rlls_std.append(
            rlls_metric["std"]
        )

        oracle_mean.append(
            oracle_metric["mean"]
        )

        oracle_std.append(
            oracle_metric["std"]
        )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.errorbar(
        x,
        bbse_mean,
        yerr=bbse_std,
        marker="o",
        capsize=4,
        linewidth=2,
        label="BBSE weights",
    )

    plt.errorbar(
        x,
        rlls_mean,
        yerr=rlls_std,
        marker="s",
        capsize=4,
        linewidth=2,
        label="RLLS weights",
    )

    plt.errorbar(
        x,
        oracle_mean,
        yerr=oracle_std,
        marker="^",
        capsize=4,
        linewidth=2,
        label="Empirical-prior oracle weights",
    )

    # Raw BBSE seed observations.
    for x_value, condition in zip(
        x,
        conditions,
    ):
        metric = get_shift_metric(
            condition,
            "bbse_macro_ce_absolute_error",
        )

        values = np.asarray(
            metric[
                "values"
            ],
            dtype=float,
        )

        point_x = deterministic_offsets(
            center=x_value,
            n_points=len(values),
            width=0.008,
        )

        plt.scatter(
            point_x,
            values,
            s=18,
            alpha=0.35,
        )

    plt.xlabel(
        "Target class-0 prior"
    )

    plt.ylabel(
        "Macro absolute classwise CE estimation error"
    )

    plt.title(
        "Stage 4: CE estimation reliability across label-shift severity"
    )

    plt.xticks(
        x
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_shift_severity_ce_error.png"
    )


# ============================================================
# Figure 2
# CE estimation error vs source sample size
# ============================================================

def plot_sample_size_ce_error(
    summary: dict,
) -> None:
    """
    Show finite-sample calibration-estimation reliability.

    BBSE, RLLS, and empirical-prior oracle errors are reported.
    Individual BBSE seed observations are displayed.
    """

    conditions = get_sample_size_conditions(
        summary
    )

    x = np.asarray(
        [
            int(
                condition[
                    "n_source"
                ]
            )
            for condition in conditions
        ],
        dtype=float,
    )

    bbse_mean = []
    bbse_std = []

    rlls_mean = []
    rlls_std = []

    oracle_mean = []
    oracle_std = []

    for condition in conditions:
        bbse_metric = condition[
            "bbse_macro_ce_absolute_error"
        ]

        rlls_metric = condition[
            "rlls_macro_ce_absolute_error"
        ]

        oracle_metric = condition[
            "oracle_macro_ce_absolute_error"
        ]

        bbse_mean.append(
            bbse_metric["mean"]
        )

        bbse_std.append(
            bbse_metric["std"]
        )

        rlls_mean.append(
            rlls_metric["mean"]
        )

        rlls_std.append(
            rlls_metric["std"]
        )

        oracle_mean.append(
            oracle_metric["mean"]
        )

        oracle_std.append(
            oracle_metric["std"]
        )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.errorbar(
        x,
        bbse_mean,
        yerr=bbse_std,
        marker="o",
        capsize=4,
        linewidth=2,
        label="BBSE weights",
    )

    plt.errorbar(
        x,
        rlls_mean,
        yerr=rlls_std,
        marker="s",
        capsize=4,
        linewidth=2,
        label="RLLS weights",
    )

    plt.errorbar(
        x,
        oracle_mean,
        yerr=oracle_std,
        marker="^",
        capsize=4,
        linewidth=2,
        label="Empirical-prior oracle weights",
    )

    for x_value, condition in zip(
        x,
        conditions,
    ):
        values = np.asarray(
            condition[
                "bbse_macro_ce_absolute_error"
            ][
                "values"
            ],
            dtype=float,
        )

        # Offset proportionally because x is logarithmic.
        multipliers = np.linspace(
            0.96,
            1.04,
            len(values),
        )

        plt.scatter(
            x_value * multipliers,
            values,
            s=18,
            alpha=0.35,
        )

    plt.xscale(
        "log",
        base=2,
    )

    plt.xlabel(
        "Source sample size"
    )

    plt.ylabel(
        "Macro absolute classwise CE estimation error"
    )

    plt.title(
        "Stage 4: Finite-sample sensitivity of CE estimation"
    )

    plt.xticks(
        x,
        [
            str(
                int(value)
            )
            for value in x
        ],
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_sample_size_ce_error.png"
    )


# ============================================================
# Figure 3
# Prior-estimation error vs shift severity
# ============================================================

def plot_prior_error_vs_shift(
    summary: dict,
) -> None:
    """
    Diagnose whether stronger label shift is accompanied by larger
    BBSE target-prior estimation error.
    """

    conditions = get_shift_conditions(
        summary
    )

    x = np.asarray(
        [
            float(
                condition[
                    "target_prior_0"
                ]
            )
            for condition in conditions
        ],
        dtype=float,
    )

    means = []
    stds = []

    plt.figure(
        figsize=(8, 5.5)
    )

    for x_value, condition in zip(
        x,
        conditions,
    ):
        metric = get_shift_metric(
            condition,
            "bbse_prior_l2_error",
        )

        means.append(
            metric[
                "mean"
            ]
        )

        stds.append(
            metric[
                "std"
            ]
        )

        values = np.asarray(
            metric[
                "values"
            ],
            dtype=float,
        )

        point_x = deterministic_offsets(
            center=x_value,
            n_points=len(values),
            width=0.008,
        )

        plt.scatter(
            point_x,
            values,
            s=22,
            alpha=0.45,
        )

    plt.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=2,
        label="BBSE prior-estimation error",
    )

    plt.xlabel(
        "Target class-0 prior"
    )

    plt.ylabel(
        "Target-prior L2 estimation error"
    )

    plt.title(
        "Stage 4: BBSE target-prior estimation across shift severity"
    )

    plt.xticks(
        x
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_bbse_prior_error_vs_shift.png"
    )


# ============================================================
# Figure 4
# Weight-estimation error vs sample size
# ============================================================

def plot_weight_error_vs_sample_size(
    summary: dict,
) -> None:
    """
    Diagnose how importance-weight estimation improves as labeled
    source-validation information increases.
    """

    conditions = get_sample_size_conditions(
        summary
    )

    x = np.asarray(
        [
            int(
                condition[
                    "n_source"
                ]
            )
            for condition in conditions
        ],
        dtype=float,
    )

    bbse_mean = []
    bbse_std = []

    rlls_mean = []
    rlls_std = []

    plt.figure(
        figsize=(8, 5.5)
    )

    for x_value, condition in zip(
        x,
        conditions,
    ):
        bbse_metric = condition[
            "bbse_weight_l2_error"
        ]

        rlls_metric = condition[
            "rlls_weight_l2_error"
        ]

        bbse_mean.append(
            bbse_metric[
                "mean"
            ]
        )

        bbse_std.append(
            bbse_metric[
                "std"
            ]
        )

        rlls_mean.append(
            rlls_metric[
                "mean"
            ]
        )

        rlls_std.append(
            rlls_metric[
                "std"
            ]
        )

        values = np.asarray(
            bbse_metric[
                "values"
            ],
            dtype=float,
        )

        multipliers = np.linspace(
            0.96,
            1.04,
            len(values),
        )

        plt.scatter(
            x_value * multipliers,
            values,
            s=20,
            alpha=0.35,
        )

    plt.errorbar(
        x,
        bbse_mean,
        yerr=bbse_std,
        marker="o",
        capsize=4,
        linewidth=2,
        label="BBSE",
    )

    plt.errorbar(
        x,
        rlls_mean,
        yerr=rlls_std,
        marker="s",
        capsize=4,
        linewidth=2,
        label="RLLS",
    )

    plt.xscale(
        "log",
        base=2,
    )

    plt.xlabel(
        "Source sample size"
    )

    plt.ylabel(
        "Importance-weight L2 error"
    )

    plt.title(
        "Stage 4: Importance-weight estimation vs source sample size"
    )

    plt.xticks(
        x,
        [
            str(
                int(value)
            )
            for value in x
        ],
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_weight_error_vs_sample_size.png"
    )


# ============================================================
# Figure 5
# Macro calibration error under temperature scaling
# ============================================================

def plot_temperature_macro_ce(
    temperature_summary: dict,
) -> None:
    """
    Compare label-free LaSCal CE and researcher-side supervised
    target CE under controlled temperature scaling.
    """

    temperature_results = temperature_summary[
        "temperature_results"
    ]

    temperatures = sorted(
        [
            float(
                key
            )
            for key in temperature_results.keys()
        ]
    )

    lascal_macro = []
    supervised_macro = []

    for temperature in temperatures:
        result = temperature_results[
            f"{temperature:.1f}"
        ]

        lascal = np.asarray(
            result[
                "lascal_classwise_ce"
            ],
            dtype=float,
        )

        supervised = np.asarray(
            result[
                "supervised_target_classwise_ce"
            ],
            dtype=float,
        )

        lascal_macro.append(
            float(
                np.mean(
                    lascal
                )
            )
        )

        supervised_macro.append(
            float(
                np.mean(
                    supervised
                )
            )
        )

    plt.figure(
        figsize=(8, 5.5)
    )

    plt.plot(
        temperatures,
        lascal_macro,
        marker="o",
        linewidth=2,
        label="Label-free LaSCal estimate",
    )

    plt.plot(
        temperatures,
        supervised_macro,
        marker="s",
        linewidth=2,
        label="Supervised target CE",
    )

    plt.xlabel(
        "Temperature"
    )

    plt.ylabel(
        "Macro classwise CE"
    )

    plt.title(
        "Stage 4: Calibration under controlled temperature miscalibration"
    )

    plt.xticks(
        temperatures
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_temperature_macro_ce.png"
    )


# ============================================================
# Figure 6
# Classwise CE under temperature scaling
# ============================================================

def plot_temperature_classwise_ce(
    temperature_summary: dict,
) -> None:
    """
    Show classwise LaSCal estimates and supervised target CE.

    This prevents macro averaging from hiding asymmetric behavior
    between classes.
    """

    temperature_results = temperature_summary[
        "temperature_results"
    ]

    temperatures = sorted(
        [
            float(
                key
            )
            for key in temperature_results.keys()
        ]
    )

    first_result = temperature_results[
        f"{temperatures[0]:.1f}"
    ]

    n_classes = len(
        first_result[
            "lascal_classwise_ce"
        ]
    )

    plt.figure(
        figsize=(9, 5.8)
    )

    for class_index in range(
        n_classes
    ):
        lascal_values = []

        supervised_values = []

        for temperature in temperatures:
            result = temperature_results[
                f"{temperature:.1f}"
            ]

            lascal_values.append(
                result[
                    "lascal_classwise_ce"
                ][
                    class_index
                ]
            )

            supervised_values.append(
                result[
                    "supervised_target_classwise_ce"
                ][
                    class_index
                ]
            )

        plt.plot(
            temperatures,
            lascal_values,
            marker="o",
            linewidth=2,
            label=(
                f"LaSCal — class {class_index}"
            ),
        )

        plt.plot(
            temperatures,
            supervised_values,
            marker="x",
            linestyle="--",
            linewidth=2,
            label=(
                f"Supervised — class {class_index}"
            ),
        )

    plt.xlabel(
        "Temperature"
    )

    plt.ylabel(
        "Classwise calibration error"
    )

    plt.title(
        "Stage 4: Classwise calibration behavior under temperature scaling"
    )

    plt.xticks(
        temperatures
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_temperature_classwise_ce.png"
    )


# ============================================================
# Figure 7
# Paired estimated-weight vs oracle diagnostic
# ============================================================

def plot_bbse_minus_oracle_vs_shift(
    summary: dict,
) -> None:
    """
    Plot the paired difference:

        BBSE macro absolute CE error
        minus
        empirical-prior-oracle macro absolute CE error.

    Positive values mean the BBSE-weight run had larger absolute
    CE-estimation error than the empirical-prior-oracle run for
    that seed.

    This is a diagnostic paired difference only. It is NOT an
    additive decomposition of calibration-estimation error.
    """

    conditions = get_shift_conditions(
        summary
    )

    x = np.asarray(
        [
            float(
                condition[
                    "target_prior_0"
                ]
            )
            for condition in conditions
        ],
        dtype=float,
    )

    means = []
    stds = []

    plt.figure(
        figsize=(8, 5.5)
    )

    for x_value, condition in zip(
        x,
        conditions,
    ):
        bbse_values = np.asarray(
            get_shift_metric(
                condition,
                "bbse_macro_ce_absolute_error",
            )[
                "values"
            ],
            dtype=float,
        )

        oracle_values = np.asarray(
            get_shift_metric(
                condition,
                "oracle_macro_ce_absolute_error",
            )[
                "values"
            ],
            dtype=float,
        )

        if (
            bbse_values.shape
            != oracle_values.shape
        ):
            raise ValueError(
                "BBSE and oracle seed arrays have incompatible shapes."
            )

        paired_difference = (
            bbse_values
            - oracle_values
        )

        means.append(
            float(
                np.mean(
                    paired_difference
                )
            )
        )

        stds.append(
            float(
                np.std(
                    paired_difference
                )
            )
        )

        point_x = deterministic_offsets(
            center=x_value,
            n_points=len(
                paired_difference
            ),
            width=0.008,
        )

        plt.scatter(
            point_x,
            paired_difference,
            s=22,
            alpha=0.45,
        )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
    )

    plt.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=4,
        linewidth=2,
        label="BBSE error − oracle error",
    )

    plt.xlabel(
        "Target class-0 prior"
    )

    plt.ylabel(
        "Paired difference in macro absolute CE error"
    )

    plt.title(
        "Stage 4: Estimated-weight vs oracle diagnostic"
    )

    plt.xticks(
        x
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend()

    save_current_figure(
        "stage4_bbse_minus_oracle_vs_shift.png"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Generate all Stage 4 research figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shift_summary = load_json(
        SHIFT_SUMMARY_PATH
    )

    sample_size_summary = load_json(
        SAMPLE_SIZE_SUMMARY_PATH
    )

    temperature_summary = load_json(
        TEMPERATURE_RESULTS_PATH
    )

    print(
        "=== GENERATING STAGE 4 RESEARCH FIGURES ==="
    )
    print()

    plot_shift_severity_ce_error(
        shift_summary
    )

    plot_sample_size_ce_error(
        sample_size_summary
    )

    plot_prior_error_vs_shift(
        shift_summary
    )

    plot_weight_error_vs_sample_size(
        sample_size_summary
    )

    plot_temperature_macro_ce(
        temperature_summary
    )

    plot_temperature_classwise_ce(
        temperature_summary
    )

    plot_bbse_minus_oracle_vs_shift(
        shift_summary
    )

    print()
    print(
        "Generated 7 Stage 4 figures."
    )


if __name__ == "__main__":
    main()