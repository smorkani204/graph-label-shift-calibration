import numpy as np

from abstention.label_shift import RLLSImbalanceAdapter
from src.label_shift.rlls import estimate_rlls_hard_weights


def main():
    source_probabilities = np.array([
        [0.90, 0.10],
        [0.80, 0.20],
        [0.70, 0.30],
        [0.30, 0.70],
        [0.20, 0.80],
        [0.10, 0.90],
    ])

    source_labels = np.array([0, 0, 0, 1, 1, 1])
    source_labels_onehot = np.eye(2)[source_labels]

    target_probabilities = np.array([
        [0.95, 0.05],
        [0.90, 0.10],
        [0.85, 0.15],
        [0.80, 0.20],
        [0.75, 0.25],
        [0.20, 0.80],
    ])

    ours = estimate_rlls_hard_weights(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        target_probabilities=target_probabilities,
    )

    reference = RLLSImbalanceAdapter()(
        source_labels_onehot,
        target_probabilities,
        source_probabilities,
    ).multipliers

    abs_diff = np.abs(ours - reference)
    max_diff = float(np.max(abs_diff))

    equivalent = np.allclose(
        ours,
        reference,
        rtol=1e-6,
        atol=1e-8,
    )

    print("Our RLLS weights      :", ours)
    print("Reference RLLS weights:", reference)
    print("Absolute difference   :", abs_diff)
    print("Maximum difference    :", max_diff)
    print("Reference equivalent  :", equivalent)

    if not equivalent:
        raise AssertionError(
            "Our RLLS implementation does not match the reference "
            "on this controlled test case."
        )


if __name__ == "__main__":
    main()
