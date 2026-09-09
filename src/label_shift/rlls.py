import numpy as np


def compute_rho(n_class: int, n_train: int, delta: float = 0.05) -> float:
    """
    Compute the RLLS concentration term used in the reference implementation.

    This matches the compute_3deltaC function used by the original RLLS code
    and by abstention's RLLSImbalanceAdapter.

    Parameters
    ----------
    n_class : int
        Number of classes.
    n_train : int
        Number of source/validation samples used to estimate the moments.
    delta : float, default=0.05
        Confidence parameter.

    Returns
    -------
    float
        The RLLS rho value.
    """
    if n_class <= 0:
        raise ValueError("n_class must be positive.")
    if n_train <= 0:
        raise ValueError("n_train must be positive.")
    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1.")

    rho = 3 * (
        2 * np.log(2 * n_class / delta) / (3 * n_train)
        + np.sqrt(2 * np.log(2 * n_class / delta) / n_train)
    )

    return float(rho)


def compute_rlls_weights(
    C_yy: np.ndarray,
    mu_y: np.ndarray,
    mu_train_y: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    Solve the regularized RLLS optimization problem.

    This follows the compute_w_opt formulation used in the original
    RLLS reference implementation and abstention.

    Parameters
    ----------
    C_yy : np.ndarray
        Joint source prediction/label moment matrix.
    mu_y : np.ndarray
        Target prediction-frequency vector.
    mu_train_y : np.ndarray
        Source prediction-frequency vector.
    rho : float
        Final regularization coefficient supplied to the optimization.

    Returns
    -------
    np.ndarray
        Estimated non-negative importance weights.
    """
    import cvxpy as cp

    C_yy = np.asarray(C_yy, dtype=float)
    mu_y = np.asarray(mu_y, dtype=float)
    mu_train_y = np.asarray(mu_train_y, dtype=float)

    if C_yy.ndim != 2 or C_yy.shape[0] != C_yy.shape[1]:
        raise ValueError("C_yy must be a square matrix.")

    n_class = C_yy.shape[0]

    if mu_y.shape != (n_class,):
        raise ValueError("mu_y must have shape (n_class,).")

    if mu_train_y.shape != (n_class,):
        raise ValueError("mu_train_y must have shape (n_class,).")

    if rho < 0:
        raise ValueError("rho must be non-negative.")

    theta = cp.Variable(n_class)
    b = mu_y - mu_train_y

    objective = cp.Minimize(
        cp.pnorm(C_yy @ theta - b)
        + rho * cp.pnorm(theta)
    )

    constraints = [theta >= -1]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(
            f"RLLS optimization failed with status: {problem.status}"
        )

    if theta.value is None:
        raise RuntimeError("RLLS optimization returned no solution.")

    weights = 1.0 + np.asarray(theta.value, dtype=float)

    return weights


def compute_hard_rlls_moments(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    target_probabilities: np.ndarray,
):
    """
    Construct the empirical moments used by reference RLLS-hard.

    Returns
    -------
    C_yy : np.ndarray
        Joint source prediction/label matrix where
        C_yy[i, j] = P_hat(source prediction=i, source label=j).

    mu_target : np.ndarray
        Target hard-prediction frequency vector.

    mu_source : np.ndarray
        Source hard-prediction frequency vector.
    """
    source_probabilities = np.asarray(source_probabilities, dtype=float)
    target_probabilities = np.asarray(target_probabilities, dtype=float)
    source_labels = np.asarray(source_labels)

    if source_probabilities.ndim != 2:
        raise ValueError("source_probabilities must be a 2D array.")

    if target_probabilities.ndim != 2:
        raise ValueError("target_probabilities must be a 2D array.")

    n_class = source_probabilities.shape[1]

    if target_probabilities.shape[1] != n_class:
        raise ValueError(
            "Source and target probabilities must have the same number of classes."
        )

    if source_labels.shape != (len(source_probabilities),):
        raise ValueError(
            "source_labels must contain one label per source sample."
        )

    if np.any(source_labels < 0) or np.any(source_labels >= n_class):
        raise ValueError("source_labels contain an invalid class index.")

    source_predictions = np.argmax(source_probabilities, axis=1)
    target_predictions = np.argmax(target_probabilities, axis=1)

    source_pred_one_hot = np.eye(n_class)[source_predictions]
    target_pred_one_hot = np.eye(n_class)[target_predictions]
    source_label_one_hot = np.eye(n_class)[source_labels.astype(int)]

    mu_target = np.mean(target_pred_one_hot, axis=0)
    mu_source = np.mean(source_pred_one_hot, axis=0)

    C_yy = np.mean(
        source_pred_one_hot[:, :, None]
        * source_label_one_hot[:, None, :],
        axis=0,
    )

    return C_yy, mu_target, mu_source


def estimate_rlls_hard_weights(
    source_probabilities: np.ndarray,
    source_labels: np.ndarray,
    target_probabilities: np.ndarray,
    alpha: float = 0.01,
    delta: float = 0.05,
) -> np.ndarray:
    """
    Estimate class importance weights using the reference RLLS-hard pipeline.

    This mirrors the sequence used by abstention's RLLSImbalanceAdapter:
    1. convert probabilities to hard predictions,
    2. construct source joint prediction/label moments,
    3. compute source and target prediction frequencies,
    4. compute the RLLS concentration term,
    5. solve the regularized constrained optimization.

    Parameters
    ----------
    source_probabilities : np.ndarray
        Source/validation posterior probabilities with shape
        (n_source, n_class).

    source_labels : np.ndarray
        Source/validation class labels with shape (n_source,).

    target_probabilities : np.ndarray
        Unlabeled target posterior probabilities with shape
        (n_target, n_class).

    alpha : float, default=0.01
        Reference regularization multiplier used by the original
        implementation and abstention.

    delta : float, default=0.05
        Confidence parameter used in the RLLS concentration term.

    Returns
    -------
    np.ndarray
        Estimated class importance weights.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    C_yy, mu_target, mu_source = compute_hard_rlls_moments(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        target_probabilities=target_probabilities,
    )

    rho = compute_rho(
        n_class=C_yy.shape[0],
        n_train=len(source_probabilities),
        delta=delta,
    )

    return compute_rlls_weights(
        C_yy=C_yy,
        mu_y=mu_target,
        mu_train_y=mu_source,
        rho=alpha * rho,
    )
