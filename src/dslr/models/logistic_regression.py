"""Binary logistic-regression math and batch gradient descent."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from dslr.errors import (
    LogisticRegressionError,
    OptimizationError,
    TrainingConfigurationError,
)


@dataclass(frozen=True)
class BinaryTrainingConfig:
    """Immutable full-batch gradient-descent configuration."""

    learning_rate: float = 0.1
    max_iterations: int = 5000
    tolerance: float = 1e-9
    minimum_iterations: int = 1


@dataclass(frozen=True)
class StochasticTrainingConfig:
    """Immutable stochastic gradient-descent configuration."""

    learning_rate: float = 0.01
    epochs: int = 50
    seed: int = 42


@dataclass(frozen=True)
class MiniBatchTrainingConfig:
    """Immutable mini-batch gradient-descent configuration."""

    learning_rate: float = 0.01
    epochs: int = 50
    seed: int = 42
    batch_size: int = 32


@dataclass(frozen=True)
class BinaryLogisticRegressionResult:
    """Immutable binary logistic-regression training result."""

    weights: tuple[float, ...]
    losses: tuple[float, ...]
    iterations_run: int
    converged: bool
    final_gradient_max_abs: float


def validate_binary_training_config(
    config: BinaryTrainingConfig | None,
) -> BinaryTrainingConfig:
    """Validate and return a binary training configuration."""
    if config is None:
        config = BinaryTrainingConfig()
    if not isinstance(config, BinaryTrainingConfig):
        raise TrainingConfigurationError(
            f"expected BinaryTrainingConfig, got {type(config).__name__}"
        )
    learning_rate = _coerce_positive_float(
        config.learning_rate,
        "learning_rate",
        TrainingConfigurationError,
    )
    max_iterations = _coerce_positive_int(
        config.max_iterations,
        "max_iterations",
        TrainingConfigurationError,
    )
    tolerance = _coerce_nonnegative_float(
        config.tolerance,
        "tolerance",
        TrainingConfigurationError,
    )
    minimum_iterations = _coerce_nonnegative_int(
        config.minimum_iterations,
        "minimum_iterations",
        TrainingConfigurationError,
    )
    if minimum_iterations > max_iterations:
        raise TrainingConfigurationError(
            "minimum_iterations must not exceed max_iterations"
        )
    return BinaryTrainingConfig(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
        minimum_iterations=minimum_iterations,
    )


def validate_stochastic_training_config(
    config: StochasticTrainingConfig | None,
) -> StochasticTrainingConfig:
    """Validate and return an SGD training configuration."""
    if config is None:
        config = StochasticTrainingConfig()
    if not isinstance(config, StochasticTrainingConfig):
        raise TrainingConfigurationError(
            f"expected StochasticTrainingConfig, got {type(config).__name__}"
        )
    learning_rate = _coerce_positive_float(
        config.learning_rate,
        "learning_rate",
        TrainingConfigurationError,
    )
    epochs = _coerce_positive_int(
        config.epochs,
        "epochs",
        TrainingConfigurationError,
    )
    seed = _coerce_int(
        config.seed,
        "seed",
        TrainingConfigurationError,
    )
    return StochasticTrainingConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
    )


def validate_minibatch_training_config(
    config: MiniBatchTrainingConfig | None,
) -> MiniBatchTrainingConfig:
    """Validate and return a mini-batch GD training configuration."""
    if config is None:
        config = MiniBatchTrainingConfig()
    if not isinstance(config, MiniBatchTrainingConfig):
        raise TrainingConfigurationError(
            f"expected MiniBatchTrainingConfig, got {type(config).__name__}"
        )
    learning_rate = _coerce_positive_float(
        config.learning_rate,
        "learning_rate",
        TrainingConfigurationError,
    )
    epochs = _coerce_positive_int(
        config.epochs,
        "epochs",
        TrainingConfigurationError,
    )
    seed = _coerce_int(
        config.seed,
        "seed",
        TrainingConfigurationError,
    )
    batch_size = _coerce_positive_int(
        config.batch_size,
        "batch_size",
        TrainingConfigurationError,
    )
    return MiniBatchTrainingConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
    )


def stable_sigmoid(value) -> float:
    """Return a numerically stable scalar sigmoid value."""
    z = _coerce_finite_float(value, "sigmoid value", LogisticRegressionError)
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    exp_z = math.exp(z)
    return float(exp_z / (1.0 + exp_z))


def validate_design_matrix(matrix) -> np.ndarray:
    """Validate a numeric design matrix and return a float64 copy."""
    rows = _materialize_matrix(matrix)
    if not rows:
        raise LogisticRegressionError("design matrix must contain at least one row")
    expected_columns: int | None = None
    clean_rows: list[list[float]] = []
    for row_index, row in enumerate(rows):
        row_values = list(row)
        if expected_columns is None:
            expected_columns = len(row_values)
            if expected_columns == 0:
                raise LogisticRegressionError(
                    "design matrix rows must contain at least one column"
                )
        elif len(row_values) != expected_columns:
            raise LogisticRegressionError(
                f"design matrix row {row_index} has {len(row_values)} columns; "
                f"expected {expected_columns}"
            )
        clean_rows.append(
            [
                _coerce_finite_float(
                    value,
                    f"matrix[{row_index}][{column_index}]",
                    LogisticRegressionError,
                )
                for column_index, value in enumerate(row_values)
            ]
        )
    return np.array(clean_rows, dtype=np.float64)


def validate_binary_labels(
    labels,
    expected_rows: int | None = None,
    require_both_classes: bool = False,
) -> np.ndarray:
    """Validate binary labels and return a float64 vector."""
    label_values = list(labels)
    if not label_values:
        raise LogisticRegressionError("binary labels must not be empty")
    if expected_rows is not None and len(label_values) != expected_rows:
        raise LogisticRegressionError(
            f"label count {len(label_values)} does not match row count {expected_rows}"
        )
    parsed: list[float] = []
    for index, value in enumerate(label_values):
        if isinstance(value, (bool, np.bool_)):
            raise LogisticRegressionError(f"label {index} must not be bool")
        if not isinstance(value, Real):
            raise LogisticRegressionError(
                f"label {index} must be binary numeric, got {value!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise LogisticRegressionError(
                f"label {index} must be 0 or 1, got {value!r}"
            )
        parsed.append(numeric)
    if require_both_classes:
        has_zero = False
        has_one = False
        for value in parsed:
            if value == 0.0:
                has_zero = True
            elif value == 1.0:
                has_one = True
        if not has_zero or not has_one:
            raise LogisticRegressionError(
                "binary training requires both classes 0 and 1"
            )
    return np.array(parsed, dtype=np.float64)


def validate_weights(weights, expected_columns: int) -> np.ndarray:
    """Validate finite weights and return a float64 vector."""
    if (
        isinstance(expected_columns, bool)
        or not isinstance(expected_columns, int)
        or expected_columns <= 0
    ):
        raise LogisticRegressionError(
            f"expected_columns must be a positive int, got {expected_columns!r}"
        )
    values = list(weights)
    if len(values) != expected_columns:
        raise LogisticRegressionError(
            f"weight count {len(values)} does not match column count {expected_columns}"
        )
    parsed = [
        _coerce_finite_float(value, f"weight[{index}]", LogisticRegressionError)
        for index, value in enumerate(values)
    ]
    return np.array(parsed, dtype=np.float64)


def predict_binary_logits(matrix, weights) -> tuple[float, ...]:
    """Return logits from X @ weights in source row order."""
    design = validate_design_matrix(matrix)
    weight_vector = validate_weights(weights, design.shape[1])
    logits = design @ weight_vector
    if not np.all(np.isfinite(logits)):
        raise OptimizationError("non-finite logits produced by matrix multiplication")
    return _array_to_tuple(logits)


def predict_binary_probabilities(matrix, weights) -> tuple[float, ...]:
    """Return stable binary probabilities in source row order."""
    logits = np.array(predict_binary_logits(matrix, weights), dtype=np.float64)
    probabilities = _sigmoid_array(logits)
    if not np.all(np.isfinite(probabilities)):
        raise OptimizationError("non-finite probabilities produced by sigmoid")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise OptimizationError("probabilities fell outside [0, 1]")
    return _array_to_tuple(probabilities)


def binary_cross_entropy_from_logits(labels, logits) -> float:
    """Return stable average binary cross-entropy from labels and logits."""
    logit_values = _validate_logit_vector(logits)
    label_values = validate_binary_labels(labels, expected_rows=logit_values.size)
    positive_part = np.maximum(logit_values, 0.0)
    losses = positive_part - label_values * logit_values + np.log1p(
        np.exp(-np.abs(logit_values))
    )
    if not np.all(np.isfinite(losses)):
        raise OptimizationError("non-finite cross-entropy loss values")
    loss = float(np.mean(losses))
    if not math.isfinite(loss):
        raise OptimizationError("non-finite cross-entropy loss")
    return loss


def compute_binary_gradient(matrix, labels, weights) -> tuple[float, ...]:
    """Return the full-batch gradient for binary logistic regression."""
    design = validate_design_matrix(matrix)
    label_values = validate_binary_labels(labels, expected_rows=design.shape[0])
    weight_vector = validate_weights(weights, design.shape[1])
    logits = design @ weight_vector
    if not np.all(np.isfinite(logits)):
        raise OptimizationError("non-finite logits while computing gradient")
    probabilities = _sigmoid_array(logits)
    if not np.all(np.isfinite(probabilities)):
        raise OptimizationError("non-finite probabilities while computing gradient")
    residuals = probabilities - label_values
    gradient = (design.T @ residuals) / design.shape[0]
    if not np.all(np.isfinite(gradient)):
        raise OptimizationError("non-finite gradient")
    return _array_to_tuple(gradient)


def predict_binary_labels(
    matrix,
    weights,
    threshold: float = 0.5,
) -> tuple[int, ...]:
    """Return binary labels using probability >= threshold as class 1."""
    threshold_value = _coerce_probability_threshold(threshold)
    probabilities = predict_binary_probabilities(matrix, weights)
    return tuple(1 if value >= threshold_value else 0 for value in probabilities)


def train_binary_logistic_regression(
    matrix,
    labels,
    config: BinaryTrainingConfig | None = None,
    initial_weights=None,
) -> BinaryLogisticRegressionResult:
    """Train a binary logistic model with deterministic full-batch GD."""
    validated_config = validate_binary_training_config(config)
    design = validate_design_matrix(matrix)
    label_values = validate_binary_labels(
        labels,
        expected_rows=design.shape[0],
        require_both_classes=True,
    )
    if initial_weights is None:
        weights = np.zeros(design.shape[1], dtype=np.float64)
    else:
        weights = validate_weights(initial_weights, design.shape[1]).copy()

    losses: list[float] = []
    current_loss = _loss_for_arrays(label_values, design @ weights)
    losses.append(current_loss)
    iterations_run = 0
    converged = False
    final_gradient = np.zeros(design.shape[1], dtype=np.float64)

    for iteration in range(validated_config.max_iterations):
        logits = design @ weights
        if not np.all(np.isfinite(logits)):
            raise OptimizationError("non-finite logits during optimization")
        probabilities = _sigmoid_array(logits)
        if not np.all(np.isfinite(probabilities)):
            raise OptimizationError("non-finite probabilities during optimization")
        residuals = probabilities - label_values
        gradient = (design.T @ residuals) / design.shape[0]
        if not np.all(np.isfinite(gradient)):
            raise OptimizationError("non-finite gradient during optimization")
        final_gradient = gradient

        with np.errstate(over="ignore", invalid="ignore"):
            new_weights = weights - validated_config.learning_rate * gradient
        if not np.all(np.isfinite(new_weights)):
            raise OptimizationError("non-finite weight update during optimization")
        new_loss = _loss_for_arrays(label_values, design @ new_weights)
        losses.append(new_loss)
        iterations_run = iteration + 1

        if (
            validated_config.tolerance > 0.0
            and iterations_run >= validated_config.minimum_iterations
            and abs(losses[-2] - losses[-1]) <= validated_config.tolerance
        ):
            weights = new_weights
            converged = True
            break
        weights = new_weights

    final_gradient_max_abs = 0.0
    if final_gradient.size:
        final_gradient_max_abs = float(np.max(np.abs(final_gradient)))
    if not math.isfinite(final_gradient_max_abs):
        raise OptimizationError("non-finite final gradient magnitude")

    result_losses = tuple(float(value) for value in losses)
    result_weights = _array_to_tuple(weights)
    for value in result_losses:
        if not math.isfinite(value):
            raise OptimizationError("non-finite loss recorded in history")
    return BinaryLogisticRegressionResult(
        weights=result_weights,
        losses=result_losses,
        iterations_run=iterations_run,
        converged=converged,
        final_gradient_max_abs=final_gradient_max_abs,
    )


def train_binary_logistic_regression_sgd(
    matrix,
    labels,
    config: StochasticTrainingConfig | None = None,
    initial_weights=None,
) -> BinaryLogisticRegressionResult:
    """Train a binary logistic model with deterministic single-sample SGD."""
    validated_config = validate_stochastic_training_config(config)
    design = validate_design_matrix(matrix)
    label_values = validate_binary_labels(
        labels,
        expected_rows=design.shape[0],
        require_both_classes=True,
    )
    if initial_weights is None:
        weights = np.zeros(design.shape[1], dtype=np.float64)
    else:
        weights = validate_weights(initial_weights, design.shape[1]).copy()

    losses: list[float] = [_loss_for_arrays(label_values, design @ weights)]
    indices = list(range(design.shape[0]))
    rng = random.Random(validated_config.seed)

    for _ in range(validated_config.epochs):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for row_index in shuffled:
            row = design[row_index]
            label = label_values[row_index]
            logit = float(row @ weights)
            if not math.isfinite(logit):
                raise OptimizationError("non-finite logit during SGD optimization")
            probability = stable_sigmoid(logit)
            error = probability - label
            with np.errstate(over="ignore", invalid="ignore"):
                new_weights = weights - validated_config.learning_rate * error * row
            if not np.all(np.isfinite(new_weights)):
                raise OptimizationError("non-finite SGD weight update")
            weights = new_weights
        losses.append(_loss_for_arrays(label_values, design @ weights))

    gradient = np.array(
        compute_binary_gradient(design, label_values, weights),
        dtype=np.float64,
    )
    final_gradient_max_abs = 0.0
    if gradient.size:
        final_gradient_max_abs = float(np.max(np.abs(gradient)))
    if not math.isfinite(final_gradient_max_abs):
        raise OptimizationError("non-finite final gradient magnitude")

    result_losses = tuple(float(value) for value in losses)
    for value in result_losses:
        if not math.isfinite(value):
            raise OptimizationError("non-finite loss recorded in history")
    return BinaryLogisticRegressionResult(
        weights=_array_to_tuple(weights),
        losses=result_losses,
        iterations_run=validated_config.epochs,
        converged=False,
        final_gradient_max_abs=final_gradient_max_abs,
    )


def train_binary_logistic_regression_minibatch(
    matrix,
    labels,
    config: MiniBatchTrainingConfig | None = None,
    initial_weights=None,
) -> BinaryLogisticRegressionResult:
    """Train a binary logistic model with deterministic mini-batch GD."""
    validated_config = validate_minibatch_training_config(config)
    design = validate_design_matrix(matrix)
    label_values = validate_binary_labels(
        labels,
        expected_rows=design.shape[0],
        require_both_classes=True,
    )
    if initial_weights is None:
        weights = np.zeros(design.shape[1], dtype=np.float64)
    else:
        weights = validate_weights(initial_weights, design.shape[1]).copy()

    losses: list[float] = [_loss_for_arrays(label_values, design @ weights)]
    indices = list(range(design.shape[0]))
    rng = random.Random(validated_config.seed)

    for _ in range(validated_config.epochs):
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), validated_config.batch_size):
            batch_indices = shuffled[start : start + validated_config.batch_size]
            batch_design = design[batch_indices, :]
            batch_labels = label_values[batch_indices]
            logits = batch_design @ weights
            if not np.all(np.isfinite(logits)):
                raise OptimizationError("non-finite logits during mini-batch optimization")
            probabilities = _sigmoid_array(logits)
            if not np.all(np.isfinite(probabilities)):
                raise OptimizationError(
                    "non-finite probabilities during mini-batch optimization"
                )
            residuals = probabilities - batch_labels
            gradient = (batch_design.T @ residuals) / batch_design.shape[0]
            if not np.all(np.isfinite(gradient)):
                raise OptimizationError("non-finite mini-batch gradient")
            with np.errstate(over="ignore", invalid="ignore"):
                new_weights = weights - validated_config.learning_rate * gradient
            if not np.all(np.isfinite(new_weights)):
                raise OptimizationError("non-finite mini-batch weight update")
            weights = new_weights
        losses.append(_loss_for_arrays(label_values, design @ weights))

    gradient = np.array(
        compute_binary_gradient(design, label_values, weights),
        dtype=np.float64,
    )
    final_gradient_max_abs = 0.0
    if gradient.size:
        final_gradient_max_abs = float(np.max(np.abs(gradient)))
    if not math.isfinite(final_gradient_max_abs):
        raise OptimizationError("non-finite final gradient magnitude")

    result_losses = tuple(float(value) for value in losses)
    for value in result_losses:
        if not math.isfinite(value):
            raise OptimizationError("non-finite loss recorded in history")
    return BinaryLogisticRegressionResult(
        weights=_array_to_tuple(weights),
        losses=result_losses,
        iterations_run=validated_config.epochs,
        converged=False,
        final_gradient_max_abs=final_gradient_max_abs,
    )


def _materialize_matrix(matrix) -> list:
    if isinstance(matrix, np.ndarray):
        if matrix.ndim != 2:
            raise LogisticRegressionError("design matrix must be two-dimensional")
        return matrix.tolist()
    return list(matrix)


def _validate_logit_vector(logits) -> np.ndarray:
    values = list(logits)
    if not values:
        raise LogisticRegressionError("logits must not be empty")
    parsed = [
        _coerce_finite_float(value, f"logit[{index}]", LogisticRegressionError)
        for index, value in enumerate(values)
    ]
    return np.array(parsed, dtype=np.float64)


def _loss_for_arrays(labels: np.ndarray, logits: np.ndarray) -> float:
    if not np.all(np.isfinite(logits)):
        raise OptimizationError("non-finite logits while computing loss")
    positive_part = np.maximum(logits, 0.0)
    losses = positive_part - labels * logits + np.log1p(np.exp(-np.abs(logits)))
    if not np.all(np.isfinite(losses)):
        raise OptimizationError("non-finite loss values")
    loss = float(np.mean(losses))
    if not math.isfinite(loss):
        raise OptimizationError("non-finite loss")
    return loss


def _sigmoid_array(logits: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(logits, dtype=np.float64)
    positive_mask = logits >= 0.0
    probabilities[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_values = np.exp(logits[~positive_mask])
    probabilities[~positive_mask] = exp_values / (1.0 + exp_values)
    return probabilities


def _coerce_probability_threshold(value) -> float:
    threshold = _coerce_finite_float(
        value,
        "threshold",
        LogisticRegressionError,
    )
    if threshold < 0.0 or threshold > 1.0:
        raise LogisticRegressionError(f"threshold must be in [0, 1], got {value!r}")
    return threshold


def _coerce_positive_float(value, label: str, error_type: type[Exception]) -> float:
    parsed = _coerce_finite_float(value, label, error_type)
    if parsed <= 0.0:
        raise error_type(f"{label} must be positive, got {value!r}")
    return parsed


def _coerce_nonnegative_float(value, label: str, error_type: type[Exception]) -> float:
    parsed = _coerce_finite_float(value, label, error_type)
    if parsed < 0.0:
        raise error_type(f"{label} must be nonnegative, got {value!r}")
    return parsed


def _coerce_positive_int(value, label: str, error_type: type[Exception]) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise error_type(f"{label} must be a positive int, got {value!r}")
    if value <= 0:
        raise error_type(f"{label} must be positive, got {value!r}")
    return value


def _coerce_nonnegative_int(value, label: str, error_type: type[Exception]) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise error_type(f"{label} must be a nonnegative int, got {value!r}")
    if value < 0:
        raise error_type(f"{label} must be nonnegative, got {value!r}")
    return value


def _coerce_int(value, label: str, error_type: type[Exception]) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise error_type(f"{label} must be an int, got {value!r}")
    return value


def _coerce_finite_float(value, label: str, error_type: type[Exception]) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise error_type(f"{label} must be a finite real number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise error_type(f"{label} must be finite, got {value!r}")
    return parsed


def _array_to_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values.tolist())
