"""One-vs-rest multiclass foundation for Hogwarts house classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dslr.data.csv_loader import Dataset
from dslr.errors import OneVsRestError, OneVsRestStateError
from dslr.data.houses import CANONICAL_HOUSE_ORDER, HOUSE_COLUMN, extract_house_labels
from dslr.models.logistic_regression import (
    BinaryLogisticRegressionResult,
    BinaryTrainingConfig,
    MiniBatchTrainingConfig,
    StochasticTrainingConfig,
    predict_binary_probabilities,
    train_binary_logistic_regression,
    train_binary_logistic_regression_minibatch,
    train_binary_logistic_regression_sgd,
    validate_binary_training_config,
    validate_minibatch_training_config,
    validate_stochastic_training_config,
    validate_weights,
)
from dslr.data.preprocessing import (
    PreprocessingState,
    fit_preprocessor,
    preprocessing_state_from_dict,
    preprocessing_state_to_dict,
    transform_dataset,
    validate_preprocessing_state,
)

ONE_VS_REST_SCHEMA_VERSION = 1
CANONICAL_CLASS_ORDER = CANONICAL_HOUSE_ORDER


@dataclass(frozen=True)
class BinaryClassifierState:
    """Immutable state for one target-vs-rest binary classifier."""

    class_label: str
    weights: tuple[float, ...]
    iterations_run: int
    initial_loss: float
    final_loss: float
    converged: bool
    final_gradient_max_abs: float


@dataclass(frozen=True)
class OneVsRestTrainingMetadata:
    """Immutable metadata for deterministic one-vs-rest training."""

    row_count: int
    feature_count: int
    matrix_column_count: int
    class_order: tuple[str, ...]
    training_config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig
    optimizer: str = "batch"


@dataclass(frozen=True)
class OneVsRestModelState:
    """Immutable reusable one-vs-rest model state."""

    schema_version: int
    class_order: tuple[str, ...]
    preprocessing_state: PreprocessingState
    classifiers: tuple[BinaryClassifierState, ...]
    training_config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig
    metadata: OneVsRestTrainingMetadata


def make_one_vs_rest_targets(
    labels,
    positive_class: str,
) -> tuple[int, ...]:
    """Return binary targets for one positive class and all others as zero."""
    if not isinstance(positive_class, str) or positive_class == "":
        raise OneVsRestError(f"positive_class must be a non-empty string, got {positive_class!r}")
    target_values: list[int] = []
    saw_positive = False
    saw_negative = False
    for index, label in enumerate(labels):
        if not isinstance(label, str) or label == "":
            raise OneVsRestError(f"label {index} must be a non-empty string, got {label!r}")
        if label == positive_class:
            target_values.append(1)
            saw_positive = True
        else:
            target_values.append(0)
            saw_negative = True
    if not target_values:
        raise OneVsRestError("cannot build one-vs-rest targets from empty labels")
    if not saw_positive:
        raise OneVsRestError(f"positive class {positive_class!r} is absent from labels")
    if not saw_negative:
        raise OneVsRestError(
            f"positive class {positive_class!r} has no negative examples"
        )
    return tuple(target_values)


def extract_training_labels(dataset: Dataset) -> tuple[str, ...]:
    """Return row-order house labels after validating canonical training classes."""
    labels = extract_house_labels(dataset, minimum_houses=len(CANONICAL_CLASS_ORDER))
    if labels != CANONICAL_CLASS_ORDER:
        raise OneVsRestError(
            "training data must contain exactly the canonical Hogwarts houses "
            f"in canonical order; found {labels!r}"
        )
    return tuple(row[HOUSE_COLUMN] for row in dataset.rows)


def train_one_vs_rest_classifier(
    dataset: Dataset,
    config: BinaryTrainingConfig | None = None,
    features=None,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> OneVsRestModelState:
    """Fit preprocessing and train one binary classifier per canonical house."""
    optimizer_name = _validate_optimizer(optimizer)
    if optimizer_name == "batch":
        training_config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig = (
            validate_binary_training_config(config)
        )
    elif optimizer_name == "sgd":
        training_config = validate_stochastic_training_config(sgd_config)
    else:
        training_config = validate_minibatch_training_config(minibatch_config)
    labels = extract_training_labels(dataset)
    preprocessing_state = fit_preprocessor(dataset, features=features, add_bias=True)
    matrix = transform_dataset(dataset, preprocessing_state)

    classifiers: list[BinaryClassifierState] = []
    for class_label in CANONICAL_CLASS_ORDER:
        targets = make_one_vs_rest_targets(labels, class_label)
        if optimizer_name == "batch":
            result = train_binary_logistic_regression(
                matrix.rows,
                targets,
                training_config,
            )
        elif optimizer_name == "sgd":
            result = train_binary_logistic_regression_sgd(
                matrix.rows,
                targets,
                training_config,
            )
        else:
            result = train_binary_logistic_regression_minibatch(
                matrix.rows,
                targets,
                training_config,
            )
        classifiers.append(_classifier_from_result(class_label, result))

    metadata = OneVsRestTrainingMetadata(
        row_count=matrix.row_count,
        feature_count=len(preprocessing_state.features),
        matrix_column_count=matrix.column_count,
        class_order=CANONICAL_CLASS_ORDER,
        training_config=training_config,
        optimizer=optimizer_name,
    )
    state = OneVsRestModelState(
        schema_version=ONE_VS_REST_SCHEMA_VERSION,
        class_order=CANONICAL_CLASS_ORDER,
        preprocessing_state=preprocessing_state,
        classifiers=tuple(classifiers),
        training_config=training_config,
        metadata=metadata,
    )
    validate_one_vs_rest_state(state)
    return state


def predict_class_probabilities(
    dataset: Dataset,
    state: OneVsRestModelState,
) -> tuple[tuple[float, ...], ...]:
    """Return one binary probability per class for each dataset row."""
    validate_one_vs_rest_state(state)
    matrix = transform_dataset(dataset, state.preprocessing_state)
    rows: list[tuple[float, ...]] = []
    per_class_probabilities = [
        predict_binary_probabilities(matrix.rows, classifier.weights)
        for classifier in state.classifiers
    ]
    for row_index in range(matrix.row_count):
        row_probabilities: list[float] = []
        for probabilities in per_class_probabilities:
            value = probabilities[row_index]
            if not math.isfinite(value):
                raise OneVsRestError("non-finite class probability produced")
            row_probabilities.append(float(value))
        rows.append(tuple(row_probabilities))
    return tuple(rows)


def predict_house_labels(
    dataset: Dataset,
    state: OneVsRestModelState,
) -> tuple[str, ...]:
    """Predict one house per row using canonical-order argmax tie-breaking."""
    probabilities = predict_class_probabilities(dataset, state)
    predictions: list[str] = []
    for row_probabilities in probabilities:
        best_index = 0
        best_probability = row_probabilities[0]
        for index in range(1, len(row_probabilities)):
            probability = row_probabilities[index]
            if probability > best_probability:
                best_index = index
                best_probability = probability
        predictions.append(state.class_order[best_index])
    return tuple(predictions)


def multiclass_accuracy(actual_labels, predicted_labels) -> float:
    """Return simple accuracy using explicit loops."""
    actual = tuple(actual_labels)
    predicted = tuple(predicted_labels)
    if not actual:
        raise OneVsRestError("accuracy requires at least one label")
    if len(actual) != len(predicted):
        raise OneVsRestError(
            f"actual label count {len(actual)} does not match prediction count {len(predicted)}"
        )
    correct = 0
    total = 0
    for expected, observed in zip(actual, predicted):
        if not isinstance(expected, str) or expected == "":
            raise OneVsRestError(f"actual label {total} is invalid: {expected!r}")
        if not isinstance(observed, str) or observed == "":
            raise OneVsRestError(f"predicted label {total} is invalid: {observed!r}")
        if expected == observed:
            correct += 1
        total += 1
    return correct / total


def validate_one_vs_rest_state(state: OneVsRestModelState) -> None:
    """Validate a one-vs-rest model state without repairing it."""
    if not isinstance(state, OneVsRestModelState):
        raise OneVsRestStateError(
            f"expected OneVsRestModelState, got {type(state).__name__}"
        )
    if (
        isinstance(state.schema_version, bool)
        or not isinstance(state.schema_version, int)
        or state.schema_version != ONE_VS_REST_SCHEMA_VERSION
    ):
        raise OneVsRestStateError(
            f"unsupported one-vs-rest schema version {state.schema_version!r}"
        )
    _validate_class_order(state.class_order)
    validate_preprocessing_state(state.preprocessing_state)
    if isinstance(state.training_config, BinaryTrainingConfig):
        training_config = validate_binary_training_config(state.training_config)
    elif isinstance(state.training_config, StochasticTrainingConfig):
        training_config = validate_stochastic_training_config(state.training_config)
    elif isinstance(state.training_config, MiniBatchTrainingConfig):
        training_config = validate_minibatch_training_config(state.training_config)
    else:
        raise OneVsRestStateError(
            f"unsupported training configuration {type(state.training_config).__name__}"
        )
    if state.training_config != training_config:
        raise OneVsRestStateError("training configuration is not normalized")
    _validate_training_metadata(state.metadata, state)
    _validate_classifiers(state.classifiers, state.class_order, state.preprocessing_state)


def one_vs_rest_state_to_dict(state: OneVsRestModelState) -> dict[str, Any]:
    """Return a fresh JSON-compatible payload for a one-vs-rest model state."""
    validate_one_vs_rest_state(state)
    return {
        "schema_version": state.schema_version,
        "class_order": list(state.class_order),
        "preprocessing_state": preprocessing_state_to_dict(state.preprocessing_state),
        "training_config": {
            **_training_config_to_dict(state.training_config),
        },
        "metadata": {
            "row_count": state.metadata.row_count,
            "feature_count": state.metadata.feature_count,
            "matrix_column_count": state.metadata.matrix_column_count,
            "class_order": list(state.metadata.class_order),
            "optimizer": state.metadata.optimizer,
            "training_config": {
                **_training_config_to_dict(state.metadata.training_config),
            },
        },
        "classifiers": [
            {
                "class_label": classifier.class_label,
                "weights": list(classifier.weights),
                "iterations_run": classifier.iterations_run,
                "initial_loss": classifier.initial_loss,
                "final_loss": classifier.final_loss,
                "converged": classifier.converged,
                "final_gradient_max_abs": classifier.final_gradient_max_abs,
            }
            for classifier in state.classifiers
        ],
    }


def one_vs_rest_state_from_dict(payload: dict[str, Any]) -> OneVsRestModelState:
    """Build and validate a one-vs-rest model state from JSON-compatible data."""
    if not isinstance(payload, dict):
        raise OneVsRestStateError(
            f"one-vs-rest payload must be a dict, got {type(payload).__name__}"
        )
    required = {
        "schema_version",
        "class_order",
        "preprocessing_state",
        "training_config",
        "metadata",
        "classifiers",
    }
    _validate_keys(set(payload), required, "one-vs-rest payload")

    class_order = _payload_string_tuple(payload["class_order"], "class_order")
    preprocessing_state = preprocessing_state_from_dict(payload["preprocessing_state"])
    training_config = _config_from_payload(payload["training_config"], "training_config")
    metadata = _metadata_from_payload(payload["metadata"])
    classifiers = _classifiers_from_payload(payload["classifiers"])
    state = OneVsRestModelState(
        schema_version=payload["schema_version"],
        class_order=class_order,
        preprocessing_state=preprocessing_state,
        classifiers=classifiers,
        training_config=training_config,
        metadata=metadata,
    )
    validate_one_vs_rest_state(state)
    return state


def _classifier_from_result(
    class_label: str,
    result: BinaryLogisticRegressionResult,
) -> BinaryClassifierState:
    return BinaryClassifierState(
        class_label=class_label,
        weights=tuple(float(value) for value in result.weights),
        iterations_run=result.iterations_run,
        initial_loss=result.losses[0],
        final_loss=result.losses[-1],
        converged=result.converged,
        final_gradient_max_abs=result.final_gradient_max_abs,
    )


def _validate_class_order(class_order: tuple[str, ...]) -> None:
    if not isinstance(class_order, tuple):
        raise OneVsRestStateError("class_order must be a tuple")
    if class_order != CANONICAL_CLASS_ORDER:
        raise OneVsRestStateError(
            f"class_order must be canonical Hogwarts order, got {class_order!r}"
        )


def _validate_classifiers(
    classifiers: tuple[BinaryClassifierState, ...],
    class_order: tuple[str, ...],
    preprocessing_state: PreprocessingState,
) -> None:
    if not isinstance(classifiers, tuple):
        raise OneVsRestStateError("classifiers must be a tuple")
    if len(classifiers) != len(class_order):
        raise OneVsRestStateError(
            f"classifier count {len(classifiers)} does not match class count {len(class_order)}"
        )
    expected_weight_count = len(preprocessing_state.features)
    if preprocessing_state.add_bias:
        expected_weight_count += 1
    seen: list[str] = []
    for index, classifier in enumerate(classifiers):
        if not isinstance(classifier, BinaryClassifierState):
            raise OneVsRestStateError(
                f"classifier {index} must be BinaryClassifierState"
            )
        if classifier.class_label != class_order[index]:
            raise OneVsRestStateError("classifier order does not match class order")
        if classifier.class_label in seen:
            raise OneVsRestStateError(f"duplicate classifier class {classifier.class_label!r}")
        seen.append(classifier.class_label)
        validate_weights(classifier.weights, expected_weight_count)
        _validate_nonnegative_int(classifier.iterations_run, f"{classifier.class_label}.iterations_run")
        _validate_finite_number(classifier.initial_loss, f"{classifier.class_label}.initial_loss")
        _validate_finite_number(classifier.final_loss, f"{classifier.class_label}.final_loss")
        if not isinstance(classifier.converged, bool):
            raise OneVsRestStateError(
                f"{classifier.class_label}.converged must be bool"
            )
        _validate_finite_number(
            classifier.final_gradient_max_abs,
            f"{classifier.class_label}.final_gradient_max_abs",
        )
        if classifier.final_gradient_max_abs < 0.0:
            raise OneVsRestStateError(
                f"{classifier.class_label}.final_gradient_max_abs must be nonnegative"
            )


def _validate_training_metadata(
    metadata: OneVsRestTrainingMetadata,
    state: OneVsRestModelState,
) -> None:
    if not isinstance(metadata, OneVsRestTrainingMetadata):
        raise OneVsRestStateError("metadata must be OneVsRestTrainingMetadata")
    _validate_positive_int(metadata.row_count, "metadata.row_count")
    _validate_positive_int(metadata.feature_count, "metadata.feature_count")
    _validate_positive_int(metadata.matrix_column_count, "metadata.matrix_column_count")
    if metadata.feature_count != len(state.preprocessing_state.features):
        raise OneVsRestStateError("metadata feature count does not match state")
    expected_columns = metadata.feature_count + (1 if state.preprocessing_state.add_bias else 0)
    if metadata.matrix_column_count != expected_columns:
        raise OneVsRestStateError("metadata matrix column count does not match state")
    if metadata.class_order != state.class_order:
        raise OneVsRestStateError("metadata class order does not match state")
    if metadata.training_config != state.training_config:
        raise OneVsRestStateError("metadata training config does not match state")
    if metadata.optimizer != _optimizer_for_config(state.training_config):
        raise OneVsRestStateError("metadata optimizer does not match training config")


def _config_from_payload(
    payload,
    label: str,
) -> BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig:
    if not isinstance(payload, dict):
        raise OneVsRestStateError(f"{label} must be a dict")
    optimizer = payload.get("optimizer", "batch")
    if optimizer == "batch":
        required = {"learning_rate", "max_iterations", "tolerance", "minimum_iterations"}
    elif optimizer == "sgd":
        required = {"optimizer", "learning_rate", "epochs", "seed"}
    elif optimizer == "mini-batch":
        required = {"optimizer", "learning_rate", "epochs", "seed", "batch_size"}
    else:
        raise OneVsRestStateError(f"{label} optimizer is invalid: {optimizer!r}")
    _validate_keys(set(payload), required, label)
    if optimizer == "batch":
        return validate_binary_training_config(
            BinaryTrainingConfig(
                learning_rate=payload["learning_rate"],
                max_iterations=payload["max_iterations"],
                tolerance=payload["tolerance"],
                minimum_iterations=payload["minimum_iterations"],
            )
        )
    if optimizer == "sgd":
        return validate_stochastic_training_config(
            StochasticTrainingConfig(
                learning_rate=payload["learning_rate"],
                epochs=payload["epochs"],
                seed=payload["seed"],
            )
        )
    return validate_minibatch_training_config(
        MiniBatchTrainingConfig(
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            seed=payload["seed"],
            batch_size=payload["batch_size"],
        )
    )


def _metadata_from_payload(payload) -> OneVsRestTrainingMetadata:
    if not isinstance(payload, dict):
        raise OneVsRestStateError("metadata must be a dict")
    required = {
        "row_count",
        "feature_count",
        "matrix_column_count",
        "class_order",
        "training_config",
    }
    allowed = set(required)
    allowed.add("optimizer")
    actual = set(payload)
    missing = required - actual
    extra = actual - allowed
    if missing:
        raise OneVsRestStateError(
            f"metadata missing keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise OneVsRestStateError(
            f"metadata has unexpected keys: {', '.join(sorted(extra))}"
        )
    training_config = _config_from_payload(
        payload["training_config"],
        "metadata.training_config",
    )
    optimizer = payload.get("optimizer", _optimizer_for_config(training_config))
    if optimizer != _optimizer_for_config(training_config):
        raise OneVsRestStateError("metadata optimizer does not match training config")
    return OneVsRestTrainingMetadata(
        row_count=payload["row_count"],
        feature_count=payload["feature_count"],
        matrix_column_count=payload["matrix_column_count"],
        class_order=_payload_string_tuple(payload["class_order"], "metadata.class_order"),
        training_config=training_config,
        optimizer=optimizer,
    )


def _classifiers_from_payload(payload) -> tuple[BinaryClassifierState, ...]:
    if not isinstance(payload, list):
        raise OneVsRestStateError("classifiers must be a list")
    required = {
        "class_label",
        "weights",
        "iterations_run",
        "initial_loss",
        "final_loss",
        "converged",
        "final_gradient_max_abs",
    }
    classifiers: list[BinaryClassifierState] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise OneVsRestStateError(f"classifier {index} must be a dict")
        _validate_keys(set(entry), required, f"classifier {index}")
        if not isinstance(entry["weights"], list):
            raise OneVsRestStateError(f"classifier {index} weights must be a list")
        classifiers.append(
            BinaryClassifierState(
                class_label=entry["class_label"],
                weights=tuple(entry["weights"]),
                iterations_run=entry["iterations_run"],
                initial_loss=entry["initial_loss"],
                final_loss=entry["final_loss"],
                converged=entry["converged"],
                final_gradient_max_abs=entry["final_gradient_max_abs"],
            )
        )
    return tuple(classifiers)


def _payload_string_tuple(payload, label: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise OneVsRestStateError(f"{label} must be a list")
    values = tuple(payload)
    for value in values:
        if not isinstance(value, str) or value == "":
            raise OneVsRestStateError(f"{label} entries must be non-empty strings")
    return values


def _validate_keys(actual: set[str], required: set[str], label: str) -> None:
    missing = required - actual
    extra = actual - required
    if missing:
        raise OneVsRestStateError(
            f"{label} missing keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise OneVsRestStateError(
            f"{label} has unexpected keys: {', '.join(sorted(extra))}"
        )


def _validate_positive_int(value, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OneVsRestStateError(f"{label} must be a positive int, got {value!r}")


def _validate_nonnegative_int(value, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OneVsRestStateError(f"{label} must be a nonnegative int, got {value!r}")


def _validate_finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OneVsRestStateError(f"{label} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise OneVsRestStateError(f"{label} must be finite, got {value!r}")
    return parsed


def _validate_optimizer(optimizer: str) -> str:
    if optimizer not in ("batch", "sgd", "mini-batch"):
        raise OneVsRestError(
            f"optimizer must be 'batch', 'sgd', or 'mini-batch', got {optimizer!r}"
        )
    return optimizer


def _optimizer_for_config(
    config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig,
) -> str:
    if isinstance(config, BinaryTrainingConfig):
        return "batch"
    if isinstance(config, StochasticTrainingConfig):
        return "sgd"
    if isinstance(config, MiniBatchTrainingConfig):
        return "mini-batch"
    raise OneVsRestStateError(
        f"unsupported training configuration {type(config).__name__}"
    )


def _training_config_to_dict(
    config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig,
) -> dict[str, int | float | str]:
    optimizer = _optimizer_for_config(config)
    if optimizer == "batch":
        assert isinstance(config, BinaryTrainingConfig)
        return {
            "learning_rate": config.learning_rate,
            "max_iterations": config.max_iterations,
            "tolerance": config.tolerance,
            "minimum_iterations": config.minimum_iterations,
        }
    if optimizer == "sgd":
        assert isinstance(config, StochasticTrainingConfig)
        return {
            "optimizer": "sgd",
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "seed": config.seed,
        }
    assert isinstance(config, MiniBatchTrainingConfig)
    return {
        "optimizer": "mini-batch",
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "seed": config.seed,
        "batch_size": config.batch_size,
    }
