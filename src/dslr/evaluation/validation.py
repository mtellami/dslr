"""Deterministic validation for the frozen DSLR classifier decisions."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import ValidationError
from dslr.data.houses import CANONICAL_HOUSE_ORDER, HOUSE_COLUMN
from dslr.models.logistic_regression import (
    BinaryTrainingConfig,
    MiniBatchTrainingConfig,
    StochasticTrainingConfig,
    validate_binary_training_config,
    validate_minibatch_training_config,
    validate_stochastic_training_config,
)
from dslr.models.one_vs_rest import (
    OneVsRestModelState,
    extract_training_labels,
    multiclass_accuracy,
    predict_house_labels,
    train_one_vs_rest_classifier,
)
from dslr.models.training import train_model_from_csv

VALIDATION_SPLIT_SEEDS = (42, 1337, 2026)
VALIDATION_FRACTION = 0.20
VALIDATION_TARGET_ACCURACY = 0.98

FROZEN_MODEL_FEATURES = (
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying",
)

FROZEN_TRAINING_CONFIG = BinaryTrainingConfig(
    learning_rate=0.1,
    max_iterations=5000,
    tolerance=1e-9,
    minimum_iterations=1,
)


@dataclass(frozen=True)
class ClassSplitCount:
    """Per-class train/validation counts for a deterministic split."""

    class_label: str
    train_count: int
    validation_count: int


@dataclass(frozen=True)
class StratifiedSplit:
    """Immutable row-index split preserving class coverage."""

    seed: int
    validation_fraction: float
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    class_counts: tuple[ClassSplitCount, ...]


@dataclass(frozen=True)
class ClassValidationMetric:
    """Manual per-class validation metric."""

    class_label: str
    correct: int
    total: int
    accuracy: float


@dataclass(frozen=True)
class ConfusionMatrixRow:
    """One actual-class row of the confusion matrix."""

    actual_class: str
    predicted_counts: tuple[int, ...]


@dataclass(frozen=True)
class ValidationClassifierSummary:
    """Per-binary-class optimization summary for validation training."""

    class_label: str
    initial_loss: float
    final_loss: float
    iterations_run: int
    converged: bool


@dataclass(frozen=True)
class ValidationReport:
    """Complete deterministic holdout validation result."""

    seed: int
    validation_fraction: float
    features: tuple[str, ...]
    training_config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig
    train_row_count: int
    validation_row_count: int
    class_counts: tuple[ClassSplitCount, ...]
    accuracy: float
    per_class: tuple[ClassValidationMetric, ...]
    confusion_matrix: tuple[ConfusionMatrixRow, ...]
    classifier_summaries: tuple[ValidationClassifierSummary, ...]
    training_seconds: float
    model_state: OneVsRestModelState


@dataclass(frozen=True)
class RobustnessReport:
    """Multiple deterministic validation runs for the frozen candidate."""

    reports: tuple[ValidationReport, ...]
    mean_accuracy: float
    minimum_accuracy: float


def stratified_holdout_split(
    dataset: Dataset,
    seed: int = 42,
    validation_fraction: float = VALIDATION_FRACTION,
) -> StratifiedSplit:
    """Return deterministic row indices for a stratified holdout split."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(f"seed must be an int, got {seed!r}")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or float(validation_fraction) <= 0.0
        or float(validation_fraction) >= 1.0
    ):
        raise ValidationError(
            f"validation_fraction must be a finite value between 0 and 1, got {validation_fraction!r}"
        )
    labels = extract_training_labels(dataset)
    grouped = {class_label: [] for class_label in CANONICAL_HOUSE_ORDER}
    for index, label in enumerate(labels):
        if label not in grouped:
            raise ValidationError(f"unexpected class label {label!r}")
        grouped[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    counts: list[ClassSplitCount] = []
    for class_label in CANONICAL_HOUSE_ORDER:
        indices = list(grouped[class_label])
        if len(indices) < 2:
            raise ValidationError(
                f"class {class_label!r} needs at least two rows for holdout validation"
            )
        rng.shuffle(indices)
        validation_count = int(round(len(indices) * float(validation_fraction)))
        if validation_count < 1:
            validation_count = 1
        if validation_count >= len(indices):
            validation_count = len(indices) - 1
        class_validation = sorted(indices[:validation_count])
        class_train = sorted(indices[validation_count:])
        validation_indices.extend(class_validation)
        train_indices.extend(class_train)
        counts.append(
            ClassSplitCount(
                class_label=class_label,
                train_count=len(class_train),
                validation_count=len(class_validation),
            )
        )

    train_tuple = tuple(sorted(train_indices))
    validation_tuple = tuple(sorted(validation_indices))
    _validate_split_indices(dataset.row_count, train_tuple, validation_tuple)
    return StratifiedSplit(
        seed=seed,
        validation_fraction=float(validation_fraction),
        train_indices=train_tuple,
        validation_indices=validation_tuple,
        class_counts=tuple(counts),
    )


def evaluate_holdout(
    dataset: Dataset,
    seed: int = 42,
    config: BinaryTrainingConfig | None = None,
    features=None,
    validation_fraction: float = VALIDATION_FRACTION,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> ValidationReport:
    """Train on split-train rows and evaluate manually on validation rows."""
    if optimizer == "batch":
        training_config: BinaryTrainingConfig | StochasticTrainingConfig | MiniBatchTrainingConfig = (
            validate_binary_training_config(config or FROZEN_TRAINING_CONFIG)
        )
    elif optimizer == "sgd":
        training_config = validate_stochastic_training_config(sgd_config)
    elif optimizer == "mini-batch":
        training_config = validate_minibatch_training_config(minibatch_config)
    else:
        raise ValidationError(f"unsupported optimizer {optimizer!r}")
    feature_tuple = _resolve_features(features)
    split = stratified_holdout_split(dataset, seed, validation_fraction)
    train_dataset = subset_dataset(dataset, split.train_indices)
    validation_dataset = subset_dataset(dataset, split.validation_indices)

    started = time.perf_counter()
    model_state = train_one_vs_rest_classifier(
        train_dataset,
        config=training_config if optimizer == "batch" else None,
        features=feature_tuple,
        optimizer=optimizer,
        sgd_config=training_config if optimizer == "sgd" else None,
        minibatch_config=training_config if optimizer == "mini-batch" else None,
    )
    training_seconds = time.perf_counter() - started

    actual = extract_training_labels(validation_dataset)
    predicted = predict_house_labels(validation_dataset, model_state)
    accuracy = multiclass_accuracy(actual, predicted)
    per_class = class_metrics(actual, predicted, CANONICAL_HOUSE_ORDER)
    confusion = confusion_matrix(actual, predicted, CANONICAL_HOUSE_ORDER)
    summaries = tuple(
        ValidationClassifierSummary(
            class_label=classifier.class_label,
            initial_loss=classifier.initial_loss,
            final_loss=classifier.final_loss,
            iterations_run=classifier.iterations_run,
            converged=classifier.converged,
        )
        for classifier in model_state.classifiers
    )
    return ValidationReport(
        seed=seed,
        validation_fraction=split.validation_fraction,
        features=feature_tuple,
        training_config=training_config,
        train_row_count=len(split.train_indices),
        validation_row_count=len(split.validation_indices),
        class_counts=split.class_counts,
        accuracy=accuracy,
        per_class=per_class,
        confusion_matrix=confusion,
        classifier_summaries=summaries,
        training_seconds=float(training_seconds),
        model_state=model_state,
    )


def evaluate_csv_holdout(
    dataset_path,
    seed: int = 42,
    config: BinaryTrainingConfig | None = None,
    features=None,
    validation_fraction: float = VALIDATION_FRACTION,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> ValidationReport:
    """Load a labeled CSV and evaluate one deterministic holdout split."""
    return evaluate_holdout(
        load_csv(dataset_path),
        seed=seed,
        config=config,
        features=features,
        validation_fraction=validation_fraction,
        optimizer=optimizer,
        sgd_config=sgd_config,
        minibatch_config=minibatch_config,
    )


def evaluate_robustness(
    dataset: Dataset,
    seeds=VALIDATION_SPLIT_SEEDS,
    config: BinaryTrainingConfig | None = None,
    features=None,
    validation_fraction: float = VALIDATION_FRACTION,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> RobustnessReport:
    """Evaluate the same candidate across deterministic stratified seeds."""
    seed_tuple = tuple(seeds)
    if not seed_tuple:
        raise ValidationError("at least one validation seed is required")
    reports = tuple(
        evaluate_holdout(
            dataset,
            seed=seed,
            config=config,
            features=features,
            validation_fraction=validation_fraction,
            optimizer=optimizer,
            sgd_config=sgd_config,
            minibatch_config=minibatch_config,
        )
        for seed in seed_tuple
    )
    total = 0.0
    minimum = reports[0].accuracy
    for report in reports:
        total += report.accuracy
        if report.accuracy < minimum:
            minimum = report.accuracy
    return RobustnessReport(
        reports=reports,
        mean_accuracy=total / len(reports),
        minimum_accuracy=minimum,
    )


def train_full_frozen_model(dataset_path) -> tuple[OneVsRestModelState, float]:
    """Train the final frozen model on all labeled training rows and report accuracy."""
    report = train_model_from_csv(dataset_path, config=FROZEN_TRAINING_CONFIG)
    return report.model_state, report.training_accuracy


def subset_dataset(dataset: Dataset, indices: tuple[int, ...]) -> Dataset:
    """Return a row-order dataset subset with copied row dictionaries."""
    return Dataset(
        headers=list(dataset.headers),
        rows=[dict(dataset.rows[index]) for index in indices],
        source_path=dataset.source_path,
    )


def class_metrics(
    actual_labels,
    predicted_labels,
    class_order=CANONICAL_HOUSE_ORDER,
) -> tuple[ClassValidationMetric, ...]:
    """Return manual per-class correct/total/accuracy metrics."""
    actual = tuple(actual_labels)
    predicted = tuple(predicted_labels)
    if len(actual) != len(predicted):
        raise ValidationError("actual and predicted label counts differ")
    metrics: list[ClassValidationMetric] = []
    for class_label in class_order:
        total = 0
        correct = 0
        for expected, observed in zip(actual, predicted):
            if expected == class_label:
                total += 1
                if observed == expected:
                    correct += 1
        if total == 0:
            raise ValidationError(f"class {class_label!r} has no validation rows")
        metrics.append(
            ClassValidationMetric(
                class_label=class_label,
                correct=correct,
                total=total,
                accuracy=correct / total,
            )
        )
    return tuple(metrics)


def confusion_matrix(
    actual_labels,
    predicted_labels,
    class_order=CANONICAL_HOUSE_ORDER,
) -> tuple[ConfusionMatrixRow, ...]:
    """Return a manual confusion matrix in class-order rows and columns."""
    actual = tuple(actual_labels)
    predicted = tuple(predicted_labels)
    if len(actual) != len(predicted):
        raise ValidationError("actual and predicted label counts differ")
    index_by_class = {class_label: index for index, class_label in enumerate(class_order)}
    rows = [[0 for _ in class_order] for _ in class_order]
    for expected, observed in zip(actual, predicted):
        if expected not in index_by_class:
            raise ValidationError(f"actual label is not in class order: {expected!r}")
        if observed not in index_by_class:
            raise ValidationError(f"predicted label is not in class order: {observed!r}")
        rows[index_by_class[expected]][index_by_class[observed]] += 1
    return tuple(
        ConfusionMatrixRow(class_label, tuple(rows[index]))
        for index, class_label in enumerate(class_order)
    )


def _validate_split_indices(
    row_count: int,
    train_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
) -> None:
    train_set = set(train_indices)
    validation_set = set(validation_indices)
    if train_set & validation_set:
        raise ValidationError("train and validation indices overlap")
    combined = train_set | validation_set
    if combined != set(range(row_count)):
        raise ValidationError("split indices do not cover every source row exactly once")


def _resolve_features(features) -> tuple[str, ...]:
    if features is None:
        return FROZEN_MODEL_FEATURES
    feature_tuple = tuple(features)
    if not feature_tuple:
        raise ValidationError("validation requires at least one feature")
    return feature_tuple
