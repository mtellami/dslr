"""Reusable orchestration for DSLR model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dslr.data.csv_loader import load_csv
from dslr.errors import TrainingConfigurationError
from dslr.models.logistic_regression import (
    BinaryTrainingConfig,
    MiniBatchTrainingConfig,
    StochasticTrainingConfig,
    validate_binary_training_config,
    validate_minibatch_training_config,
    validate_stochastic_training_config,
)
from dslr.models.model_io import save_model
from dslr.models.one_vs_rest import (
    OneVsRestModelState,
    extract_training_labels,
    multiclass_accuracy,
    predict_house_labels,
    train_one_vs_rest_classifier,
)


@dataclass(frozen=True)
class TrainingClassSummary:
    """Immutable per-class training report row."""

    class_label: str
    iterations_run: int
    final_loss: float
    converged: bool


@dataclass(frozen=True)
class TrainingReport:
    """Immutable training orchestration result."""

    model_state: OneVsRestModelState
    training_rows: int
    feature_count: int
    matrix_column_count: int
    class_order: tuple[str, ...]
    class_summaries: tuple[TrainingClassSummary, ...]
    training_accuracy: float
    optimizer: str


def train_model_from_csv(
    dataset_path,
    config: BinaryTrainingConfig | None = None,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> TrainingReport:
    """Load a labeled training CSV and train the reusable one-vs-rest model."""
    if optimizer == "batch":
        training_config = validate_binary_training_config(config)
    elif optimizer == "sgd":
        training_config = validate_stochastic_training_config(sgd_config)
    elif optimizer == "mini-batch":
        training_config = validate_minibatch_training_config(minibatch_config)
    else:
        raise TrainingConfigurationError(f"unsupported optimizer {optimizer!r}")
    dataset = load_csv(dataset_path)
    model_state = train_one_vs_rest_classifier(
        dataset,
        config=training_config if optimizer == "batch" else None,
        optimizer=optimizer,
        sgd_config=training_config if optimizer == "sgd" else None,
        minibatch_config=training_config if optimizer == "mini-batch" else None,
    )
    labels = extract_training_labels(dataset)
    predictions = predict_house_labels(dataset, model_state)
    accuracy = multiclass_accuracy(labels, predictions)
    return TrainingReport(
        model_state=model_state,
        training_rows=model_state.metadata.row_count,
        feature_count=model_state.metadata.feature_count,
        matrix_column_count=model_state.metadata.matrix_column_count,
        class_order=model_state.class_order,
        class_summaries=tuple(
            TrainingClassSummary(
                class_label=classifier.class_label,
                iterations_run=classifier.iterations_run,
                final_loss=classifier.final_loss,
                converged=classifier.converged,
            )
            for classifier in model_state.classifiers
        ),
        training_accuracy=accuracy,
        optimizer=optimizer,
    )


def train_and_save_model(
    dataset_path,
    output_path,
    config: BinaryTrainingConfig | None = None,
    optimizer: str = "batch",
    sgd_config: StochasticTrainingConfig | None = None,
    minibatch_config: MiniBatchTrainingConfig | None = None,
) -> tuple[TrainingReport, Path]:
    """Train a model from a CSV path and save it as deterministic JSON."""
    report = train_model_from_csv(
        dataset_path,
        config=config,
        optimizer=optimizer,
        sgd_config=sgd_config,
        minibatch_config=minibatch_config,
    )
    saved_path = save_model(report.model_state, output_path)
    return report, saved_path


def format_training_report(report: TrainingReport, saved_path) -> str:
    """Return a concise deterministic training summary."""
    lines = [
        f"Optimizer: {report.optimizer}",
        f"Classes: {', '.join(report.class_order)}",
        f"Features ({report.feature_count}): {', '.join(report.model_state.preprocessing_state.features)}",
        f"Training rows: {report.training_rows}",
        f"Transformed columns: {report.matrix_column_count}",
        "Per-class training:",
    ]
    for summary in report.class_summaries:
        lines.append(
            f"  {summary.class_label}: iterations={summary.iterations_run}, "
            f"final_loss={summary.final_loss:.12f}, converged={summary.converged}"
        )
    lines.extend(
        [
            f"Training accuracy: {report.training_accuracy:.6f}",
            f"Model saved to: {Path(saved_path)}",
        ]
    )
    return "\n".join(lines)
