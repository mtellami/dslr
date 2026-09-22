"""Optional training diagnostics for DSLR logistic-regression models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import VisualizationError
from dslr.data.houses import CANONICAL_HOUSE_ORDER
from dslr.models.logistic_regression import (
    BinaryTrainingConfig,
    train_binary_logistic_regression,
    validate_binary_training_config,
)
from dslr.models.one_vs_rest import (
    BinaryClassifierState,
    OneVsRestModelState,
    OneVsRestTrainingMetadata,
    extract_training_labels,
    make_one_vs_rest_targets,
    multiclass_accuracy,
    predict_house_labels,
    validate_one_vs_rest_state,
)
from dslr.analysis.plotting import import_pyplot
from dslr.data.preprocessing import fit_preprocessor, transform_dataset
from dslr.evaluation.validation import (
    FROZEN_TRAINING_CONFIG,
    VALIDATION_FRACTION,
    stratified_holdout_split,
    subset_dataset,
)

DIAGNOSTIC_CHECKPOINTS = (0, 250, 500, 1000, 2000, 3000, 5000)
DIAGNOSTIC_SPLIT_SEED = 42
DIAGNOSTIC_OUTPUT_FILES = (
    "loss_curves.png",
    "convergence.png",
    "validation_accuracy.png",
    "weight_comparison.png",
)


@dataclass(frozen=True)
class ClassTrainingDiagnostic:
    """Immutable per-class diagnostic data from one binary classifier."""

    class_label: str
    weights: tuple[float, ...]
    losses: tuple[float, ...]
    iterations_run: int
    converged: bool
    final_gradient_max_abs: float


@dataclass(frozen=True)
class CheckpointAccuracy:
    """Validation accuracy at one deterministic training checkpoint."""

    checkpoint: int
    accuracy: float


@dataclass(frozen=True)
class TrainingDiagnostics:
    """Complete immutable diagnostics analysis ready for rendering."""

    class_order: tuple[str, ...]
    features: tuple[str, ...]
    training_rows: int
    matrix_column_count: int
    class_diagnostics: tuple[ClassTrainingDiagnostic, ...]
    checkpoint_accuracies: tuple[CheckpointAccuracy, ...]
    training_config: BinaryTrainingConfig
    validation_seed: int


def analyze_training_diagnostics(
    dataset: Dataset,
    config: BinaryTrainingConfig | None = None,
    features=None,
    checkpoints=DIAGNOSTIC_CHECKPOINTS,
    validation_seed: int = DIAGNOSTIC_SPLIT_SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> TrainingDiagnostics:
    """Train diagnostic models and evaluate deterministic holdout checkpoints."""
    training_config = validate_binary_training_config(config or FROZEN_TRAINING_CONFIG)
    checkpoint_tuple = _validate_checkpoints(checkpoints, training_config.max_iterations)
    class_diagnostics = _train_class_diagnostics(dataset, training_config, features)
    checkpoint_accuracies = _checkpoint_accuracies(
        dataset,
        training_config,
        features,
        checkpoint_tuple,
        validation_seed,
        validation_fraction,
    )
    matrix_column_count = len(class_diagnostics[0].weights)
    return TrainingDiagnostics(
        class_order=CANONICAL_HOUSE_ORDER,
        features=_resolve_features_from_diagnostics(class_diagnostics, dataset, features),
        training_rows=dataset.row_count,
        matrix_column_count=matrix_column_count,
        class_diagnostics=class_diagnostics,
        checkpoint_accuracies=checkpoint_accuracies,
        training_config=training_config,
        validation_seed=validation_seed,
    )


def analyze_training_diagnostics_file(
    dataset_path,
    config: BinaryTrainingConfig | None = None,
) -> TrainingDiagnostics:
    """Load a training CSV and return deterministic diagnostics analysis."""
    return analyze_training_diagnostics(load_csv(dataset_path), config=config)


def render_training_diagnostics(
    diagnostics: TrainingDiagnostics,
    output_dir,
) -> tuple[Path, ...]:
    """Render all diagnostic PNGs into the requested output directory."""
    _validate_diagnostics(diagnostics)
    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise VisualizationError(f"diagnostic output path is not a directory: {directory}")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VisualizationError(f"could not create diagnostic output directory {directory}: {exc}") from exc

    pyplot = import_pyplot()
    renderers = (
        ("loss_curves.png", _create_loss_curves_figure),
        ("convergence.png", _create_convergence_figure),
        ("validation_accuracy.png", _create_validation_accuracy_figure),
        ("weight_comparison.png", _create_weight_comparison_figure),
    )
    paths: list[Path] = []
    for filename, renderer in renderers:
        figure = renderer(pyplot, diagnostics)
        paths.append(_save_figure(pyplot, figure, directory / filename))
    return tuple(paths)


def create_training_diagnostics(dataset_path, output_dir) -> tuple[TrainingDiagnostics, tuple[Path, ...]]:
    """Run diagnostics from a CSV path and render the four standard PNGs."""
    diagnostics = analyze_training_diagnostics_file(dataset_path)
    paths = render_training_diagnostics(diagnostics, output_dir)
    return diagnostics, paths


def format_diagnostics_report(
    diagnostics: TrainingDiagnostics,
    paths: tuple[Path, ...],
) -> str:
    """Return a concise deterministic diagnostics report."""
    _validate_diagnostics(diagnostics)
    lines = [
        f"Classes: {', '.join(diagnostics.class_order)}",
        f"Features ({len(diagnostics.features)}): {', '.join(diagnostics.features)}",
        f"Training rows: {diagnostics.training_rows}",
        f"Matrix columns: {diagnostics.matrix_column_count}",
        "Final class losses:",
    ]
    for class_diagnostic in diagnostics.class_diagnostics:
        lines.append(
            f"  {class_diagnostic.class_label}: "
            f"loss={class_diagnostic.losses[-1]:.12f}, "
            f"iterations={class_diagnostic.iterations_run}, "
            f"converged={class_diagnostic.converged}"
        )
    lines.append("Validation checkpoints:")
    for checkpoint in diagnostics.checkpoint_accuracies:
        lines.append(f"  {checkpoint.checkpoint}: accuracy={checkpoint.accuracy:.6f}")
    lines.append("Generated files:")
    for path in paths:
        lines.append(f"  {Path(path)}")
    return "\n".join(lines)


def _train_class_diagnostics(
    dataset: Dataset,
    config: BinaryTrainingConfig,
    features,
) -> tuple[ClassTrainingDiagnostic, ...]:
    labels = extract_training_labels(dataset)
    preprocessing_state = fit_preprocessor(dataset, features=features, add_bias=True)
    matrix = transform_dataset(dataset, preprocessing_state)
    diagnostics: list[ClassTrainingDiagnostic] = []
    for class_label in CANONICAL_HOUSE_ORDER:
        targets = make_one_vs_rest_targets(labels, class_label)
        result = train_binary_logistic_regression(matrix.rows, targets, config)
        diagnostics.append(
            ClassTrainingDiagnostic(
                class_label=class_label,
                weights=result.weights,
                losses=result.losses,
                iterations_run=result.iterations_run,
                converged=result.converged,
                final_gradient_max_abs=result.final_gradient_max_abs,
            )
        )
    return tuple(diagnostics)


def _checkpoint_accuracies(
    dataset: Dataset,
    config: BinaryTrainingConfig,
    features,
    checkpoints: tuple[int, ...],
    validation_seed: int,
    validation_fraction: float,
) -> tuple[CheckpointAccuracy, ...]:
    split = stratified_holdout_split(dataset, validation_seed, validation_fraction)
    train_dataset = subset_dataset(dataset, split.train_indices)
    validation_dataset = subset_dataset(dataset, split.validation_indices)
    actual = extract_training_labels(validation_dataset)
    accuracies: list[CheckpointAccuracy] = []
    for checkpoint in checkpoints:
        state = _train_checkpoint_state(train_dataset, config, features, checkpoint)
        predicted = predict_house_labels(validation_dataset, state)
        accuracies.append(
            CheckpointAccuracy(
                checkpoint=checkpoint,
                accuracy=multiclass_accuracy(actual, predicted),
            )
        )
    return tuple(accuracies)


def _train_checkpoint_state(
    dataset: Dataset,
    config: BinaryTrainingConfig,
    features,
    checkpoint: int,
) -> OneVsRestModelState:
    labels = extract_training_labels(dataset)
    preprocessing_state = fit_preprocessor(dataset, features=features, add_bias=True)
    matrix = transform_dataset(dataset, preprocessing_state)
    classifiers: list[BinaryClassifierState] = []
    for class_label in CANONICAL_HOUSE_ORDER:
        targets = make_one_vs_rest_targets(labels, class_label)
        if checkpoint == 0:
            weights = tuple(0.0 for _ in range(matrix.column_count))
            initial_loss = math.log(2.0)
            final_loss = initial_loss
            converged = False
            final_gradient = 0.0
        else:
            checkpoint_config = BinaryTrainingConfig(
                learning_rate=config.learning_rate,
                max_iterations=checkpoint,
                tolerance=0.0,
                minimum_iterations=0,
            )
            result = train_binary_logistic_regression(
                matrix.rows,
                targets,
                checkpoint_config,
            )
            weights = result.weights
            initial_loss = result.losses[0]
            final_loss = result.losses[-1]
            converged = result.converged
            final_gradient = result.final_gradient_max_abs
        classifiers.append(
            BinaryClassifierState(
                class_label=class_label,
                weights=weights,
                iterations_run=checkpoint,
                initial_loss=initial_loss,
                final_loss=final_loss,
                converged=converged,
                final_gradient_max_abs=final_gradient,
            )
        )
    metadata = OneVsRestTrainingMetadata(
        row_count=matrix.row_count,
        feature_count=len(preprocessing_state.features),
        matrix_column_count=matrix.column_count,
        class_order=CANONICAL_HOUSE_ORDER,
        training_config=config,
        optimizer="batch",
    )
    state = OneVsRestModelState(
        schema_version=1,
        class_order=CANONICAL_HOUSE_ORDER,
        preprocessing_state=preprocessing_state,
        classifiers=tuple(classifiers),
        training_config=config,
        metadata=metadata,
    )
    validate_one_vs_rest_state(state)
    return state


def _resolve_features_from_diagnostics(
    class_diagnostics: tuple[ClassTrainingDiagnostic, ...],
    dataset: Dataset,
    features,
) -> tuple[str, ...]:
    preprocessing_state = fit_preprocessor(dataset, features=features, add_bias=True)
    expected_columns = len(preprocessing_state.features) + 1
    for class_diagnostic in class_diagnostics:
        if len(class_diagnostic.weights) != expected_columns:
            raise VisualizationError("diagnostic weight dimensions are inconsistent")
    return preprocessing_state.features


def _validate_checkpoints(checkpoints, max_iterations: int) -> tuple[int, ...]:
    values = tuple(checkpoints)
    if not values:
        raise VisualizationError("at least one diagnostic checkpoint is required")
    previous = -1
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise VisualizationError(f"checkpoint must be an integer, got {value!r}")
        if value < 0:
            raise VisualizationError(f"checkpoint must be nonnegative, got {value!r}")
        if value > max_iterations:
            raise VisualizationError(
                f"checkpoint {value} exceeds max_iterations {max_iterations}"
            )
        if value <= previous:
            raise VisualizationError("diagnostic checkpoints must be strictly increasing")
        parsed.append(value)
        previous = value
    return tuple(parsed)


def _validate_diagnostics(diagnostics: TrainingDiagnostics) -> None:
    if not isinstance(diagnostics, TrainingDiagnostics):
        raise VisualizationError(
            f"expected TrainingDiagnostics, got {type(diagnostics).__name__}"
        )
    if diagnostics.class_order != CANONICAL_HOUSE_ORDER:
        raise VisualizationError("diagnostic class order is not canonical")
    if not diagnostics.features:
        raise VisualizationError("diagnostics require at least one feature")
    if diagnostics.matrix_column_count != len(diagnostics.features) + 1:
        raise VisualizationError("diagnostic matrix column count is inconsistent")
    if tuple(item.class_label for item in diagnostics.class_diagnostics) != CANONICAL_HOUSE_ORDER:
        raise VisualizationError("class diagnostics are not in canonical order")
    for item in diagnostics.class_diagnostics:
        if len(item.weights) != diagnostics.matrix_column_count:
            raise VisualizationError("diagnostic weight dimension mismatch")
        if not item.losses:
            raise VisualizationError("class diagnostic loss history is empty")
        for value in item.weights + item.losses:
            if not math.isfinite(value):
                raise VisualizationError("non-finite diagnostic value")


def _create_loss_curves_figure(pyplot, diagnostics: TrainingDiagnostics):
    figure, axes = pyplot.subplots(figsize=(9, 5))
    for item in diagnostics.class_diagnostics:
        axes.plot(range(len(item.losses)), item.losses, label=item.class_label)
    axes.set_title("Per-class training loss curves")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Binary cross-entropy")
    axes.grid(True, alpha=0.25)
    axes.legend()
    figure.tight_layout()
    return figure


def _create_convergence_figure(pyplot, diagnostics: TrainingDiagnostics):
    figure, axes = pyplot.subplots(figsize=(8, 5))
    labels = [item.class_label for item in diagnostics.class_diagnostics]
    final_losses = [item.losses[-1] for item in diagnostics.class_diagnostics]
    bars = axes.bar(labels, final_losses)
    axes.set_title("Per-class convergence summary")
    axes.set_ylabel("Final loss")
    axes.tick_params(axis="x", rotation=20)
    for bar, item in zip(bars, diagnostics.class_diagnostics):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{item.iterations_run}\n{item.converged}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    return figure


def _create_validation_accuracy_figure(pyplot, diagnostics: TrainingDiagnostics):
    figure, axes = pyplot.subplots(figsize=(8, 5))
    checkpoints = [item.checkpoint for item in diagnostics.checkpoint_accuracies]
    accuracies = [item.accuracy for item in diagnostics.checkpoint_accuracies]
    axes.plot(checkpoints, accuracies, marker="o")
    axes.set_title("Validation accuracy by training progress")
    axes.set_xlabel("Batch GD iterations")
    axes.set_ylabel("Validation accuracy")
    axes.set_ylim(0.0, 1.05)
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def _create_weight_comparison_figure(pyplot, diagnostics: TrainingDiagnostics):
    figure, axes = pyplot.subplots(figsize=(12, 5))
    labels = ("Bias",) + diagnostics.features
    x_positions = list(range(len(labels)))
    class_count = len(diagnostics.class_diagnostics)
    width = 0.8 / class_count
    for class_index, item in enumerate(diagnostics.class_diagnostics):
        offset = (class_index - (class_count - 1) / 2) * width
        positions = [position + offset for position in x_positions]
        axes.bar(positions, item.weights, width=width, label=item.class_label)
    axes.set_title("Learned coefficient comparison")
    axes.set_ylabel("Weight")
    axes.set_xticks(x_positions)
    axes.set_xticklabels(labels, rotation=35, ha="right")
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.legend()
    figure.tight_layout()
    return figure


def _save_figure(pyplot, figure, path: Path) -> Path:
    temp_path = path.with_name(f".{path.name}.tmp{path.suffix}")
    try:
        figure.savefig(temp_path)
        temp_path.replace(path)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise VisualizationError(f"could not save diagnostic plot {path}: {exc}") from exc
    finally:
        pyplot.close(figure)
    return path
