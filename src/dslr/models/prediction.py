"""Prediction workflow for saved DSLR one-vs-rest models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import ModelIOError, PredictionError
from dslr.models.model_io import INDEX_COLUMN, load_model
from dslr.models.one_vs_rest import OneVsRestModelState, predict_house_labels

PREDICTION_HEADER = ("Index", "Hogwarts House")


@dataclass(frozen=True)
class PredictionResult:
    """Immutable row-order predictions ready for houses.csv output."""

    header: tuple[str, str]
    rows: tuple[tuple[str, str], ...]
    class_order: tuple[str, ...]

    @property
    def row_count(self) -> int:
        """Return the number of predicted rows."""
        return len(self.rows)


def predict_dataset(dataset: Dataset, state: OneVsRestModelState) -> PredictionResult:
    """Predict Hogwarts houses for a loaded dataset using only saved model state."""
    predictions = predict_house_labels(dataset, state)
    if len(predictions) != dataset.row_count:
        raise PredictionError(
            f"prediction count {len(predictions)} does not match row count {dataset.row_count}"
        )

    indexes = _prediction_indexes(dataset)
    rows: list[tuple[str, str]] = []
    allowed = set(state.class_order)
    for index, house in zip(indexes, predictions):
        if house not in allowed:
            raise PredictionError(f"predicted unknown class {house!r}")
        rows.append((index, house))
    return PredictionResult(
        header=PREDICTION_HEADER,
        rows=tuple(rows),
        class_order=tuple(state.class_order),
    )


def predict_file(dataset_path, model_path) -> PredictionResult:
    """Load a CSV dataset and model JSON, then return deterministic predictions."""
    dataset = load_csv(dataset_path)
    state = load_model(model_path)
    return predict_dataset(dataset, state)


def write_predictions(result: PredictionResult, output_path) -> Path:
    """Write prediction rows as the houses.csv file."""
    _validate_prediction_result(result)
    path = Path(output_path)
    parent = path.parent
    if parent and not parent.exists():
        raise ModelIOError(f"prediction output directory does not exist: {parent}")
    if path.exists() and not path.is_file():
        raise ModelIOError(f"prediction output path is not a regular file: {path}")

    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(result.header)
            writer.writerows(result.rows)
        temp_path.replace(path)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise ModelIOError(f"could not write prediction file {path}: {exc}") from exc
    return path


def predict_and_write(dataset_path, model_path, output_path) -> tuple[PredictionResult, Path]:
    """Run the full prediction workflow and write the requested CSV output."""
    result = predict_file(dataset_path, model_path)
    saved_path = write_predictions(result, output_path)
    return result, saved_path


def format_prediction_report(result: PredictionResult, saved_path) -> str:
    """Return a concise deterministic CLI success report."""
    counts = _class_counts(result)
    count_text = ", ".join(
        f"{class_label}={counts[class_label]}" for class_label in result.class_order
    )
    return "\n".join(
        (
            f"Prediction rows: {result.row_count}",
            f"Classes: {', '.join(result.class_order)}",
            f"Class distribution: {count_text}",
            f"Predictions saved to: {Path(saved_path)}",
        )
    )


def _prediction_indexes(dataset: Dataset) -> tuple[str, ...]:
    if INDEX_COLUMN not in dataset.headers:
        return tuple(str(index) for index in range(dataset.row_count))

    indexes: list[str] = []
    for row_number, row in enumerate(dataset.rows, start=2):
        value = row[INDEX_COLUMN]
        if value is None:
            raise PredictionError(
                f"{dataset.source_path}: missing {INDEX_COLUMN!r} value at row {row_number}"
            )
        text = value.strip()
        if text == "":
            raise PredictionError(
                f"{dataset.source_path}: empty {INDEX_COLUMN!r} value at row {row_number}"
            )
        if not text.isdecimal():
            raise PredictionError(
                f"{dataset.source_path}: invalid {INDEX_COLUMN!r} value {value!r} at row {row_number}"
            )
        indexes.append(text)
    return tuple(indexes)


def _validate_prediction_result(result: PredictionResult) -> None:
    if not isinstance(result, PredictionResult):
        raise PredictionError(
            f"expected PredictionResult, got {type(result).__name__}"
        )
    if result.header != PREDICTION_HEADER:
        raise PredictionError(f"prediction header must be {PREDICTION_HEADER!r}")
    if not isinstance(result.rows, tuple):
        raise PredictionError("prediction rows must be a tuple")
    allowed = set(result.class_order)
    if not allowed:
        raise PredictionError("prediction class order must not be empty")
    for row_index, row in enumerate(result.rows):
        if (
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], str)
            or row[0] == ""
            or row[1] not in allowed
        ):
            raise PredictionError(f"invalid prediction row {row_index}: {row!r}")


def _class_counts(result: PredictionResult) -> dict[str, int]:
    counts = {class_label: 0 for class_label in result.class_order}
    for _, house in result.rows:
        counts[house] += 1
    return counts
