"""Descriptive-statistics helpers for numerical CSV features."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import DescribeError
from dslr.analysis.statistics import DescriptiveStatistics, summarize

DEFAULT_DESCRIBE_EXCLUDED_COLUMNS = frozenset(("Index", "Hogwarts House"))
DESCRIBE_ROW_LABELS = (
    ("Count", "count"),
    ("Mean", "mean"),
    ("Std", "std"),
    ("Min", "minimum"),
    ("25%", "q1"),
    ("50%", "median"),
    ("75%", "q3"),
    ("Max", "maximum"),
    ("Variance", "sample_variance"),
    ("Range", "value_range"),
    ("IQR", "iqr"),
    ("Missing", "missing_count"),
    ("Skewness", "skewness"),
)
STATISTIC_COLUMN_WIDTH = 9
FEATURE_COLUMN_WIDTH = 28
FEATURES_PER_BLOCK = 4
COLUMN_SEPARATOR = "  "


@dataclass(frozen=True)
class FeatureDescription:
    """Manual statistics for one selected numerical feature."""

    name: str
    statistics: DescriptiveStatistics


def select_describe_columns(
    dataset: Dataset,
    excluded: Iterable[str] | None = None,
) -> list[str]:
    """Return numerical describe columns in source order."""
    exclusions = set(DEFAULT_DESCRIBE_EXCLUDED_COLUMNS)
    if excluded is not None:
        exclusions.update(excluded)

    selected: list[str] = []
    for column in dataset.detect_numeric_columns():
        if column not in exclusions:
            selected.append(column)
    return selected


def summarize_dataset(
    dataset: Dataset,
    columns: Iterable[str] | None = None,
) -> tuple[FeatureDescription, ...]:
    """Summarize selected dataset columns without formatting or printing."""
    if columns is None:
        selected_columns = select_describe_columns(dataset)
    else:
        selected_columns = _validate_requested_columns(dataset, columns)

    if not selected_columns:
        raise DescribeError(
            f"{dataset.source_path}: no numerical feature columns to describe"
        )

    descriptions: list[FeatureDescription] = []
    for column in selected_columns:
        values = dataset.numeric_values(column)
        statistics = summarize(values)
        descriptions.append(
            FeatureDescription(
                column,
                replace(
                    statistics,
                    missing_count=dataset.row_count - statistics.count,
                ),
            )
        )
    return tuple(descriptions)


def format_describe_table(
    feature_descriptions: Iterable[FeatureDescription],
) -> str:
    """Render calculated feature summaries as a deterministic fixed-width table."""
    descriptions = tuple(feature_descriptions)
    if not descriptions:
        raise DescribeError("cannot format an empty describe table")

    blocks: list[str] = []
    for start in range(0, len(descriptions), FEATURES_PER_BLOCK):
        block = descriptions[start : start + FEATURES_PER_BLOCK]
        blocks.append(_format_block(block))
    return "\n\n".join(blocks).rstrip()


def describe_dataset(path: str | Path) -> str:
    """Load, summarize, and render a CSV dataset without printing."""
    dataset = load_csv(path)
    return format_describe_table(summarize_dataset(dataset))


def _validate_requested_columns(
    dataset: Dataset,
    columns: Iterable[str],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in columns:
        dataset.require_column(column)
        if column in seen:
            duplicates.append(column)
        else:
            seen.add(column)
            selected.append(column)
    if duplicates:
        formatted = ", ".join(repr(name) for name in duplicates)
        raise DescribeError(f"{dataset.source_path}: duplicate describe columns: {formatted}")
    return selected


def _format_block(block: Sequence[FeatureDescription]) -> str:
    wrapped_headers = [_wrap_header(description.name) for description in block]
    header_line_count = _largest_wrapped_header_count(wrapped_headers)
    lines: list[str] = []

    for line_index in range(header_line_count):
        statistic_cell = (
            "Statistic" if line_index == header_line_count - 1 else ""
        ).ljust(STATISTIC_COLUMN_WIDTH)
        cells = [statistic_cell]
        for wrapped in wrapped_headers:
            offset = line_index - (header_line_count - len(wrapped))
            text = wrapped[offset] if 0 <= offset < len(wrapped) else ""
            cells.append(text.center(FEATURE_COLUMN_WIDTH))
        lines.append(COLUMN_SEPARATOR.join(cells).rstrip())

    for label, field_name in DESCRIBE_ROW_LABELS:
        cells = [label.ljust(STATISTIC_COLUMN_WIDTH)]
        for description in block:
            value = getattr(description.statistics, field_name)
            cells.append(_format_value(value).rjust(FEATURE_COLUMN_WIDTH))
        lines.append(COLUMN_SEPARATOR.join(cells).rstrip())
    return "\n".join(lines)


def _wrap_header(name: str) -> list[str]:
    wrapped = textwrap.wrap(
        name,
        width=FEATURE_COLUMN_WIDTH,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return wrapped if wrapped else [""]


def _largest_wrapped_header_count(wrapped_headers: Iterable[Sequence[str]]) -> int:
    largest = 0
    for wrapped in wrapped_headers:
        current = len(wrapped)
        if current > largest:
            largest = current
    return largest


def _format_value(value: int | float | None) -> str:
    if value is None:
        return "NaN"
    return f"{float(value):.6f}"
