"""Histogram analysis for homogeneous Hogwarts course distributions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import DatasetFormatError, HistogramError, VisualizationError
from dslr.data.houses import (
    CANONICAL_HOUSE_ORDER,
    HOUSE_COLUMN,
    extract_house_labels,
)
from dslr.analysis.plotting import import_pyplot

TARGET_COLUMN = HOUSE_COLUMN
DEFAULT_HISTOGRAM_EXCLUDED_COLUMNS = frozenset(("Index", TARGET_COLUMN))
DEFAULT_HISTOGRAM_BIN_COUNT = 20
DEFAULT_BIN_COUNT = DEFAULT_HISTOGRAM_BIN_COUNT
HOUSE_ORDER = CANONICAL_HOUSE_ORDER


@dataclass(frozen=True)
class PairwiseHouseDistance:
    """Manual ECDF maximum distance for one pair of houses."""

    house_a: str
    house_b: str
    distance: float


@dataclass(frozen=True)
class CourseHistogramAnalysis:
    """Histogram homogeneity analysis for one numerical course."""

    course: str
    average_pairwise_distance: float
    sample_counts: Mapping[str, int]
    values_by_house: Mapping[str, tuple[float, ...]]
    pairwise_distances: tuple[PairwiseHouseDistance, ...]


@dataclass(frozen=True)
class HistogramAnalysisReport:
    """Ranked histogram analysis for all eligible numerical courses."""

    selected_course: CourseHistogramAnalysis
    ranking: tuple[CourseHistogramAnalysis, ...]
    house_order: tuple[str, ...]


def select_histogram_columns(
    dataset: Dataset,
    excluded: Iterable[str] | None = None,
) -> list[str]:
    """Return numerical histogram candidates in source order."""
    exclusions = set(DEFAULT_HISTOGRAM_EXCLUDED_COLUMNS)
    if excluded is not None:
        exclusions.update(excluded)

    selected: list[str] = []
    for column in dataset.detect_numeric_columns():
        if column not in exclusions:
            selected.append(column)
    return selected


def ecdf_max_distance(sample_a: Iterable[float], sample_b: Iterable[float]) -> float:
    """Return the manual two-sample maximum ECDF distance."""
    left = sorted(float(value) for value in sample_a)
    right = sorted(float(value) for value in sample_b)
    left_count = len(left)
    right_count = len(right)
    if left_count == 0 or right_count == 0:
        raise HistogramError("ECDF distance requires two non-empty samples")

    left_index = 0
    right_index = 0
    largest_distance = 0.0
    while left_index < left_count or right_index < right_count:
        if right_index >= right_count:
            current = left[left_index]
        elif left_index >= left_count:
            current = right[right_index]
        elif left[left_index] <= right[right_index]:
            current = left[left_index]
        else:
            current = right[right_index]

        while left_index < left_count and left[left_index] <= current:
            left_index += 1
        while right_index < right_count and right[right_index] <= current:
            right_index += 1

        left_ecdf = left_index / left_count
        right_ecdf = right_index / right_count
        distance = abs(left_ecdf - right_ecdf)
        if distance > largest_distance:
            largest_distance = distance
    return largest_distance


def shared_histogram_bins(
    values_by_house: Mapping[str, Sequence[float]],
    bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
) -> tuple[float, ...]:
    """Build shared equal-width histogram bin edges for all plotted houses."""
    if bin_count < 1:
        raise HistogramError(f"bin_count must be positive, got {bin_count!r}")

    seen = False
    lowest = 0.0
    highest = 0.0
    for values in values_by_house.values():
        for value in values:
            if not math.isfinite(value):
                raise HistogramError(f"non-finite histogram value {value!r}")
            if not seen:
                lowest = value
                highest = value
                seen = True
            else:
                if value < lowest:
                    lowest = value
                if value > highest:
                    highest = value
    if not seen:
        raise HistogramError("cannot build histogram bins without values")

    if lowest == highest:
        lowest -= 0.5
        highest += 0.5

    width = (highest - lowest) / bin_count
    bins: list[float] = []
    for index in range(bin_count + 1):
        bins.append(lowest + width * index)
    bins[-1] = highest
    return tuple(bins)


def analyze_histogram_dataset(
    dataset: Dataset,
    *,
    columns: Iterable[str] | None = None,
) -> HistogramAnalysisReport:
    """Rank eligible courses by average pairwise house ECDF distance."""
    dataset.require_column(TARGET_COLUMN)
    house_order = _house_order(dataset)
    if len(house_order) < 4:
        raise HistogramError(
            f"{dataset.source_path}: histogram analysis requires all four houses"
        )

    selected_columns = list(columns) if columns is not None else select_histogram_columns(dataset)
    if not selected_columns:
        raise HistogramError(
            f"{dataset.source_path}: no numerical course columns for histogram analysis"
        )

    analyses: list[CourseHistogramAnalysis] = []
    for column in selected_columns:
        dataset.require_column(column)
        grouped = _values_by_house(dataset, column, house_order)
        if not _has_all_house_samples(grouped, house_order):
            continue
        pairwise = _pairwise_distances(grouped, house_order)
        total = 0.0
        pair_count = 0
        for item in pairwise:
            total += item.distance
            pair_count += 1
        analyses.append(
            CourseHistogramAnalysis(
                course=column,
                average_pairwise_distance=total / pair_count,
                sample_counts={house: len(grouped[house]) for house in house_order},
                values_by_house={house: tuple(grouped[house]) for house in house_order},
                pairwise_distances=pairwise,
            )
        )

    if not analyses:
        raise HistogramError(
            f"{dataset.source_path}: no courses contain samples for all four houses"
        )

    ranking = tuple(
        sorted(
            analyses,
            key=lambda analysis: (
                analysis.average_pairwise_distance,
                selected_columns.index(analysis.course),
            ),
        )
    )
    return HistogramAnalysisReport(
        selected_course=ranking[0],
        ranking=ranking,
        house_order=house_order,
    )


def analyze_histogram_file(path: str | Path) -> HistogramAnalysisReport:
    """Load a CSV file and return ranked histogram analysis."""
    return analyze_histogram_dataset(load_csv(path))


def create_histogram_figure(
    analysis: CourseHistogramAnalysis,
    house_order: Sequence[str],
    *,
    bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
):
    """Create the selected-course histogram figure without showing or saving it."""
    pyplot = import_pyplot()
    bins = shared_histogram_bins(analysis.values_by_house, bin_count=bin_count)

    figure, axes = pyplot.subplots(figsize=(10, 6))
    for house in house_order:
        values = analysis.values_by_house[house]
        weights = [1.0 / len(values) for _ in values]
        axes.hist(
            values,
            bins=bins,
            weights=weights,
            alpha=0.45,
            label=house,
            edgecolor="black",
            linewidth=0.5,
            density=False,
            stacked=False,
        )
    axes.set_title(
        "Most homogeneous course score distribution: "
        f"{analysis.course} (score {analysis.average_pairwise_distance:.6f})"
    )
    axes.set_xlabel("Score")
    axes.set_ylabel("Proportion")
    axes.legend()
    axes.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def render_histogram(
    analysis: CourseHistogramAnalysis,
    house_order: Sequence[str],
    *,
    output_path: str | Path | None = None,
    bin_count: int = DEFAULT_HISTOGRAM_BIN_COUNT,
) -> Path | None:
    """Render the selected-course histogram with lazily imported Matplotlib."""
    pyplot = import_pyplot()
    figure = create_histogram_figure(
        analysis,
        house_order,
        bin_count=bin_count,
    )

    if output_path is None:
        try:
            pyplot.show()
            return None
        finally:
            pyplot.close(figure)

    destination = Path(output_path)
    if destination.exists() and not destination.is_file():
        pyplot.close(figure)
        raise VisualizationError(f"{destination}: output path is not a regular file")

    existed_before = destination.exists()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="png", dpi=120)
        return destination
    except (OSError, ValueError) as exc:
        if not existed_before and destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise VisualizationError(f"{destination}: could not save histogram image") from exc
    finally:
        pyplot.close(figure)


def format_histogram_report(report: HistogramAnalysisReport) -> str:
    """Return a deterministic text summary of the histogram ranking."""
    lines = [
        "Histogram homogeneity ranking",
        f"Selected course: {report.selected_course.course}",
        (
            "Selected score: "
            f"{report.selected_course.average_pairwise_distance:.6f}"
        ),
        "Ranking:",
    ]
    for index, analysis in enumerate(report.ranking, start=1):
        counts = ", ".join(
            f"{house}={analysis.sample_counts[house]}"
            for house in report.house_order
        )
        lines.append(
            f"{index}. {analysis.course}: "
            f"{analysis.average_pairwise_distance:.6f} ({counts})"
        )
    return "\n".join(lines)


def create_histogram(path: str | Path, output_path: str | Path | None = None) -> str:
    """Analyze a dataset, render the selected histogram, and return text output."""
    report = analyze_histogram_file(path)
    render_histogram(
        report.selected_course,
        report.house_order,
        output_path=output_path,
    )
    return format_histogram_report(report)


def _house_order(dataset: Dataset) -> tuple[str, ...]:
    return extract_house_labels(dataset, minimum_houses=1)


def _values_by_house(
    dataset: Dataset,
    column: str,
    house_order: Sequence[str],
) -> dict[str, list[float]]:
    grouped = {house: [] for house in house_order}
    for row_offset, row in enumerate(dataset.rows, start=2):
        house = row[TARGET_COLUMN]
        if house not in grouped:
            continue
        value = row[column]
        if value is None:
            continue
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise DatasetFormatError(
                f"{dataset.source_path}: column {column!r}, row {row_offset}: "
                f"invalid numeric value {value!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise DatasetFormatError(
                f"{dataset.source_path}: column {column!r}, row {row_offset}: "
                f"invalid numeric value {value!r}"
            )
        grouped[house].append(parsed)
    return grouped


def _has_all_house_samples(
    grouped: Mapping[str, Sequence[float]],
    house_order: Sequence[str],
) -> bool:
    for house in house_order:
        if len(grouped[house]) == 0:
            return False
    return True


def _pairwise_distances(
    grouped: Mapping[str, Sequence[float]],
    house_order: Sequence[str],
) -> tuple[PairwiseHouseDistance, ...]:
    distances: list[PairwiseHouseDistance] = []
    for left_index, house_a in enumerate(house_order):
        for house_b in house_order[left_index + 1 :]:
            distances.append(
                PairwiseHouseDistance(
                    house_a=house_a,
                    house_b=house_b,
                    distance=ecdf_max_distance(grouped[house_a], grouped[house_b]),
                )
            )
    return tuple(distances)
