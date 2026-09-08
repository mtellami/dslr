"""Scatter-plot analysis for similar Hogwarts course features."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import DatasetFormatError, VisualizationError
from dslr.data.houses import HOUSE_COLUMN, extract_house_labels
from dslr.analysis.plotting import import_pyplot

DEFAULT_SCATTER_EXCLUDED_COLUMNS = frozenset(("Index", HOUSE_COLUMN))
MIN_PAIRED_POINTS = 2


@dataclass(frozen=True)
class FeaturePairSimilarity:
    """Manual Pearson similarity result for one feature pair."""

    x_feature: str
    y_feature: str
    correlation: float
    absolute_correlation: float
    paired_count: int


@dataclass(frozen=True)
class ScatterAnalysis:
    """Complete scatter analysis for the strongest feature pair."""

    selected: FeaturePairSimilarity
    ranking: tuple[FeaturePairSimilarity, ...]
    houses: tuple[str, ...]
    grouped_points: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]


def select_scatter_columns(
    dataset: Dataset,
    excluded: Iterable[str] | None = None,
) -> list[str]:
    """Return numerical scatter candidates in source order."""
    exclusions = set(DEFAULT_SCATTER_EXCLUDED_COLUMNS)
    if excluded is not None:
        exclusions.update(excluded)

    selected: list[str] = []
    for column in dataset.detect_numeric_columns():
        if column not in exclusions:
            selected.append(column)
    return selected


def extract_paired_points_by_house(
    dataset: Dataset,
    x_feature: str,
    y_feature: str,
    houses: Sequence[str],
) -> tuple[tuple[str, tuple[tuple[float, float], ...]], ...]:
    """Return finite paired points for two features, grouped by requested houses."""
    if x_feature == y_feature:
        raise VisualizationError("scatter plot requires two distinct features")
    dataset.require_column(HOUSE_COLUMN)
    dataset.require_column(x_feature)
    dataset.require_column(y_feature)

    grouped = {house: [] for house in houses}
    for row_offset, row in enumerate(dataset.rows, start=2):
        house = row[HOUSE_COLUMN]
        if house is None or house.strip() == "":
            raise VisualizationError(
                f"{dataset.source_path}: missing house label at row {row_offset}"
            )
        if house not in grouped:
            continue

        x_raw = row[x_feature]
        y_raw = row[y_feature]
        if x_raw is None or y_raw is None:
            continue

        x_value = _parse_dataset_float(dataset, x_feature, x_raw, row_offset)
        y_value = _parse_dataset_float(dataset, y_feature, y_raw, row_offset)
        grouped[house].append((x_value, y_value))

    return tuple((house, tuple(grouped[house])) for house in houses)


def manual_pearson_correlation(x_values, y_values) -> float:
    """Return manual Pearson product-moment correlation for paired values."""
    x_items = _validated_numeric_values(x_values, "x")
    y_items = _validated_numeric_values(y_values, "y")
    if len(x_items) != len(y_items):
        raise VisualizationError(
            f"correlation requires equal lengths; got {len(x_items)} and {len(y_items)}"
        )
    count = len(x_items)
    if count < MIN_PAIRED_POINTS:
        raise VisualizationError(
            f"correlation requires at least {MIN_PAIRED_POINTS} paired points"
        )

    total_x = 0.0
    total_y = 0.0
    for index in range(count):
        total_x += x_items[index]
        total_y += y_items[index]
    mean_x = total_x / count
    mean_y = total_y / count

    numerator = 0.0
    x_square_total = 0.0
    y_square_total = 0.0
    for index in range(count):
        x_delta = x_items[index] - mean_x
        y_delta = y_items[index] - mean_y
        numerator += x_delta * y_delta
        x_square_total += x_delta * x_delta
        y_square_total += y_delta * y_delta

    if x_square_total == 0.0:
        raise VisualizationError("correlation is undefined for constant x values")
    if y_square_total == 0.0:
        raise VisualizationError("correlation is undefined for constant y values")

    denominator = math.sqrt(x_square_total * y_square_total)
    if denominator == 0.0:
        raise VisualizationError("correlation denominator is zero")
    correlation = numerator / denominator
    if correlation > 1.0 and correlation <= 1.0 + 1e-12:
        correlation = 1.0
    elif correlation < -1.0 and correlation >= -1.0 - 1e-12:
        correlation = -1.0
    if not math.isfinite(correlation) or correlation < -1.0 or correlation > 1.0:
        raise VisualizationError(f"invalid correlation result {correlation!r}")
    return correlation


def calculate_pair_similarity(
    dataset: Dataset,
    x_feature: str,
    y_feature: str,
) -> FeaturePairSimilarity:
    """Calculate signed and absolute manual Pearson similarity for one pair."""
    houses = extract_house_labels(dataset)
    return _calculate_pair_similarity(dataset, x_feature, y_feature, houses)


def rank_feature_pairs(dataset: Dataset) -> tuple[FeaturePairSimilarity, ...]:
    """Rank every eligible feature pair by descending absolute correlation."""
    houses = extract_house_labels(dataset)
    columns = select_scatter_columns(dataset)
    if len(columns) < 2:
        raise VisualizationError(
            f"{dataset.source_path}: scatter analysis requires at least two "
            "numerical course columns"
        )

    ranked: list[tuple[int, FeaturePairSimilarity]] = []
    pair_order = 0
    for left_index, x_feature in enumerate(columns):
        for y_feature in columns[left_index + 1 :]:
            try:
                similarity = _calculate_pair_similarity(
                    dataset,
                    x_feature,
                    y_feature,
                    houses,
                )
            except VisualizationError:
                pair_order += 1
                continue
            ranked.append((pair_order, similarity))
            pair_order += 1

    if not ranked:
        raise VisualizationError(f"{dataset.source_path}: no eligible scatter pairs")

    ranked.sort(
        key=lambda item: (
            -item[1].absolute_correlation,
            item[0],
        )
    )
    return tuple(item[1] for item in ranked)


def select_most_similar_pair(dataset: Dataset) -> FeaturePairSimilarity:
    """Return the strongest absolute-correlation feature pair."""
    return rank_feature_pairs(dataset)[0]


def analyze_scatter_dataset(dataset: Dataset) -> ScatterAnalysis:
    """Return complete scatter analysis without plotting or printing."""
    houses = extract_house_labels(dataset)
    ranking = rank_feature_pairs(dataset)
    selected = ranking[0]
    grouped = extract_paired_points_by_house(
        dataset,
        selected.x_feature,
        selected.y_feature,
        houses,
    )
    total_points = 0
    non_empty_houses = 0
    for _house, points in grouped:
        if points:
            non_empty_houses += 1
            total_points += len(points)
    if total_points < MIN_PAIRED_POINTS or non_empty_houses < 2:
        raise VisualizationError(
            f"{dataset.source_path}: selected scatter pair has insufficient points"
        )
    return ScatterAnalysis(
        selected=selected,
        ranking=ranking,
        houses=houses,
        grouped_points=grouped,
    )


def analyze_scatter_file(path: str | Path) -> ScatterAnalysis:
    """Load a CSV file and return complete scatter analysis."""
    return analyze_scatter_dataset(load_csv(path))


def create_scatter_figure(analysis: ScatterAnalysis):
    """Create the scatter Figure without showing, saving, or closing it."""
    pyplot = import_pyplot()
    figure, axes = pyplot.subplots(figsize=(9, 7))
    for house, points in analysis.grouped_points:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        axes.scatter(
            x_values,
            y_values,
            s=28,
            alpha=0.65,
            label=house,
        )
    axes.set_xlabel(analysis.selected.x_feature)
    axes.set_ylabel(analysis.selected.y_feature)
    axes.set_title(
        f"{analysis.selected.x_feature} vs {analysis.selected.y_feature} "
        f"(Pearson r={analysis.selected.correlation:.6f})"
    )
    axes.legend()
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def render_scatter(
    analysis: ScatterAnalysis,
    save_path: str | Path | None = None,
) -> Path | None:
    """Render the scatter figure interactively or save it to a requested path."""
    pyplot = import_pyplot()
    figure = create_scatter_figure(analysis)

    if save_path is None:
        try:
            pyplot.show()
            return None
        finally:
            pyplot.close(figure)

    destination = Path(save_path)
    if destination.exists() and not destination.is_file():
        pyplot.close(figure)
        raise VisualizationError(f"{destination}: output path is not a regular file")

    existed_before = destination.exists()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image_format = destination.suffix[1:] if destination.suffix else None
        figure.savefig(destination, format=image_format, dpi=120)
        return destination
    except (OSError, ValueError) as exc:
        if not existed_before and destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass
        raise VisualizationError(f"{destination}: could not save scatter image") from exc
    finally:
        pyplot.close(figure)


def format_scatter_report(
    analysis: ScatterAnalysis,
    saved_path: str | Path | None = None,
) -> str:
    """Return deterministic text describing the selected scatter pair."""
    lines = [
        (
            "Most similar features: "
            f"{analysis.selected.x_feature} and {analysis.selected.y_feature}"
        ),
        f"Pearson correlation: {analysis.selected.correlation:.6f}",
        f"Paired observations: {analysis.selected.paired_count}",
    ]
    if saved_path is not None:
        lines.append(f"Scatter plot saved to: {Path(saved_path)}")
    return "\n".join(lines)


def create_scatter(path: str | Path, save_path: str | Path | None = None) -> str:
    """Analyze, render, and return a deterministic scatter report."""
    analysis = analyze_scatter_file(path)
    rendered_path = render_scatter(analysis, save_path=save_path)
    return format_scatter_report(analysis, saved_path=rendered_path)


def _calculate_pair_similarity(
    dataset: Dataset,
    x_feature: str,
    y_feature: str,
    houses: Sequence[str],
) -> FeaturePairSimilarity:
    grouped = extract_paired_points_by_house(dataset, x_feature, y_feature, houses)
    x_values: list[float] = []
    y_values: list[float] = []
    for _house, points in grouped:
        for x_value, y_value in points:
            x_values.append(x_value)
            y_values.append(y_value)
    correlation = manual_pearson_correlation(x_values, y_values)
    return FeaturePairSimilarity(
        x_feature=x_feature,
        y_feature=y_feature,
        correlation=correlation,
        absolute_correlation=abs(correlation),
        paired_count=len(x_values),
    )


def _parse_dataset_float(
    dataset: Dataset,
    column: str,
    raw_value: str,
    row_offset: int,
) -> float:
    try:
        value = float(raw_value.strip())
    except ValueError as exc:
        raise DatasetFormatError(
            f"{dataset.source_path}: column {column!r}, row {row_offset}: "
            f"invalid numeric value {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise DatasetFormatError(
            f"{dataset.source_path}: column {column!r}, row {row_offset}: "
            f"invalid numeric value {raw_value!r}"
        )
    return value


def _validated_numeric_values(values, label: str) -> list[float]:
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise VisualizationError(
                f"{label}: unsupported bool value {value!r}"
            )
        if not isinstance(value, (int, float)):
            raise VisualizationError(
                f"{label}: unsupported value {value!r} of type {type(value).__name__}"
            )
        parsed_value = float(value)
        if not math.isfinite(parsed_value):
            raise VisualizationError(f"{label}: non-finite value {value!r}")
        parsed.append(parsed_value)
    return parsed
