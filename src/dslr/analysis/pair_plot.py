"""Pair-plot matrix and final feature-selection reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from dslr.data.csv_loader import Dataset, load_csv
from dslr.errors import DatasetFormatError, PairPlotError, VisualizationError
from dslr.analysis.feature_selection import (
    FeatureSelectionDecision,
    derive_model_feature_selection,
)
from dslr.analysis.histogram import shared_histogram_bins
from dslr.data.houses import extract_house_labels
from dslr.analysis.plotting import import_pyplot
from dslr.analysis.scatter import (
    FeaturePairSimilarity,
    extract_paired_points_by_house,
    rank_feature_pairs,
    select_scatter_columns,
)

PAIR_PLOT_CELL_SIZE = 1.9
PAIR_PLOT_POINT_SIZE = 4.0
PAIR_PLOT_ALPHA = 0.35
PAIR_PLOT_HISTOGRAM_BINS = 15
PAIR_PLOT_LABEL_WRAP_WIDTH = 18


@dataclass(frozen=True)
class PairPlotAnalysis:
    """Complete pair-plot analysis and final feature selection."""

    features: tuple[str, ...]
    houses: tuple[str, ...]
    pair_ranking: tuple[FeaturePairSimilarity, ...]
    selection: FeatureSelectionDecision


def analyze_pair_plot_dataset(dataset: Dataset) -> PairPlotAnalysis:
    """Return pair-plot analysis without rendering or printing."""
    houses = extract_house_labels(dataset)
    _reject_malformed_mixed_numeric_columns(dataset)
    features = tuple(select_scatter_columns(dataset))
    if len(features) < 2:
        raise PairPlotError(
            f"{dataset.source_path}: pair plot requires at least two numerical "
            "course features"
        )
    ranking = rank_feature_pairs(dataset)
    selection = derive_model_feature_selection(dataset)
    return PairPlotAnalysis(
        features=features,
        houses=houses,
        pair_ranking=ranking,
        selection=selection,
    )


def analyze_pair_plot_file(path: str | Path) -> tuple[Dataset, PairPlotAnalysis]:
    """Load a CSV file and return the dataset plus pair-plot analysis."""
    dataset = load_csv(path)
    return dataset, analyze_pair_plot_dataset(dataset)


def create_pair_plot_figure(dataset: Dataset, analysis: PairPlotAnalysis):
    """Create the pair-plot Figure without showing, saving, or closing it."""
    pyplot = import_pyplot()
    features = analysis.features
    feature_count = len(features)
    if feature_count < 2:
        raise PairPlotError("pair plot matrix requires at least two features")

    figure_size = (
        PAIR_PLOT_CELL_SIZE * feature_count,
        PAIR_PLOT_CELL_SIZE * feature_count,
    )
    figure, axes_grid = pyplot.subplots(
        feature_count,
        feature_count,
        figsize=figure_size,
        squeeze=False,
    )
    pair_lookup = _pair_lookup(analysis.pair_ranking)

    for row_index, y_feature in enumerate(features):
        for column_index, x_feature in enumerate(features):
            axes = axes_grid[row_index][column_index]
            if row_index == column_index:
                _draw_diagonal_histogram(dataset, axes, x_feature, analysis.houses)
            elif row_index > column_index:
                _draw_lower_scatter(
                    dataset,
                    axes,
                    x_feature,
                    y_feature,
                    analysis.houses,
                )
            else:
                _draw_upper_evidence(axes, x_feature, y_feature, pair_lookup)

            if row_index == feature_count - 1:
                axes.set_xlabel(_wrapped_label(x_feature), fontsize=7)
            else:
                axes.set_xlabel("")
                axes.set_xticklabels([])
            if column_index == 0:
                axes.set_ylabel(_wrapped_label(y_feature), fontsize=7)
            else:
                axes.set_ylabel("")
                axes.set_yticklabels([])
            axes.tick_params(labelsize=6)

    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            fontsize=8,
        )
    figure.suptitle(
        "Hogwarts course pair plot "
        f"({len(analysis.selection.selected)} selected model features)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    return figure


def render_pair_plot(
    dataset: Dataset,
    analysis: PairPlotAnalysis,
    save_path: str | Path | None = None,
) -> Path | None:
    """Render the pair plot interactively or save it to a requested path."""
    pyplot = import_pyplot()
    figure = create_pair_plot_figure(dataset, analysis)
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
        raise VisualizationError(f"{destination}: could not save pair plot image") from exc
    finally:
        pyplot.close(figure)


def format_pair_plot_report(
    analysis: PairPlotAnalysis,
    saved_path: str | Path | None = None,
) -> str:
    """Return deterministic text for the pair-plot feature decision."""
    selected = ", ".join(analysis.selection.selected)
    lines = [
        f"Pair plot course features: {len(analysis.features)}",
        f"Evaluated feature pairs: {len(analysis.pair_ranking)}",
        "",
        f"Selected logistic-regression features ({len(analysis.selection.selected)}):",
        selected,
        "",
        "Excluded features:",
    ]
    for exclusion in analysis.selection.excluded:
        if exclusion.related_feature is None:
            lines.append(
                f"{exclusion.feature} - {exclusion.reason} "
                f"({exclusion.evidence_value:.15f})"
            )
        else:
            lines.append(
                f"{exclusion.feature} - {exclusion.reason} with "
                f"{exclusion.related_feature} ({exclusion.evidence_value:.15f})"
            )
    if saved_path is not None:
        lines.extend(["", f"Pair plot saved to: {Path(saved_path)}"])
    return "\n".join(lines)


def create_pair_plot(path: str | Path, save_path: str | Path | None = None) -> str:
    """Analyze, render, and return a deterministic pair-plot report."""
    dataset, analysis = analyze_pair_plot_file(path)
    rendered_path = render_pair_plot(dataset, analysis, save_path=save_path)
    return format_pair_plot_report(analysis, saved_path=rendered_path)


def _draw_diagonal_histogram(dataset, axes, feature: str, houses: tuple[str, ...]) -> None:
    values_by_house = _values_by_house(dataset, feature, houses)
    bins = shared_histogram_bins(values_by_house, bin_count=PAIR_PLOT_HISTOGRAM_BINS)
    for house in houses:
        values = values_by_house[house]
        if not values:
            continue
        weights = [1.0 / len(values) for _value in values]
        axes.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=0.9,
            label=house,
            density=False,
            stacked=False,
        )
    axes.grid(alpha=0.18)


def _draw_lower_scatter(
    dataset: Dataset,
    axes,
    x_feature: str,
    y_feature: str,
    houses: tuple[str, ...],
) -> None:
    grouped = extract_paired_points_by_house(dataset, x_feature, y_feature, houses)
    for house, points in grouped:
        axes.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            s=PAIR_PLOT_POINT_SIZE,
            alpha=PAIR_PLOT_ALPHA,
            label=house,
        )
    axes.grid(alpha=0.18)


def _draw_upper_evidence(
    axes,
    x_feature: str,
    y_feature: str,
    pair_lookup: dict[frozenset[str], FeaturePairSimilarity],
) -> None:
    pair = pair_lookup[frozenset((x_feature, y_feature))]
    axes.text(
        0.5,
        0.5,
        f"r = {pair.correlation:.3f}\nn = {pair.paired_count}",
        ha="center",
        va="center",
        transform=axes.transAxes,
        fontsize=8,
    )
    axes.set_xticks([])
    axes.set_yticks([])


def _values_by_house(
    dataset: Dataset,
    feature: str,
    houses: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    dataset.require_column(feature)
    dataset.require_column("Hogwarts House")
    grouped: dict[str, list[float]] = {house: [] for house in houses}
    for row_offset, row in enumerate(dataset.rows, start=2):
        house = row["Hogwarts House"]
        if house is None or house.strip() == "":
            raise VisualizationError(
                f"{dataset.source_path}: missing house label at row {row_offset}"
            )
        if house not in grouped:
            continue
        raw_value = row[feature]
        if raw_value is None:
            continue
        grouped[house].append(_parse_float(dataset, feature, raw_value, row_offset))
    return {house: tuple(grouped[house]) for house in houses}


def _parse_float(dataset: Dataset, column: str, raw_value: str, row_offset: int) -> float:
    try:
        value = float(raw_value.strip())
    except ValueError as exc:
        raise DatasetFormatError(
            f"{dataset.source_path}: column {column!r}, row {row_offset}: "
            f"invalid numeric value {raw_value!r}"
        ) from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise DatasetFormatError(
            f"{dataset.source_path}: column {column!r}, row {row_offset}: "
            f"invalid numeric value {raw_value!r}"
        )
    return value


def _reject_malformed_mixed_numeric_columns(dataset: Dataset) -> None:
    for column in dataset.headers:
        if column in {"Index", "Hogwarts House"}:
            continue
        numeric_seen = False
        malformed_seen: tuple[int, str] | None = None
        for row_offset, row in enumerate(dataset.rows, start=2):
            raw_value = row[column]
            if raw_value is None:
                continue
            try:
                value = float(raw_value.strip())
            except ValueError:
                malformed_seen = (row_offset, raw_value)
                continue
            if value != value or value in (float("inf"), float("-inf")):
                malformed_seen = (row_offset, raw_value)
                continue
            numeric_seen = True
        if numeric_seen and malformed_seen is not None:
            row_offset, raw_value = malformed_seen
            raise DatasetFormatError(
                f"{dataset.source_path}: column {column!r}, row {row_offset}: "
                f"invalid numeric value {raw_value!r}"
            )


def _pair_lookup(
    ranking: tuple[FeaturePairSimilarity, ...],
) -> dict[frozenset[str], FeaturePairSimilarity]:
    return {frozenset((pair.x_feature, pair.y_feature)): pair for pair in ranking}


def _wrapped_label(label: str) -> str:
    return "\n".join(wrap(label, width=PAIR_PLOT_LABEL_WRAP_WIDTH)) or label
