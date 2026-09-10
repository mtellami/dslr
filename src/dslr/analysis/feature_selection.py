"""Final model feature-selection policy derived from visualization evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dslr.data.csv_loader import Dataset
from dslr.errors import FeatureSelectionError
from dslr.analysis.histogram import analyze_histogram_dataset
from dslr.data.houses import extract_house_labels
from dslr.analysis.scatter import rank_feature_pairs, select_scatter_columns

HOMOGENEITY_EXCLUSION_THRESHOLD = 0.10
REDUNDANCY_ABSOLUTE_CORRELATION_THRESHOLD = 0.999999

LOW_HOUSE_DISCRIMINATION = "low_house_discrimination"
REDUNDANT_FEATURE = "redundant_feature"


@dataclass(frozen=True)
class FeatureExclusion:
    """Evidence explaining why one candidate feature was excluded."""

    feature: str
    reason: str
    evidence_value: float
    related_feature: str | None = None


@dataclass(frozen=True)
class FeatureSelectionDecision:
    """Frozen model-feature contract derived from the labeled training data."""

    candidates: tuple[str, ...]
    selected: tuple[str, ...]
    excluded: tuple[FeatureExclusion, ...]
    homogeneity_threshold: float
    redundancy_threshold: float


def validate_selection_thresholds(
    homogeneity_threshold: float = HOMOGENEITY_EXCLUSION_THRESHOLD,
    redundancy_threshold: float = REDUNDANCY_ABSOLUTE_CORRELATION_THRESHOLD,
) -> None:
    """Validate project feature-selection thresholds."""
    _validate_probability_threshold(homogeneity_threshold, "homogeneity_threshold")
    _validate_probability_threshold(redundancy_threshold, "redundancy_threshold")


def derive_model_feature_selection(dataset: Dataset) -> FeatureSelectionDecision:
    """Derive the final ordered model-feature decision from training labels."""
    validate_selection_thresholds()
    extract_house_labels(dataset)
    candidates = tuple(select_scatter_columns(dataset))
    if len(candidates) < 2:
        raise FeatureSelectionError(
            f"{dataset.source_path}: feature selection requires at least two "
            "candidate course features"
        )

    excluded: list[FeatureExclusion] = []
    excluded_names: set[str] = set()

    histogram_report = analyze_histogram_dataset(dataset, columns=candidates)
    for analysis in histogram_report.ranking:
        score = analysis.average_pairwise_distance
        if score <= HOMOGENEITY_EXCLUSION_THRESHOLD:
            excluded.append(
                FeatureExclusion(
                    feature=analysis.course,
                    reason=LOW_HOUSE_DISCRIMINATION,
                    evidence_value=score,
                )
            )
            excluded_names.add(analysis.course)

    candidate_index = {feature: index for index, feature in enumerate(candidates)}
    for pair in rank_feature_pairs(dataset):
        if pair.absolute_correlation < REDUNDANCY_ABSOLUTE_CORRELATION_THRESHOLD:
            continue
        left_excluded = pair.x_feature in excluded_names
        right_excluded = pair.y_feature in excluded_names
        if left_excluded or right_excluded:
            continue
        if candidate_index[pair.x_feature] <= candidate_index[pair.y_feature]:
            retained = pair.x_feature
            removed = pair.y_feature
        else:
            retained = pair.y_feature
            removed = pair.x_feature
        excluded.append(
            FeatureExclusion(
                feature=removed,
                reason=REDUNDANT_FEATURE,
                evidence_value=pair.absolute_correlation,
                related_feature=retained,
            )
        )
        excluded_names.add(removed)

    selected = tuple(feature for feature in candidates if feature not in excluded_names)
    decision = FeatureSelectionDecision(
        candidates=candidates,
        selected=selected,
        excluded=tuple(excluded),
        homogeneity_threshold=HOMOGENEITY_EXCLUSION_THRESHOLD,
        redundancy_threshold=REDUNDANCY_ABSOLUTE_CORRELATION_THRESHOLD,
    )
    validate_feature_selection(dataset, decision)
    return decision


def validate_feature_selection(
    dataset: Dataset,
    decision: FeatureSelectionDecision,
) -> None:
    """Validate a derived feature-selection contract against a dataset."""
    validate_selection_thresholds(
        decision.homogeneity_threshold,
        decision.redundancy_threshold,
    )
    _reject_duplicates(decision.candidates, "candidate")
    _reject_duplicates(decision.selected, "selected")

    numerical_columns = tuple(dataset.detect_numeric_columns())
    numerical_set = set(numerical_columns)
    candidate_set = set(decision.candidates)
    selected_set = set(decision.selected)
    excluded_features = tuple(exclusion.feature for exclusion in decision.excluded)
    excluded_set = set(excluded_features)
    _reject_duplicates(excluded_features, "excluded")

    for feature in decision.candidates:
        if feature not in dataset.headers:
            raise FeatureSelectionError(f"candidate feature {feature!r} is absent")
        if feature not in numerical_set:
            raise FeatureSelectionError(f"{feature!r} is not a numerical candidate")

    for feature in decision.selected:
        if feature not in dataset.headers:
            raise FeatureSelectionError(f"selected feature {feature!r} is absent")
        if feature not in candidate_set:
            raise FeatureSelectionError(
                f"selected feature {feature!r} is absent from candidates"
            )
        if feature not in numerical_set:
            raise FeatureSelectionError(f"selected feature {feature!r} is not numerical")

    overlap = selected_set & excluded_set
    if overlap:
        names = ", ".join(sorted(overlap))
        raise FeatureSelectionError(f"selected features also excluded: {names}")

    for exclusion in decision.excluded:
        if exclusion.feature not in candidate_set:
            raise FeatureSelectionError(
                f"excluded feature {exclusion.feature!r} is absent from candidates"
            )
        if exclusion.reason == REDUNDANT_FEATURE:
            if exclusion.related_feature is None:
                raise FeatureSelectionError(
                    f"redundant exclusion {exclusion.feature!r} lacks related feature"
                )
            if exclusion.related_feature not in candidate_set:
                raise FeatureSelectionError(
                    f"related feature {exclusion.related_feature!r} is absent "
                    "from candidates"
                )
        elif exclusion.reason != LOW_HOUSE_DISCRIMINATION:
            raise FeatureSelectionError(f"unsupported exclusion reason {exclusion.reason!r}")
        if not math.isfinite(exclusion.evidence_value):
            raise FeatureSelectionError(
                f"invalid evidence value for {exclusion.feature!r}: "
                f"{exclusion.evidence_value!r}"
            )

    candidate_order = {feature: index for index, feature in enumerate(decision.candidates)}
    selected_order = [candidate_order[feature] for feature in decision.selected]
    if selected_order != sorted(selected_order):
        raise FeatureSelectionError("selected features do not preserve source order")
    if not decision.selected:
        raise FeatureSelectionError("feature selection removed every candidate")


def selected_model_features(dataset: Dataset) -> tuple[str, ...]:
    """Return the frozen model feature tuple for a labeled training dataset."""
    decision = derive_model_feature_selection(dataset)
    validate_feature_selection(dataset, decision)
    return tuple(decision.selected)


def _validate_probability_threshold(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureSelectionError(f"{name} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise FeatureSelectionError(f"{name} must be between 0 and 1, got {value!r}")


def _reject_duplicates(values: tuple[str, ...], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(duplicates)
        raise FeatureSelectionError(f"duplicate {label} features: {joined}")
