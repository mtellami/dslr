"""Leak-free preprocessing state for DSLR training and prediction."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from dslr.data.csv_loader import Dataset
from dslr.errors import (
    DatasetFormatError,
    MissingColumnError,
    PreprocessingError,
    PreprocessingStateError,
    ZeroVarianceFeatureError,
)
from dslr.analysis.feature_selection import selected_model_features
from dslr.data.houses import HOUSE_COLUMN
from dslr.analysis.statistics import manual_mean

PREPROCESSING_SCHEMA_VERSION = 1
ZERO_VARIANCE_EPSILON = 1e-12
BIAS_VALUE = 1.0
IDENTIFIER_COLUMN = "Index"


@dataclass(frozen=True)
class FeaturePreprocessingState:
    """Fitted imputation and scaling parameters for one feature."""

    feature: str
    imputation_value: float
    scaling_mean: float
    scaling_scale: float


@dataclass(frozen=True)
class PreprocessingState:
    """Immutable fitted preprocessing contract."""

    schema_version: int
    features: tuple[str, ...]
    parameters: tuple[FeaturePreprocessingState, ...]
    add_bias: bool


@dataclass(frozen=True)
class TransformedMatrix:
    """Immutable transformed numeric matrix with explicit feature order."""

    features: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    includes_bias: bool

    @property
    def row_count(self) -> int:
        """Return the transformed row count."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Return the transformed matrix column count."""
        if not self.rows:
            return len(self.features) + (1 if self.includes_bias else 0)
        return len(self.rows[0])


def fit_preprocessor(
    dataset: Dataset,
    features: Iterable[str] | None = None,
    add_bias: bool = True,
) -> PreprocessingState:
    """Fit imputation and population z-score scaling on training data only."""
    if not isinstance(add_bias, bool):
        raise PreprocessingError(f"add_bias must be a bool, got {add_bias!r}")
    if dataset.row_count == 0:
        raise PreprocessingError(f"{dataset.source_path}: cannot fit an empty dataset")

    feature_tuple = _resolve_feature_contract(dataset, features)
    parameters: list[FeaturePreprocessingState] = []
    for feature in feature_tuple:
        observed = dataset.numeric_values(feature)
        if not observed:
            raise PreprocessingError(
                f"{dataset.source_path}: selected feature {feature!r} has no "
                "observed training values"
            )
        imputation_value = manual_mean(observed)
        if imputation_value is None or not math.isfinite(imputation_value):
            raise PreprocessingError(
                f"{dataset.source_path}: invalid imputation value for {feature!r}"
            )

        imputed_column = _training_imputed_column(dataset, feature, imputation_value)
        scaling_mean = manual_mean(imputed_column)
        if scaling_mean is None or not math.isfinite(scaling_mean):
            raise PreprocessingError(
                f"{dataset.source_path}: invalid scaling mean for {feature!r}"
            )
        scaling_scale = _population_scale(imputed_column, scaling_mean, feature)
        parameters.append(
            FeaturePreprocessingState(
                feature=feature,
                imputation_value=float(imputation_value),
                scaling_mean=float(scaling_mean),
                scaling_scale=float(scaling_scale),
            )
        )

    state = PreprocessingState(
        schema_version=PREPROCESSING_SCHEMA_VERSION,
        features=feature_tuple,
        parameters=tuple(parameters),
        add_bias=add_bias,
    )
    validate_preprocessing_state(state)
    return state


def transform_dataset(dataset: Dataset, state: PreprocessingState) -> TransformedMatrix:
    """Transform a dataset using only the stored fitted preprocessing state."""
    validate_preprocessing_state(state)
    dataset.validate_schema(state.features)

    rows: list[tuple[float, ...]] = []
    for row_offset, row in enumerate(dataset.rows, start=2):
        transformed: list[float] = []
        if state.add_bias:
            transformed.append(BIAS_VALUE)
        for parameter in state.parameters:
            value = row[parameter.feature]
            if value is None:
                parsed = parameter.imputation_value
            else:
                parsed = _parse_source_float(
                    value,
                    dataset,
                    parameter.feature,
                    row_offset,
                )
            standardized = (
                (parsed - parameter.scaling_mean) / parameter.scaling_scale
            )
            if not math.isfinite(standardized):
                raise PreprocessingError(
                    f"{dataset.source_path}: non-finite transformed value for "
                    f"{parameter.feature!r} at row {row_offset}"
                )
            transformed.append(float(standardized))
        rows.append(tuple(transformed))

    return TransformedMatrix(
        features=tuple(state.features),
        rows=tuple(rows),
        includes_bias=state.add_bias,
    )


def fit_transform_training(
    dataset: Dataset,
    features: Iterable[str] | None = None,
    add_bias: bool = True,
) -> tuple[PreprocessingState, TransformedMatrix]:
    """Fit on a training dataset and return the fitted state and matrix."""
    state = fit_preprocessor(dataset, features=features, add_bias=add_bias)
    return state, transform_dataset(dataset, state)


def validate_preprocessing_state(state: PreprocessingState) -> None:
    """Validate a preprocessing state without silently repairing it."""
    if not isinstance(state, PreprocessingState):
        raise PreprocessingStateError(
            f"expected PreprocessingState, got {type(state).__name__}"
        )
    if (
        isinstance(state.schema_version, bool)
        or not isinstance(state.schema_version, int)
        or state.schema_version != PREPROCESSING_SCHEMA_VERSION
    ):
        raise PreprocessingStateError(
            f"unsupported preprocessing schema version {state.schema_version!r}"
        )
    if not isinstance(state.add_bias, bool):
        raise PreprocessingStateError("add_bias must be a bool")
    _validate_feature_tuple(state.features, PreprocessingStateError)
    if not state.features:
        raise PreprocessingStateError("preprocessing state has no features")
    if not isinstance(state.parameters, tuple):
        raise PreprocessingStateError("parameters must be a tuple")
    if len(state.parameters) != len(state.features):
        raise PreprocessingStateError(
            "parameter count does not match feature count"
        )

    parameter_features: list[str] = []
    for index, parameter in enumerate(state.parameters):
        if not isinstance(parameter, FeaturePreprocessingState):
            raise PreprocessingStateError(
                f"parameter {index} must be FeaturePreprocessingState"
            )
        if parameter.feature != state.features[index]:
            raise PreprocessingStateError(
                "parameter order does not match feature order"
            )
        parameter_features.append(parameter.feature)
        _coerce_finite_number(
            parameter.imputation_value,
            f"{parameter.feature}.imputation_value",
            PreprocessingStateError,
        )
        _coerce_finite_number(
            parameter.scaling_mean,
            f"{parameter.feature}.scaling_mean",
            PreprocessingStateError,
        )
        scale = _coerce_finite_number(
            parameter.scaling_scale,
            f"{parameter.feature}.scaling_scale",
            PreprocessingStateError,
        )
        if scale <= ZERO_VARIANCE_EPSILON:
            raise PreprocessingStateError(
                f"{parameter.feature!r} scaling scale is too small: {scale!r}"
            )
    _reject_duplicates(tuple(parameter_features), "parameter", PreprocessingStateError)


def preprocessing_state_to_dict(state: PreprocessingState) -> dict[str, Any]:
    """Return a fresh JSON-compatible payload for a preprocessing state."""
    validate_preprocessing_state(state)
    return {
        "schema_version": state.schema_version,
        "features": list(state.features),
        "parameters": [
            {
                "feature": parameter.feature,
                "imputation_value": float(parameter.imputation_value),
                "scaling_mean": float(parameter.scaling_mean),
                "scaling_scale": float(parameter.scaling_scale),
            }
            for parameter in state.parameters
        ],
        "add_bias": state.add_bias,
    }


def preprocessing_state_from_dict(payload: dict[str, Any]) -> PreprocessingState:
    """Build and validate a preprocessing state from a JSON-compatible payload."""
    if not isinstance(payload, dict):
        raise PreprocessingStateError(
            f"preprocessing payload must be a dict, got {type(payload).__name__}"
        )
    required_keys = {"schema_version", "features", "parameters", "add_bias"}
    actual_keys = set(payload)
    missing = required_keys - actual_keys
    extra = actual_keys - required_keys
    if missing:
        raise PreprocessingStateError(
            f"preprocessing payload missing keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise PreprocessingStateError(
            f"preprocessing payload has unexpected keys: {', '.join(sorted(extra))}"
        )
    if not isinstance(payload["features"], list):
        raise PreprocessingStateError("features must be a list")
    if not isinstance(payload["parameters"], list):
        raise PreprocessingStateError("parameters must be a list")

    features = tuple(payload["features"])
    _validate_feature_tuple(features, PreprocessingStateError)

    parameters: list[FeaturePreprocessingState] = []
    parameter_keys = {
        "feature",
        "imputation_value",
        "scaling_mean",
        "scaling_scale",
    }
    for index, entry in enumerate(payload["parameters"]):
        if not isinstance(entry, dict):
            raise PreprocessingStateError(
                f"parameter {index} must be a dict, got {type(entry).__name__}"
            )
        entry_keys = set(entry)
        missing = parameter_keys - entry_keys
        extra = entry_keys - parameter_keys
        if missing:
            raise PreprocessingStateError(
                f"parameter {index} missing keys: {', '.join(sorted(missing))}"
            )
        if extra:
            raise PreprocessingStateError(
                f"parameter {index} has unexpected keys: {', '.join(sorted(extra))}"
            )
        parameters.append(
            FeaturePreprocessingState(
                feature=entry["feature"],
                imputation_value=_coerce_finite_number(
                    entry["imputation_value"],
                    f"parameter {index} imputation_value",
                    PreprocessingStateError,
                ),
                scaling_mean=_coerce_finite_number(
                    entry["scaling_mean"],
                    f"parameter {index} scaling_mean",
                    PreprocessingStateError,
                ),
                scaling_scale=_coerce_finite_number(
                    entry["scaling_scale"],
                    f"parameter {index} scaling_scale",
                    PreprocessingStateError,
                ),
            )
        )

    state = PreprocessingState(
        schema_version=payload["schema_version"],
        features=features,
        parameters=tuple(parameters),
        add_bias=payload["add_bias"],
    )
    validate_preprocessing_state(state)
    return state


def _resolve_feature_contract(
    dataset: Dataset,
    features: Iterable[str] | None,
) -> tuple[str, ...]:
    if features is None:
        feature_tuple = tuple(selected_model_features(dataset))
    else:
        feature_tuple = tuple(features)
        _validate_explicit_features(dataset, feature_tuple)
    if not feature_tuple:
        raise PreprocessingError("preprocessing requires at least one feature")
    return feature_tuple


def _validate_explicit_features(dataset: Dataset, features: tuple[str, ...]) -> None:
    _validate_feature_tuple(features, PreprocessingError)
    if not features:
        raise PreprocessingError("explicit feature contract is empty")
    dataset.validate_schema(features)

    numeric_columns = set(dataset.detect_numeric_columns(excluded=()))
    for feature in features:
        if feature == IDENTIFIER_COLUMN:
            raise PreprocessingError("Index cannot be a model feature")
        if feature == HOUSE_COLUMN:
            raise PreprocessingError("Hogwarts House cannot be a model feature")
        if feature not in numeric_columns:
            raise PreprocessingError(f"{feature!r} is not a usable numerical feature")

    source_positions = [dataset.headers.index(feature) for feature in features]
    if source_positions != sorted(source_positions):
        raise PreprocessingError(
            "explicit feature contract must preserve source-column order"
        )


def _validate_feature_tuple(values: tuple[str, ...], error_type: type[Exception]) -> None:
    if not isinstance(values, tuple):
        raise error_type("features must be a tuple")
    for value in values:
        if not isinstance(value, str):
            raise error_type(f"feature names must be strings, got {value!r}")
        if value == "":
            raise error_type("feature names must not be empty")
        if value == IDENTIFIER_COLUMN:
            raise error_type("Index cannot be a model feature")
        if value == HOUSE_COLUMN:
            raise error_type("Hogwarts House cannot be a model feature")
    _reject_duplicates(values, "feature", error_type)


def _training_imputed_column(
    dataset: Dataset,
    feature: str,
    imputation_value: float,
) -> tuple[float, ...]:
    values: list[float] = []
    for row_offset, row in enumerate(dataset.rows, start=2):
        value = row[feature]
        if value is None:
            values.append(float(imputation_value))
        else:
            values.append(_parse_source_float(value, dataset, feature, row_offset))
    return tuple(values)


def _population_scale(
    values: tuple[float, ...],
    scaling_mean: float,
    feature: str,
) -> float:
    if not values:
        raise PreprocessingError(f"{feature!r} has no values to scale")
    squared_total = 0.0
    count = 0
    for value in values:
        delta = value - scaling_mean
        squared_total += delta * delta
        count += 1
    variance = squared_total / count
    scale = math.sqrt(variance)
    if not math.isfinite(scale):
        raise PreprocessingError(f"{feature!r} has a non-finite scaling scale")
    if scale <= ZERO_VARIANCE_EPSILON:
        raise ZeroVarianceFeatureError(
            f"{feature!r} has zero or near-zero variance after imputation"
        )
    return scale


def _parse_source_float(
    value: str,
    dataset: Dataset,
    feature: str,
    row_offset: int,
) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise DatasetFormatError(
            f"{dataset.source_path}: column {feature!r}, row {row_offset}: "
            f"invalid numeric value {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise DatasetFormatError(
            f"{dataset.source_path}: column {feature!r}, row {row_offset}: "
            f"non-finite numeric value {value!r}"
        )
    return parsed


def _coerce_finite_number(
    value: Any,
    label: str,
    error_type: type[Exception],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{label} must be a finite number, got {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise error_type(f"{label} must be finite, got {value!r}")
    return parsed


def _reject_duplicates(
    values: tuple[str, ...],
    label: str,
    error_type: type[Exception],
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise error_type(f"duplicate {label} names: {', '.join(duplicates)}")
