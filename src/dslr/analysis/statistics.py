"""Manual descriptive statistics for the DSLR project."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from dslr.errors import StatisticsError


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Descriptive-statistics fields for one numerical series."""

    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    q1: float | None
    median: float | None
    q3: float | None
    maximum: float | None
    sample_variance: float | None = None
    value_range: float | None = None
    iqr: float | None = None
    missing_count: int = 0
    skewness: float | None = None


def manual_count(values: Iterable[int | float | None]) -> int:
    """Count non-missing finite numerical values with an explicit loop."""
    count = 0
    for position, value in enumerate(values):
        if value is None:
            continue
        _validate_number(value, position)
        count += 1
    return count


def manual_sum(values: Iterable[int | float | None]) -> float:
    """Sum non-missing finite numerical values with an explicit accumulator."""
    total = 0.0
    for position, value in enumerate(values):
        if value is None:
            continue
        total += _validate_number(value, position)
    return total


def manual_mean(values: Iterable[int | float | None]) -> float | None:
    """Return the arithmetic mean, or None when no values are present."""
    total = 0.0
    count = 0
    for position, value in enumerate(values):
        if value is None:
            continue
        total += _validate_number(value, position)
        count += 1
    if count == 0:
        return None
    return total / count


def manual_std(values: Iterable[int | float | None]) -> float | None:
    """Return sample standard deviation using n - 1 in the denominator."""
    cleaned = _validated_values(values)
    return _std_validated(cleaned)


def manual_sample_variance(values: Iterable[int | float | None]) -> float | None:
    """Return sample variance using n - 1 in the denominator."""
    cleaned = _validated_values(values)
    return _sample_variance_validated(cleaned)


def manual_min(values: Iterable[int | float | None]) -> float | None:
    """Return the minimum non-missing value with an explicit loop."""
    current = None
    seen_value = False
    for position, value in enumerate(values):
        if value is None:
            continue
        parsed = _validate_number(value, position)
        if not seen_value or parsed < current:
            current = parsed
            seen_value = True
    return current


def manual_max(values: Iterable[int | float | None]) -> float | None:
    """Return the maximum non-missing value with an explicit loop."""
    current = None
    seen_value = False
    for position, value in enumerate(values):
        if value is None:
            continue
        parsed = _validate_number(value, position)
        if not seen_value or parsed > current:
            current = parsed
            seen_value = True
    return current


def manual_value_range(values: Iterable[int | float | None]) -> float | None:
    """Return maximum minus minimum for non-missing values."""
    cleaned = _validated_values(values)
    return _range_validated(cleaned)


def manual_quantile(
    values: Iterable[int | float | None],
    probability: int | float,
) -> float | None:
    """Return a linearly interpolated quantile for probability 0.0 through 1.0."""
    parsed_probability = _validate_probability(probability)
    cleaned = _validated_values(values)
    return _quantile_validated(cleaned, parsed_probability)


def manual_iqr(values: Iterable[int | float | None]) -> float | None:
    """Return Q3 minus Q1 using the project quantile interpolation policy."""
    cleaned = _validated_values(values)
    return _iqr_validated(cleaned)


def manual_skewness(values: Iterable[int | float | None]) -> float | None:
    """Return population-moment skewness, or None when undefined."""
    cleaned = _validated_values(values)
    return _skewness_validated(cleaned)


def summarize(values: Iterable[int | float | None]) -> DescriptiveStatistics:
    """Return descriptive-statistics fields."""
    cleaned, missing_count = _validated_values_and_missing_count(values)
    count = _count_validated(cleaned)
    minimum = _min_validated(cleaned)
    q1 = _quantile_validated(cleaned, 0.25)
    median = _quantile_validated(cleaned, 0.50)
    q3 = _quantile_validated(cleaned, 0.75)
    maximum = _max_validated(cleaned)
    return DescriptiveStatistics(
        count=count,
        mean=_mean_validated(cleaned, count),
        std=_std_validated(cleaned),
        minimum=minimum,
        q1=q1,
        median=median,
        q3=q3,
        maximum=maximum,
        sample_variance=_sample_variance_validated(cleaned),
        value_range=_range_from_bounds(minimum, maximum),
        iqr=_iqr_from_quartiles(q1, q3),
        missing_count=missing_count,
        skewness=_skewness_validated(cleaned),
    )


def _validated_values(values: Iterable[int | float | None]) -> list[float]:
    cleaned, _ = _validated_values_and_missing_count(values)
    return cleaned


def _validated_values_and_missing_count(
    values: Iterable[int | float | None],
) -> tuple[list[float], int]:
    cleaned: list[float] = []
    missing_count = 0
    for position, value in enumerate(values):
        if value is None:
            missing_count += 1
            continue
        cleaned.append(_validate_number(value, position))
    return cleaned, missing_count


def _validate_number(value: int | float, position: int) -> float:
    if isinstance(value, bool):
        raise StatisticsError(
            f"invalid numerical value at position {position}: "
            f"{value!r} has unsupported type bool"
        )
    if not isinstance(value, (int, float)):
        raise StatisticsError(
            f"invalid numerical value at position {position}: "
            f"{value!r} has unsupported type {type(value).__name__}"
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StatisticsError(
            f"invalid numerical value at position {position}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise StatisticsError(
            f"invalid numerical value at position {position}: "
            f"{value!r} is not finite"
        )
    return parsed


def _validate_probability(probability: int | float) -> float:
    if isinstance(probability, bool):
        raise StatisticsError(
            f"invalid quantile probability {probability!r}: "
            "unsupported type bool"
        )
    if not isinstance(probability, (int, float)):
        raise StatisticsError(
            f"invalid quantile probability {probability!r}: "
            f"unsupported type {type(probability).__name__}"
        )
    try:
        parsed = float(probability)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StatisticsError(
            f"invalid quantile probability {probability!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise StatisticsError(
            f"invalid quantile probability {probability!r}: not finite"
        )
    if parsed < 0.0 or parsed > 1.0:
        raise StatisticsError(
            f"invalid quantile probability {probability!r}: expected 0.0 to 1.0"
        )
    return parsed


def _count_validated(values: Iterable[float]) -> int:
    count = 0
    for _ in values:
        count += 1
    return count


def _mean_validated(values: Iterable[float], count: int | None = None) -> float | None:
    total = 0.0
    local_count = 0
    for value in values:
        total += value
        local_count += 1
    effective_count = local_count if count is None else count
    if effective_count == 0:
        return None
    return total / effective_count


def _std_validated(values: Iterable[float]) -> float | None:
    sample_variance = _sample_variance_validated(values)
    if sample_variance is None:
        return None
    return math.sqrt(sample_variance)


def _sample_variance_validated(values: Iterable[float]) -> float | None:
    count = _count_validated(values)
    if count < 2:
        return None
    mean = _mean_validated(values, count)
    squared_total = 0.0
    for value in values:
        difference = value - mean
        squared_total += difference * difference
    return squared_total / (count - 1)


def _min_validated(values: Iterable[float]) -> float | None:
    current = None
    seen_value = False
    for value in values:
        if not seen_value or value < current:
            current = value
            seen_value = True
    return current


def _max_validated(values: Iterable[float]) -> float | None:
    current = None
    seen_value = False
    for value in values:
        if not seen_value or value > current:
            current = value
            seen_value = True
    return current


def _range_validated(values: Iterable[float]) -> float | None:
    return _range_from_bounds(_min_validated(values), _max_validated(values))


def _range_from_bounds(
    minimum: float | None,
    maximum: float | None,
) -> float | None:
    if minimum is None or maximum is None:
        return None
    return maximum - minimum


def _quantile_validated(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(values)
    count = _count_validated(ordered)
    if count == 0:
        return None
    if count == 1:
        return ordered[0]

    position = (count - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    if lower_index == upper_index:
        return lower_value
    fraction = position - lower_index
    return lower_value + fraction * (upper_value - lower_value)


def _iqr_validated(values: Iterable[float]) -> float | None:
    return _iqr_from_quartiles(
        _quantile_validated(values, 0.25),
        _quantile_validated(values, 0.75),
    )


def _iqr_from_quartiles(q1: float | None, q3: float | None) -> float | None:
    if q1 is None or q3 is None:
        return None
    return q3 - q1


def _skewness_validated(values: Iterable[float]) -> float | None:
    count = _count_validated(values)
    if count < 3:
        return None
    mean = _mean_validated(values, count)
    second_moment = 0.0
    third_moment = 0.0
    for value in values:
        difference = value - mean
        squared = difference * difference
        second_moment += squared
        third_moment += squared * difference
    second_moment /= count
    if second_moment == 0.0:
        return None
    third_moment /= count
    return third_moment / (second_moment * math.sqrt(second_moment))
