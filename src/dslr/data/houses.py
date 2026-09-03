"""Shared Hogwarts house-label handling."""

from __future__ import annotations

from dslr.data.csv_loader import Dataset
from dslr.errors import VisualizationError

HOUSE_COLUMN = "Hogwarts House"
CANONICAL_HOUSE_ORDER = (
    "Gryffindor",
    "Hufflepuff",
    "Ravenclaw",
    "Slytherin",
)


def extract_house_labels(
    dataset: Dataset,
    minimum_houses: int = 2,
) -> tuple[str, ...]:
    """Return deterministic house labels after validating the target column."""
    dataset.require_column(HOUSE_COLUMN)
    first_seen: list[str] = []
    for row_offset, row in enumerate(dataset.rows, start=2):
        label = row[HOUSE_COLUMN]
        if label is None or label.strip() == "":
            raise VisualizationError(
                f"{dataset.source_path}: missing house label at row {row_offset}"
            )
        if label not in first_seen:
            first_seen.append(label)

    ordered: list[str] = []
    for label in CANONICAL_HOUSE_ORDER:
        if label in first_seen:
            ordered.append(label)
    for label in first_seen:
        if label not in ordered:
            ordered.append(label)

    if len(ordered) < minimum_houses:
        raise VisualizationError(
            f"{dataset.source_path}: expected at least {minimum_houses} "
            f"distinct house labels; found {len(ordered)}"
        )
    return tuple(ordered)
