"""Strict standard-library CSV loading for the DSLR project."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from dslr.errors import (
    DatasetFormatError,
    DatasetNotFoundError,
    MissingColumnError,
)

MISSING_TOKENS = frozenset(("", "NaN", "nan", "NULL", "null", "None"))
DEFAULT_NUMERIC_EXCLUSIONS = frozenset(("Hogwarts House",))


@dataclass
class Dataset:
    """In-memory CSV data with preserved header and source row order."""

    headers: list[str]
    rows: list[dict[str, str | None]]
    source_path: Path

    @property
    def row_count(self) -> int:
        """Return the number of data rows."""
        return len(self.rows)

    def require_column(self, name: str) -> str:
        """Return the column name when present, or raise MissingColumnError."""
        if name not in self.headers:
            raise MissingColumnError(
                f"{self.source_path}: missing required column {name!r}"
            )
        return name

    def get_column(self, name: str) -> list[str | None]:
        """Return a new list of column values in source row order."""
        column = self.require_column(name)
        return [row[column] for row in self.rows]

    def categorical_values(self, name: str) -> list[str]:
        """Return non-missing string values in source row order."""
        return [value for value in self.get_column(name) if value is not None]

    def numeric_values(self, name: str) -> list[float]:
        """Return finite float values for a column, skipping missing cells."""
        column = self.require_column(name)
        values: list[float] = []
        for row_offset, row in enumerate(self.rows, start=2):
            value = row[column]
            if value is None:
                continue
            parsed = _parse_finite_float(value)
            if parsed is None:
                raise DatasetFormatError(
                    f"{self.source_path}: column {column!r}, row {row_offset}: "
                    f"invalid numeric value {value!r}"
                )
            values.append(parsed)
        return values

    def detect_numeric_columns(
        self,
        excluded: Iterable[str] | None = None,
    ) -> list[str]:
        """Detect finite numeric columns while preserving source header order."""
        exclusions = set(DEFAULT_NUMERIC_EXCLUSIONS)
        if excluded is not None:
            exclusions.update(excluded)

        numeric_columns: list[str] = []
        for header in self.headers:
            if header in exclusions:
                continue
            seen_value = False
            is_numeric = True
            for row in self.rows:
                value = row[header]
                if value is None:
                    continue
                seen_value = True
                if _parse_finite_float(value) is None:
                    is_numeric = False
                    break
            if seen_value and is_numeric:
                numeric_columns.append(header)
        return numeric_columns

    def validate_schema(self, required_columns: Iterable[str]) -> None:
        """Raise MissingColumnError if any required columns are absent."""
        missing = [name for name in required_columns if name not in self.headers]
        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise MissingColumnError(
                f"{self.source_path}: missing required columns: {formatted}"
            )


def load_csv(path) -> Dataset:
    """Load a CSV file strictly without changing the source file."""
    source_path = Path(path)
    if not source_path.exists():
        raise DatasetNotFoundError(f"{source_path}: dataset file not found")
    if not source_path.is_file():
        raise DatasetFormatError(f"{source_path}: path is not a regular file")

    try:
        with source_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise DatasetFormatError(
                    f"{source_path}: CSV file is empty"
                ) from exc

            _validate_headers(source_path, headers)
            rows: list[dict[str, str | None]] = []
            for data_index, raw_row in enumerate(reader, start=2):
                if len(raw_row) != len(headers):
                    raise DatasetFormatError(
                        f"{source_path}: row {data_index} has {len(raw_row)} "
                        f"fields; expected {len(headers)}"
                    )
                rows.append(
                    {
                        header: _normalize_cell(value)
                        for header, value in zip(headers, raw_row)
                    }
                )
    except csv.Error as exc:
        raise DatasetFormatError(f"{source_path}: malformed CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            f"{source_path}: unable to decode CSV as UTF-8"
        ) from exc
    except OSError as exc:
        raise DatasetFormatError(f"{source_path}: unable to read CSV: {exc}") from exc

    return Dataset(headers=list(headers), rows=rows, source_path=source_path)


def _validate_headers(source_path: Path, headers: list[str]) -> None:
    if not headers:
        raise DatasetFormatError(f"{source_path}: missing header row")

    seen: set[str] = set()
    for position, header in enumerate(headers, start=1):
        if header.strip() == "":
            raise DatasetFormatError(
                f"{source_path}: empty header name at position {position}"
            )
        if header in seen:
            raise DatasetFormatError(f"{source_path}: duplicate header {header!r}")
        seen.add(header)


def _normalize_cell(value: str) -> str | None:
    if value.strip() in MISSING_TOKENS:
        return None
    return value


def _parse_finite_float(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
