"""Persistent JSON model serialization for DSLR one-vs-rest models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dslr.errors import DatasetNotFoundError, ModelFormatError, ModelIOError
from dslr.data.houses import HOUSE_COLUMN
from dslr.models.one_vs_rest import (
    CANONICAL_CLASS_ORDER,
    OneVsRestModelState,
    one_vs_rest_state_from_dict,
    one_vs_rest_state_to_dict,
    validate_one_vs_rest_state,
)

MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "dslr_logistic_regression"
MULTICLASS_STRATEGY = "one_vs_rest"
INDEX_COLUMN = "Index"


def model_state_to_dict(state: OneVsRestModelState) -> dict[str, Any]:
    """Return a deterministic JSON-compatible persistent model payload."""
    validate_one_vs_rest_state(state)
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_type": MODEL_TYPE,
        "multiclass_strategy": MULTICLASS_STRATEGY,
        "target_column": HOUSE_COLUMN,
        "index_column": INDEX_COLUMN,
        "class_order": list(state.class_order),
        "features": list(state.preprocessing_state.features),
        "one_vs_rest_state": one_vs_rest_state_to_dict(state),
    }


def model_state_from_dict(payload: dict[str, Any]) -> OneVsRestModelState:
    """Build and validate a one-vs-rest model state from a persistent payload."""
    if not isinstance(payload, dict):
        raise ModelFormatError(
            f"model payload must be a dict, got {type(payload).__name__}"
        )
    required = {
        "schema_version",
        "model_type",
        "multiclass_strategy",
        "target_column",
        "index_column",
        "class_order",
        "features",
        "one_vs_rest_state",
    }
    actual = set(payload)
    missing = required - actual
    extra = actual - required
    if missing:
        raise ModelFormatError(
            f"model payload missing keys: {', '.join(sorted(missing))}"
        )
    if extra:
        raise ModelFormatError(
            f"model payload has unexpected keys: {', '.join(sorted(extra))}"
        )
    if (
        isinstance(payload["schema_version"], bool)
        or not isinstance(payload["schema_version"], int)
        or payload["schema_version"] != MODEL_SCHEMA_VERSION
    ):
        raise ModelFormatError(
            f"unsupported model schema version {payload['schema_version']!r}"
        )
    if payload["model_type"] != MODEL_TYPE:
        raise ModelFormatError(f"unsupported model type {payload['model_type']!r}")
    if payload["multiclass_strategy"] != MULTICLASS_STRATEGY:
        raise ModelFormatError(
            f"unsupported multiclass strategy {payload['multiclass_strategy']!r}"
        )
    if payload["target_column"] != HOUSE_COLUMN:
        raise ModelFormatError(
            f"target column must be {HOUSE_COLUMN!r}, got {payload['target_column']!r}"
        )
    if payload["index_column"] != INDEX_COLUMN:
        raise ModelFormatError(
            f"index column must be {INDEX_COLUMN!r}, got {payload['index_column']!r}"
        )
    class_order = _string_list(payload["class_order"], "class_order")
    if tuple(class_order) != CANONICAL_CLASS_ORDER:
        raise ModelFormatError("class_order must match canonical Hogwarts order")
    features = _string_list(payload["features"], "features")
    state = one_vs_rest_state_from_dict(payload["one_vs_rest_state"])
    if tuple(class_order) != state.class_order:
        raise ModelFormatError("top-level class_order does not match model state")
    if tuple(features) != state.preprocessing_state.features:
        raise ModelFormatError("top-level features do not match preprocessing state")
    validate_one_vs_rest_state(state)
    return state


def save_model(state: OneVsRestModelState, path) -> Path:
    """Atomically write a one-vs-rest model state as JSON."""
    output_path = Path(path)
    payload = model_state_to_dict(state)
    parent = output_path.parent
    if parent and not parent.exists():
        raise ModelIOError(f"model output directory does not exist: {parent}")
    if output_path.exists() and not output_path.is_file():
        raise ModelIOError(f"model output path is not a regular file: {output_path}")
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        temp_path.replace(output_path)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise ModelIOError(f"could not write model file {output_path}: {exc}") from exc
    return output_path


def load_model(path) -> OneVsRestModelState:
    """Read and validate a persistent JSON one-vs-rest model file."""
    model_path = Path(path)
    if not model_path.exists():
        raise DatasetNotFoundError(f"model file not found: {model_path}")
    if not model_path.is_file():
        raise ModelIOError(f"model path is not a regular file: {model_path}")
    try:
        with model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ModelFormatError(f"{model_path}: malformed JSON: {exc.msg}") from exc
    except OSError as exc:
        raise ModelIOError(f"could not read model file {model_path}: {exc}") from exc
    return model_state_from_dict(payload)


def _string_list(payload, label: str) -> list[str]:
    if not isinstance(payload, list):
        raise ModelFormatError(f"{label} must be a list")
    values: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in payload:
        if not isinstance(value, str) or value == "":
            raise ModelFormatError(f"{label} entries must be non-empty strings")
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
        values.append(value)
    if not values:
        raise ModelFormatError(f"{label} must not be empty")
    if duplicates:
        raise ModelFormatError(f"duplicate {label}: {', '.join(duplicates)}")
    return values
