"""Lazy plotting dependency helpers for DSLR visualizations."""

from __future__ import annotations

import os

from dslr.errors import PlottingDependencyError


def import_pyplot():
    """Import and return matplotlib.pyplot only when plotting is requested."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/dslr_matplotlib_config")
    try:
        import matplotlib.pyplot as pyplot
    except (ImportError, ModuleNotFoundError) as exc:
        raise PlottingDependencyError(
            "Matplotlib is required for plotting; run `make install` first"
        ) from exc
    return pyplot
