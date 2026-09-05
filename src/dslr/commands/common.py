"""Shared argument definitions for DSLR commands."""

from __future__ import annotations

import argparse
from pathlib import Path

def create_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Create a consistently formatted argument parser."""
    return argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def add_dataset_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the shared positional dataset argument."""
    parser.add_argument("dataset", metavar="DATASET", type=Path, help=help_text)


def add_weights_argument(parser: argparse.ArgumentParser) -> None:
    """Add the positional trained-weights file argument."""
    parser.add_argument(
        "weights",
        metavar="WEIGHTS",
        type=Path,
        help="Path to the trained weights file.",
    )


def add_save_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional plot output path."""
    parser.add_argument(
        "--save",
        metavar="PATH",
        type=Path,
        help="Save the plot to this path instead of opening a window.",
    )


def add_output_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    help_text: str,
) -> None:
    """Add an optional output path with a documented default."""
    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default=Path(default),
        help=help_text,
    )

