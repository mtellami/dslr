"""CLI entry point for the DSLR histogram program."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli
from dslr.analysis.histogram import create_histogram


def build_parser():
    """Build the histogram command parser."""
    parser = cli.create_parser(
        "dslr-histogram",
        "Compare course score distributions across Hogwarts houses.",
    )
    cli.add_dataset_argument(parser, "Path to the training CSV dataset.")
    cli.add_save_argument(parser)
    return parser


def main(argv=None):
    """Run the histogram command."""
    args = build_parser().parse_args(argv)
    try:
        report = create_histogram(args.dataset, args.save)
    except DSLRError as exc:
        print(f"dslr-histogram: error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
