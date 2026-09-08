"""CLI entry point for the DSLR scatter plot program."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli
from dslr.analysis.scatter import create_scatter


def build_parser():
    """Build the scatter plot command parser."""
    parser = cli.create_parser(
        "dslr-scatter-plot",
        "Find and plot the two most similar numerical course features.",
    )
    cli.add_dataset_argument(parser, "Path to the training CSV dataset.")
    cli.add_save_argument(parser)
    return parser


def main(argv=None):
    """Run the scatter plot command."""
    args = build_parser().parse_args(argv)
    try:
        report = create_scatter(args.dataset, args.save)
    except DSLRError as exc:
        print(f"dslr-scatter-plot: error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
