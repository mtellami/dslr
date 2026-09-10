"""Render the DSLR pair plot and final model feature decision."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli
from dslr.analysis.pair_plot import create_pair_plot


def build_parser():
    """Build the pair plot command parser."""
    parser = cli.create_parser(
        "dslr-pair-plot",
        "Build the pair-plot matrix and final model-feature decision.",
    )
    cli.add_dataset_argument(parser, "Path to the training CSV dataset.")
    cli.add_save_argument(parser)
    return parser


def main(argv=None):
    """Run pair-plot analysis and return a process exit status."""
    args = build_parser().parse_args(argv)
    try:
        report = create_pair_plot(args.dataset, save_path=args.save)
    except DSLRError as exc:
        print(f"dslr-pair-plot: error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
