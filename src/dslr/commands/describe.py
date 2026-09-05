"""CLI entry point for the DSLR describe program."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli
from dslr.analysis.describe import describe_dataset


def build_parser():
    """Build the describe command parser."""
    parser = cli.create_parser(
        "dslr-describe",
        "Display manual descriptive statistics for numerical dataset features.",
    )
    cli.add_dataset_argument(parser, "Path to the CSV dataset to describe.")
    return parser


def main(argv=None):
    """Run the descriptive-statistics command."""
    args = build_parser().parse_args(argv)
    try:
        table = describe_dataset(args.dataset)
    except DSLRError as exc:
        print(f"dslr-describe: error: {exc}", file=sys.stderr)
        return 1
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
