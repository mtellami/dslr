"""Predict Hogwarts houses with a saved DSLR logistic-regression model."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli


def build_parser():
    """Build the prediction command parser."""
    parser = cli.create_parser(
        "dslr-logreg-predict",
        "Predict Hogwarts houses from a saved DSLR logistic-regression model.",
    )
    cli.add_dataset_argument(parser, "Path to dataset_test.csv.")
    cli.add_weights_argument(parser)
    cli.add_output_argument(
        parser,
        default="houses.csv",
        help_text="Prediction output path.",
    )
    return parser


def main(argv=None):
    """Run prediction and write the houses.csv output."""
    args = build_parser().parse_args(argv)
    from dslr.models.prediction import format_prediction_report, predict_and_write

    try:
        result, saved_path = predict_and_write(args.dataset, args.weights, args.output)
    except DSLRError as exc:
        print(f"dslr-logreg-predict: error: {exc}", file=sys.stderr)
        return 1
    print(format_prediction_report(result, saved_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
