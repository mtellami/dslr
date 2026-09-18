"""Train the DSLR logistic-regression model and write weights JSON."""

from __future__ import annotations

import sys
from dslr.errors import DSLRError
from dslr.commands import common as cli


def build_parser():
    """Build the training command parser."""
    parser = cli.create_parser(
        "dslr-logreg-train",
        "Train the DSLR one-vs-rest logistic-regression model.",
    )
    cli.add_dataset_argument(parser, "Path to dataset_train.csv.")
    cli.add_output_argument(
        parser,
        default="weights.json",
        help_text="Model output path.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("batch", "sgd", "mini-batch"),
        default="batch",
        help="Training optimizer to use. Defaults to batch gradient descent.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epoch count for stochastic or mini-batch optimizers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic shuffle seed for stochastic or mini-batch optimizers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size when --optimizer mini-batch is selected.",
    )
    return parser


def main(argv=None):
    """Train the model and write the requested JSON file."""
    args = build_parser().parse_args(argv)
    from dslr.models.logistic_regression import (
        MiniBatchTrainingConfig,
        StochasticTrainingConfig,
    )
    from dslr.models.training import format_training_report, train_and_save_model

    try:
        sgd_config = None
        minibatch_config = None
        if args.optimizer == "sgd":
            sgd_config = StochasticTrainingConfig(epochs=args.epochs, seed=args.seed)
        elif args.optimizer == "mini-batch":
            minibatch_config = MiniBatchTrainingConfig(
                epochs=args.epochs,
                seed=args.seed,
                batch_size=args.batch_size,
            )
        report, saved_path = train_and_save_model(
            args.dataset,
            args.output,
            optimizer=args.optimizer,
            sgd_config=sgd_config,
            minibatch_config=minibatch_config,
        )
    except DSLRError as exc:
        print(f"dslr-logreg-train: error: {exc}", file=sys.stderr)
        return 1
    print(format_training_report(report, saved_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
