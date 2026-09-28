# DSLR

**Descriptive statistics and logistic regression, built from scratch.**

DSLR explores tabular data, identifies useful features, and trains a multiclass
classifier without a machine learning framework. The included Hogwarts dataset
uses course scores to predict one of four houses: Gryffindor, Hufflepuff,
Ravenclaw, or Slytherin.

NumPy handles numerical arrays and Matplotlib renders charts. Statistics,
feature selection, preprocessing, optimization, and evaluation are implemented
in Python.

## Features

- Descriptive statistics including missing-value counts, quartiles, variance,
  interquartile range, and skewness.
- House-by-house histograms, correlated-feature scatter plots, and a pair-plot
  matrix for exploring the training data.
- Feature selection based on distribution similarity and correlation.
- Training-only mean imputation and z-score scaling, saved with the model for
  consistent prediction.
- One-vs-rest logistic regression with batch, stochastic, and mini-batch
  gradient descent.
- JSON model persistence, CSV predictions, stratified holdout evaluation, and
  training diagnostics.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mtellami/dslr.git
cd dslr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Alternatively, run `make install` and then activate `.venv`.

## Quick start

Run these commands from the repository root after activating the environment:

```bash
# Inspect numerical features.
dslr-describe data/dataset_train.csv

# Train a model and save its weights and preprocessing parameters.
mkdir -p artifacts
dslr-logreg-train data/dataset_train.csv --output artifacts/model.json

# Predict the 400 unlabeled rows.
dslr-logreg-predict data/dataset_test.csv artifacts/model.json \
  --output artifacts/predictions.csv
```

The prediction CSV has two columns, `Index` and `Hogwarts House`. Without
`--output`, training writes `weights.json` and prediction writes `houses.csv`
in the current directory. Generated artifacts are ignored by Git.
Create the destination directory before saving models or prediction CSVs.

## Explore the data

```bash
dslr-histogram data/dataset_train.csv --save artifacts/histogram.png
dslr-scatter-plot data/dataset_train.csv --save artifacts/scatter.png
dslr-pair-plot data/dataset_train.csv --save artifacts/pair-plot.png
```

Omit `--save` to open an interactive plot. On a server without a display, set
`MPLBACKEND=Agg` when saving figures.

The histogram command compares score distributions across houses. The scatter
command finds the most strongly correlated pair of courses. The pair plot
combines distributions and feature relationships with the model's feature
selection report.

## Training options

Batch gradient descent is the default, with a learning rate of `0.1`, up to
`5000` iterations, and a convergence tolerance of `1e-9`.

```bash
dslr-logreg-train data/dataset_train.csv --optimizer sgd \
  --epochs 100 --seed 7 --output artifacts/sgd.json

dslr-logreg-train data/dataset_train.csv --optimizer mini-batch \
  --epochs 100 --batch-size 32 --seed 7 --output artifacts/mini-batch.json
```

`--epochs` and `--seed` configure stochastic or mini-batch training;
`--batch-size` applies only to mini-batch training. Batch training uses its
iteration and convergence defaults. Each command supports `--help`. Commands
also run as modules, for example
`python -m dslr.commands.logreg_train --help`.

## Evaluation and Python API

Evaluate on a deterministic, stratified 80/20 split of the labeled dataset:

```python
from dslr.evaluation.validation import evaluate_csv_holdout

report = evaluate_csv_holdout("data/dataset_train.csv", seed=7)
print(f"Holdout accuracy: {report.accuracy:.2%}")
for metric in report.per_class:
    print(f"{metric.class_label}: {metric.accuracy:.2%}")
```

Evaluation uses a fixed ten-course feature set and fits preprocessing only on
the training partition. The training command selects features from its input
dataset. Training accuracy printed by the command measures the fitted data;
use holdout evaluation to assess generalization. The unlabeled test CSV cannot
be used to measure accuracy.

Generate loss, convergence, validation-accuracy, and weight-comparison figures:

```python
from dslr.evaluation.diagnostics import create_training_diagnostics

diagnostics, paths = create_training_diagnostics(
    "data/dataset_train.csv", "artifacts/diagnostics"
)
for path in paths:
    print(path)
```

## Dataset

`data/dataset_train.csv` contains 1,600 labeled rows;
`data/dataset_test.csv` contains 400 rows with empty house labels. Both include
an index, student metadata, and 13 numerical course scores. Some scores are
missing. Names, birthdays, handedness, and the index are excluded from model
features. Prediction reuses the feature order, imputation values, and scaling
parameters stored in the model.

The classifier expects this dataset's course schema and four house labels.
The statistics utilities can also inspect other numerical CSV datasets.

## Project structure

```text
.
├── data/                     # Included training and prediction datasets
├── src/dslr/
│   ├── analysis/             # Statistics, plots, and feature selection
│   ├── commands/             # Six command-line entry points
│   ├── data/                 # CSV loading, labels, and preprocessing
│   ├── evaluation/           # Holdout validation and training diagnostics
│   ├── models/               # Optimizers, classifiers, persistence, prediction
│   └── errors.py             # Shared exception types
├── tests/                    # CLI and pipeline regression tests
├── Makefile                  # Installation and validation shortcuts
└── pyproject.toml            # Package metadata, dependencies, and commands
```

## Development

After installing the project, run:

```bash
make check
```

This checks all command help screens and runs the regression suite, covering
training and prediction, model persistence, saved plots, and error handling.
Use `make test` to run the tests alone.
