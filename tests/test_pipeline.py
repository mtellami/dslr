"""Exercise the installed commands and the data-to-prediction workflow."""

from __future__ import annotations

import csv
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dslr.data.csv_loader import load_csv
from dslr.data.houses import CANONICAL_HOUSE_ORDER
from dslr.data.preprocessing import fit_preprocessor, transform_dataset
from dslr.evaluation.validation import evaluate_csv_holdout
from dslr.models.model_io import load_model
from dslr.models.prediction import predict_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/dataset_train.csv"
TEST = ROOT / "data/dataset_test.csv"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def command(self, name, *args):
        return subprocess.run(
            [sys.executable, "-m", f"dslr.commands.{name}", *map(str, args)],
            cwd=self.output,
            env={**os.environ, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
            timeout=120,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_commands_show_help_outside_repository(self):
        for name in (
            "describe", "histogram", "scatter_plot", "pair_plot",
            "logreg_train", "logreg_predict",
        ):
            with self.subTest(command=name):
                result = self.command(name, "--help")
                self.assert_success(result)
                self.assertIn("usage:", result.stdout)

    def test_all_package_modules_import(self):
        package = ROOT / "src/dslr"
        for path in sorted(package.rglob("*.py")):
            if path.name == "__init__.py":
                continue
            name = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_describe_reports_course_statistics(self):
        result = self.command("describe", TRAIN)
        self.assert_success(result)
        for text in ("Count", "Mean", "Std", "Astronomy", "Herbology"):
            self.assertIn(text, result.stdout)

    def test_missing_dataset_returns_readable_error(self):
        result = self.command("describe", self.output / "missing.csv")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_train_save_reload_and_predict_for_each_optimizer(self):
        for optimizer in ("batch", "sgd", "mini-batch"):
            with self.subTest(optimizer=optimizer):
                model = self.output / optimizer / "model.json"
                model.parent.mkdir()
                predictions = self.output / optimizer / "predictions.csv"
                trained = self.command(
                    "logreg_train", TRAIN, "--optimizer", optimizer,
                    "--output", model,
                )
                self.assert_success(trained)
                self.assertIn("Training rows: 1600", trained.stdout)
                state = load_model(model)
                self.assertEqual(set(state.class_order), set(CANONICAL_HOUSE_ORDER))
                predicted = self.command(
                    "logreg_predict", TEST, model, "--output", predictions,
                )
                self.assert_success(predicted)
                with predictions.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, ["Index", "Hogwarts House"])
                    rows = list(reader)
                self.assertEqual(len(rows), 400)
                self.assertEqual([row["Index"] for row in rows], list(map(str, range(400))))
                labels = tuple(row["Hogwarts House"] for row in rows)
                self.assertTrue(set(labels).issubset(CANONICAL_HOUSE_ORDER))
                self.assertEqual(labels, tuple(row[1] for row in predict_file(TEST, model).rows))

    def test_malformed_model_returns_readable_error(self):
        model = self.output / "invalid.json"
        model.write_text("{}")
        result = self.command("logreg_predict", TEST, model)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.output / "houses.csv").exists())

    def test_preprocessing_reuses_training_state_for_test_data(self):
        state = fit_preprocessor(load_csv(TRAIN))
        matrix = transform_dataset(load_csv(TEST), state)
        self.assertEqual(matrix.row_count, 400)
        self.assertEqual(matrix.column_count, len(state.features) + 1)
        self.assertTrue(all(row[0] == 1.0 for row in matrix.rows))

    def test_holdout_keeps_partitions_separate_and_predicts_accurately(self):
        report = evaluate_csv_holdout(TRAIN, seed=7)
        self.assertEqual(report.train_row_count + report.validation_row_count, 1600)
        self.assertEqual(report.validation_row_count, 320)
        self.assertGreaterEqual(report.accuracy, 0.95)
        self.assertEqual(sum(metric.total for metric in report.per_class), 320)

    def test_plots_save_png_files(self):
        for name in ("histogram", "scatter_plot", "pair_plot"):
            with self.subTest(command=name):
                path = self.output / "plots" / f"{name}.png"
                result = self.command(name, TRAIN, "--save", path)
                self.assert_success(result)
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
