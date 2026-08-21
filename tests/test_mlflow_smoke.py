import os
import tempfile
import unittest
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

import mlflow_tracking.tracking_utils as tracking_utils

from mlflow_tracking.tracking_utils import (
    configure_tracking,
    mlflow_run,
    log_dataset_context,
    log_split_config,
    log_model_params,
    log_metrics,
    log_model_and_artifacts,
    get_or_create_experiment,
)
from mlflow_tracking.experiments import get_experiment_name


class TestMLflowSmoke(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tracking_dir = Path(self.temp_dir.name)
        self.backend_uri = f"sqlite:///{self.tracking_dir / 'mlflow.db'}"
        self.artifact_root = self.tracking_dir / "mlruns"
        self.artifact_root.mkdir(parents=True, exist_ok=True)

        # Save and patch the default artifact root so this test writes into the
        # temporary directory rather than the project's mlflow/mlruns folder.
        self.original_artifact_root = tracking_utils.DEFAULT_ARTIFACT_ROOT
        tracking_utils.DEFAULT_ARTIFACT_ROOT = str(self.artifact_root)

        # Configure MLflow tracking explicitly for this test.
        configure_tracking(tracking_uri=self.backend_uri)

    def tearDown(self):
        try:
            mlflow.end_run()
        except Exception:
            pass

        # Release SQLite connections before deleting the temporary directory.
        # This is essential on Windows to avoid WinError 32.
        tracking_utils.close_tracking_store()

        # Restore the original default artifact root.
        tracking_utils.DEFAULT_ARTIFACT_ROOT = self.original_artifact_root

        self.temp_dir.cleanup()

    def test_local_run_records_metadata(self):
        # Create a dummy dataset file to compute hash
        dataset_path = self.tracking_dir / "dummy.parquet"
        pd.DataFrame({"a": [1, 2, 3]}).to_parquet(dataset_path)

        experiment_name = get_experiment_name("p1")
        with mlflow_run(experiment_name, run_name="smoke_run") as run:
            log_dataset_context(dataset_path, row_count=3, feature_count=1, target="a")
            log_split_config("train_test_split", None, 42, {"scale": "standard"})
            log_model_params({"n_estimators": 100})
            log_metrics({"accuracy": 0.9, "f1": 0.85})
            # Log a simple sklearn model
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy="most_frequent")
            model.fit([[1], [2], [3]], [0, 1, 0])
            log_model_and_artifacts(model)

        # Check run
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        self.assertIsNotNone(experiment)
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        self.assertEqual(len(runs), 1)
        run_data = runs[0].data

        self.assertEqual(run_data.params["row_count"], "3")
        self.assertEqual(run_data.params["feature_count"], "1")
        self.assertIn("accuracy", run_data.metrics)
        self.assertAlmostEqual(run_data.metrics["accuracy"], 0.9)
        # Verify standard tag problem_name is present
        self.assertEqual(run_data.tags.get("problem_name"), experiment_name)
