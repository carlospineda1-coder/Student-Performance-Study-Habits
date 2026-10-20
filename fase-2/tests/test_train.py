import json

import pandas as pd
from sklearn.pipeline import Pipeline

from conftest import make_synthetic_df
from train import load_metadata, next_version, save_model, train_model


def test_train_model_returns_fitted_pipeline():
    X = make_synthetic_df()
    y = pd.Series([50.0 + i for i in range(len(X))])

    pipeline = train_model(X, y, alpha=0.1)

    assert isinstance(pipeline, Pipeline)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_save_model_writes_joblib_and_metadata(tmp_path):
    X = make_synthetic_df()
    y = pd.Series([50.0 + i for i in range(len(X))])
    pipeline = train_model(X, y, alpha=0.1)
    metrics = {"mae": 1.0, "rmse": 1.5, "r2": 0.9}

    model_dir = str(tmp_path / "modelo")
    model_path = save_model(pipeline, model_dir, "v1", alpha=0.1, metrics=metrics, n_train=len(X))

    assert model_path.endswith("pipeline_modelo_v1.joblib")
    assert (tmp_path / "modelo" / "pipeline_modelo_v1.joblib").exists()

    with open(tmp_path / "modelo" / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["latest"] == "v1"
    assert metadata["versions"]["v1"]["metrics"] == metrics


def test_next_version_increments(tmp_path):
    model_dir = str(tmp_path / "modelo")
    metadata = load_metadata(model_dir)
    assert next_version(metadata) == "v1"

    metadata["versions"]["v1"] = {}
    metadata["versions"]["v3"] = {}
    assert next_version(metadata) == "v4"
