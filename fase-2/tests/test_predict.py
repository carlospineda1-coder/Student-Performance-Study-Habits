import json

import joblib
import pandas as pd
import pytest

from conftest import make_synthetic_df
from predict import predict_dataframe, resolve_model_path
from train import save_model, train_model


@pytest.fixture
def trained_model_dir(tmp_path):
    X = make_synthetic_df()
    y = pd.Series([50.0 + i for i in range(len(X))])
    pipeline = train_model(X, y, alpha=0.1)
    metrics = {"mae": 1.0, "rmse": 1.5, "r2": 0.9}

    model_dir = str(tmp_path / "modelo")
    save_model(pipeline, model_dir, "v1", alpha=0.1, metrics=metrics, n_train=len(X))
    return model_dir


def test_resolve_model_path_uses_latest(trained_model_dir):
    model_path = resolve_model_path(trained_model_dir)
    assert model_path.endswith("pipeline_modelo_v1.joblib")


def test_resolve_model_path_missing_metadata_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_path(str(tmp_path / "no-existe"))


def test_predict_dataframe_returns_expected_length(trained_model_dir):
    model_path = resolve_model_path(trained_model_dir)
    pipeline = joblib.load(model_path)

    df_input = make_synthetic_df(n=5)
    predictions = predict_dataframe(pipeline, df_input)

    assert len(predictions) == len(df_input)


def test_predict_dataframe_missing_columns_raises(trained_model_dir):
    model_path = resolve_model_path(trained_model_dir)
    pipeline = joblib.load(model_path)

    df_input = make_synthetic_df(n=3).drop(columns=["study_time_hours"])

    with pytest.raises(ValueError):
        predict_dataframe(pipeline, df_input)
