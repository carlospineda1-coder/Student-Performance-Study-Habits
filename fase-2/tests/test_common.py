import numpy as np
import pytest

from common import CAT_FEATURES, NUM_FEATURES, build_pipeline, compute_metrics


def test_build_pipeline_steps():
    pipeline = build_pipeline(alpha=0.1)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "model"]
    assert pipeline.named_steps["model"].alpha == 0.1

    transformers = {name: cols for name, _, cols in pipeline.named_steps["preprocessor"].transformers}
    assert transformers["num"] == NUM_FEATURES
    assert transformers["cat"] == CAT_FEATURES


def test_compute_metrics_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_compute_metrics_mae():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(7 / 3)
