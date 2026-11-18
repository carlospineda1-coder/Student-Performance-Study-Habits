import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FASE_3 = Path(__file__).resolve().parent.parent
FASE_2 = FASE_3.parent / "fase-2"
sys.path[:0] = [str(FASE_3), str(FASE_2)]

import api  # noqa: E402
import train  # noqa: E402

DATASET = FASE_2 / "data" / "student_performance_dataset.csv"


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Carpeta temporal con la versión v1 entrenada: las pruebas no tocan fase-2/models."""
    output_dir = tmp_path / "models"
    assert train.main(["--data", str(DATASET), "--output-dir", str(output_dir), "--version", "v1"]) == 0
    monkeypatch.setenv("MODELS_DIR", str(output_dir))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("DATA_PATH", str(DATASET))
    return output_dir


@pytest.fixture
def client(models_dir):
    return TestClient(api.app)


@pytest.fixture
def record():
    return {
        "study_time_hours": 5.0,
        "attendance_percent": 90.0,
        "sleep_hours": 7.0,
        "previous_grade": 80.0,
        "gender": "Female",
        "parental_education": "Bachelors",
        "internet_access": "Yes",
        "extracurricular_activities": "Yes",
        "part_time_job": "No",
    }
