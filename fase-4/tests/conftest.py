import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FASE_4 = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(FASE_4), str(FASE_4.parent / "fase-3"), str(FASE_4.parent / "fase-2")]

import api_registro  # noqa: E402
from cliente import RECORD  # noqa: E402


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    """Registro temporal. Se predice con la versión v1 de fase-2/models, que solo se lee."""
    path = tmp_path / "logs" / "predicciones.csv"
    monkeypatch.setenv("LOG_PATH", str(path))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.delenv("MODELS_DIR", raising=False)
    return path


@pytest.fixture
def client(log_file):
    return TestClient(api_registro.app)


@pytest.fixture
def record():
    return dict(RECORD)
