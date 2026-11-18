import pytest
from fastapi.testclient import TestClient

import api
from common import load_dataset


def test_health_responde_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_entrada_valida_devuelve_prediccion_y_version(client, record):
    response = client.post("/predict", json=record)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediccion", "version"}
    assert body["version"] == "v1"
    assert 0 <= body["prediccion"] <= 100


def test_campo_obligatorio_ausente_se_rechaza(client, record):
    del record["sleep_hours"]

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert response.json()["detail"] == "Falta el campo obligatorio sleep_hours"


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("sleep_hours", "7", "El campo sleep_hours debe ser un número o null"),
        ("sleep_hours", True, "El campo sleep_hours debe ser un número o null"),
        ("gender", 1, "El campo gender debe ser un texto o null"),
    ],
)
def test_tipo_incompatible_se_rechaza(client, record, field, value, message):
    record[field] = value

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert response.json()["detail"] == message


def test_categoria_no_vista_se_rechaza(client, record):
    record["gender"] = "Otro"

    response = client.post("/predict", json=record)

    assert response.status_code == 422
    assert "gender" in response.json()["detail"]
    assert "Female, Male" in response.json()["detail"]


def test_la_prediccion_se_recorta_a_la_escala_0_100(client, record):
    # Sin recorte, este estudiante recibe más de 100
    record.update(study_time_hours=12.0, attendance_percent=100.0, previous_grade=100.0)

    response = client.post("/predict", json=record)

    assert response.json()["prediccion"] == 100.0


def test_null_se_imputa_con_la_regla_del_entrenamiento(client, record, models_dir):
    import joblib

    pipeline = joblib.load(models_dir / "v1" / "modelo.joblib")
    median = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"].statistics_[2]

    with_null = client.post("/predict", json={**record, "sleep_hours": None})
    with_median = client.post("/predict", json={**record, "sleep_hours": float(median)})

    assert with_null.status_code == 200
    assert with_null.json()["prediccion"] == pytest.approx(with_median.json()["prediccion"])


def test_la_variable_objetivo_no_es_necesaria_y_se_ignora(client, record):
    without_target = client.post("/predict", json=record)
    with_target = client.post("/predict", json={**record, "final_exam_score": 12.0})

    assert with_target.status_code == 200
    assert with_target.json() == without_target.json()


def test_version_no_disponible_responde_503(client, record, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v9")

    response = client.post("/predict", json=record)

    assert response.status_code == 503
    assert "v9" in response.json()["detail"]


def test_train_guarda_la_siguiente_version_y_conserva_la_anterior(client, record, models_dir):
    before = (models_dir / "v1" / "modelo.joblib").read_bytes()

    response = client.post("/train")

    assert response.status_code == 201
    assert response.json()["version"] == "v2"
    assert {p.name for p in (models_dir / "v2").iterdir()} == {"modelo.joblib", "metrics.json", "metadata.json"}
    assert (models_dir / "v1" / "modelo.joblib").read_bytes() == before
    # /predict sigue usando la versión configurada
    assert client.post("/predict", json=record).json()["version"] == "v1"



def test_entrada_conocida_da_las_mismas_predicciones_que_en_la_fase_2(monkeypatch):
    """Envía las filas de fase-2/data/ejemplo_prediccion.csv a la versión v1 entregada. Los valores
    esperados son los que dio predict.py en la evidencia de la fase 2."""
    monkeypatch.setenv("MODELS_DIR", str(api.FASE_2 / "models"))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    rows = load_dataset(api.FASE_2 / "data" / "ejemplo_prediccion.csv")
    # La tercera fila tiene sleep_hours vacío: en JSON se envía como null
    records = rows.astype(object).where(rows.notna(), None).to_dict(orient="records")

    client = TestClient(api.app)
    predictions = [client.post("/predict", json=record).json()["prediccion"] for record in records]

    assert predictions == pytest.approx([96.71768844752711, 54.092953845099565, 80.059732182935])
