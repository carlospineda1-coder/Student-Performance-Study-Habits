import csv


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_cada_solicitud_deja_una_fila_incluida_la_rechazada(client, record, log_file):
    first = client.post("/predict", json=record)
    client.post("/predict", json={**record, "study_time_hours": 1.0})
    incomplete = {key: value for key, value in record.items() if key != "sleep_hours"}
    rejected = client.post("/predict", json=incomplete)

    rows = read_rows(log_file)
    assert [row["estado_http"] for row in rows] == ["200", "200", "422"]
    assert rows[0]["version"] == "v1"
    assert float(rows[0]["prediccion"]) == first.json()["prediccion"]
    assert rows[2]["version"] == "" and rows[2]["prediccion"] == ""
    assert all(row["fecha_hora"] for row in rows)
    assert rejected.json()["detail"] == "Falta el campo obligatorio sleep_hours"


def test_el_registro_no_cambia_la_respuesta(client, record):
    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert set(response.json()) == {"prediccion", "version"}
    assert response.headers["content-type"] == "application/json"


def test_sin_modelo_se_registra_el_503_sin_version(client, record, log_file, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v9")

    client.post("/predict", json=record)

    (row,) = read_rows(log_file)
    assert (row["estado_http"], row["version"], row["prediccion"]) == ("503", "", "")


def test_las_demas_rutas_no_se_registran(client, log_file):
    client.get("/health")

    assert not log_file.exists()


def test_el_registro_se_conserva_entre_ejecuciones(client, record, log_file):
    from fastapi.testclient import TestClient

    import api_registro

    client.post("/predict", json=record)
    TestClient(api_registro.app).post("/predict", json=record)

    with open(log_file, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert len(lines) == 3
    assert lines[0] == "fecha_hora,estado_http,version,prediccion"
