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


def test_un_error_interno_se_registra_como_500(record, log_file, monkeypatch):
    import api
    import api_registro
    from fastapi.testclient import TestClient

    def fail(*args):
        raise RuntimeError("fallo no previsto")

    monkeypatch.setattr(api, "predict_scores", fail)

    response = TestClient(api_registro.app, raise_server_exceptions=False).post("/predict", json=record)

    assert response.status_code == 500
    (row,) = read_rows(log_file)
    assert (row["estado_http"], row["version"], row["prediccion"]) == ("500", "", "")


def test_las_demas_rutas_no_se_registran(client, log_file):
    client.get("/health")

    assert not log_file.exists()


def test_el_registro_conserva_el_historial(client, record, log_file):
    log_file.parent.mkdir()
    previous = "fecha_hora,estado_http,version,prediccion\n2026-11-06T10:00:00+00:00,200,v1,18.7\n"
    log_file.write_text(previous, encoding="utf-8")

    client.post("/predict", json=record)

    content = log_file.read_text(encoding="utf-8")
    assert content.startswith(previous)
    assert len(content.splitlines()) == 3


def test_un_registro_que_no_se_puede_escribir_no_cambia_la_respuesta(client, record, tmp_path, monkeypatch):
    blocker = tmp_path / "archivo"
    blocker.write_text("")
    monkeypatch.setenv("LOG_PATH", str(blocker / "predicciones.csv"))

    response = client.post("/predict", json=record)

    assert response.status_code == 200
    assert set(response.json()) == {"prediccion", "version"}


def test_la_api_de_la_fase_3_no_queda_modificada(record, log_file):
    import api
    from fastapi.testclient import TestClient

    TestClient(api.app).post("/predict", json=record)

    assert not log_file.exists()
