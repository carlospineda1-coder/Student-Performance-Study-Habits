import monitoreo


def test_dos_validas_y_una_invalida_dan_tres_solicitudes_y_un_tercio_de_fallos(client, record, log_file, capsys):
    client.post("/predict", json=record)
    client.post("/predict", json=record)
    client.post("/predict", json={})

    assert monitoreo.main([str(log_file)]) == 0

    output = capsys.readouterr().out
    assert "Total de solicitudes: 3" in output
    assert "Porcentaje de solicitudes fallidas: 33,3 % (1 de 3)" in output


def test_los_5xx_cuentan_como_fallidas(tmp_path):
    path = tmp_path / "registro.csv"
    path.write_text(
        "fecha_hora,estado_http,version,prediccion\n"
        "2026-11-06T10:00:00+00:00,200,v1,18.7\n"
        "2026-11-06T10:01:00+00:00,503,,\n"
        "2026-11-06T10:02:00+00:00,422,,\n"
        "2026-11-06T10:03:00+00:00,399,,\n",
        encoding="utf-8",
    )

    assert monitoreo.count_requests(path) == (4, 2)


def test_registro_inexistente_o_vacio_informa_sin_solicitudes(tmp_path, capsys):
    empty = tmp_path / "vacio.csv"
    empty.write_text("fecha_hora,estado_http,version,prediccion\n", encoding="utf-8")

    for path in (tmp_path / "no_existe.csv", empty):
        assert monitoreo.main([str(path)]) == 0
        output = capsys.readouterr().out
        assert "Total de solicitudes: 0" in output
        assert "Porcentaje de solicitudes fallidas: sin solicitudes" in output


def test_un_registro_con_una_fila_cortada_da_un_error_legible(tmp_path, capsys):
    path = tmp_path / "cortado.csv"
    path.write_text("fecha_hora,estado_http,version,prediccion\n2026-11-06T10:00:00+00:00\n", encoding="utf-8")

    assert monitoreo.main([str(path)]) == 1
    assert "no se pudo leer" in capsys.readouterr().err
