"""Registro de las solicitudes a /predict: un archivo CSV con una fila por solicitud."""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["fecha_hora", "estado_http", "version", "prediccion"]


def log_path():
    return os.environ.get("LOG_PATH", str(Path(__file__).resolve().parent / "logs" / "predicciones.csv"))


def append_event(status_code, version="", prediction=""):
    """Agrega una fila al final del registro. El encabezado se escribe al crear el archivo."""
    path = log_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(FIELDS)
        writer.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), status_code, version, prediction])
