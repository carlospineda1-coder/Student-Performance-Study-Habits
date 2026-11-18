"""Cliente de prueba: consulta /health y pide una predicción para el primer estudiante del dataset.

Uso, con la API en ejecución:
    python cliente.py [--url http://localhost:8000] [--omitir CAMPO]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# Primera fila de student_performance_dataset.csv, sin la variable objetivo (su puntaje real es 100.0)
RECORD = {
    "study_time_hours": 4.0,
    "attendance_percent": 98.0,
    "sleep_hours": 6.5,
    "previous_grade": 76.9,
    "gender": "Male",
    "parental_education": "Bachelors",
    "internet_access": "Yes",
    "extracurricular_activities": "Yes",
    "part_time_job": "No",
}


def call(url, payload=None):
    """GET si no hay payload; POST con el JSON si lo hay. Devuelve el código HTTP y el cuerpo."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        # Un error interno puede llegar como texto plano y no como JSON
        text = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(text)
        except ValueError:
            return error.code, text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Envía una predicción de ejemplo a la API.")
    parser.add_argument("--url", default="http://localhost:8000", help="Dirección de la API (default: %(default)s)")
    parser.add_argument("--omitir", metavar="CAMPO", choices=RECORD, help="Envía el registro sin ese campo, para ver el rechazo")
    args = parser.parse_args(argv)
    base_url = args.url.rstrip("/")
    record = {key: value for key, value in RECORD.items() if key != args.omitir}

    try:
        status, body = call(f"{base_url}/health")
        print("GET /health ->", status, json.dumps(body))
        print("POST /predict")
        print(json.dumps(record, indent=2))
        status, body = call(f"{base_url}/predict", record)
    except OSError as error:
        # Incluye la conexión rechazada y la que se corta mientras la API todavía está arrancando
        print(f"Error: no se pudo conectar con {base_url}: {getattr(error, 'reason', error)}", file=sys.stderr)
        return 1

    print("->", status, json.dumps(body, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
