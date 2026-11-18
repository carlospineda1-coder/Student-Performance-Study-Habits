"""Cliente de prueba: consulta /health y pide una predicción para el primer estudiante del dataset.

Uso, con la API en ejecución:
    python cliente.py [--url http://localhost:8000]
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
        return error.code, json.load(error)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Envía una predicción de ejemplo a la API.")
    parser.add_argument("--url", default="http://localhost:8000", help="Dirección de la API (default: %(default)s)")
    base_url = parser.parse_args(argv).url.rstrip("/")

    try:
        status, body = call(f"{base_url}/health")
        print("GET /health ->", status, json.dumps(body))
        print("POST /predict")
        print(json.dumps(RECORD, indent=2))
        status, body = call(f"{base_url}/predict", RECORD)
    except urllib.error.URLError as error:
        print(f"Error: no se pudo conectar con {base_url}: {error.reason}", file=sys.stderr)
        return 1

    print("->", status, json.dumps(body, ensure_ascii=False))
    return 0 if status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
