"""Muestra el total de solicitudes a /predict y el porcentaje de fallidas (HTTP 4xx o 5xx).

Uso, desde fase-4:
    python monitoreo.py [ruta del registro]
"""

import argparse
import csv
import os
import sys

from registro import log_path


def count_requests(path):
    """Devuelve el total de solicitudes y cuántas fallaron. Un registro que no existe está vacío."""
    if not os.path.exists(path):
        return 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        codes = [int(row["estado_http"]) for row in csv.DictReader(f)]
    return len(codes), sum(400 <= code <= 599 for code in codes)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Métricas de operación del registro de /predict.")
    parser.add_argument("registro", nargs="?", default=log_path(), help="CSV del registro (default: %(default)s)")
    path = parser.parse_args(argv).registro

    try:
        total, failed = count_requests(path)
    except (OSError, KeyError, ValueError) as error:
        print(f"Error: no se pudo leer {path} como registro: {error!r}", file=sys.stderr)
        return 1

    print(f"Registro: {path}")
    print(f"Total de solicitudes: {total}")
    if total == 0:
        print("Porcentaje de solicitudes fallidas: sin solicitudes")
    else:
        percent = f"{failed / total * 100:.1f}".replace(".", ",")
        print(f"Porcentaje de solicitudes fallidas: {percent} % ({failed} de {total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
