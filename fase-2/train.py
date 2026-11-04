"""Entrena el modelo y guarda una versión nueva en <output-dir>/<version>/.

Uso, desde fase-2:
    python train.py --data data/student_performance_dataset.csv --output-dir models --version v1 --seed 42
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from common import (
    FEATURE_COLS,
    MODEL_FILENAME,
    NUM_FEATURES,
    SCORE_MAX,
    SCORE_MIN,
    TARGET_COL,
    InvalidInputError,
    build_pipeline,
    compute_metrics,
    learned_categories,
    load_dataset,
    predict_scores,
    prepare_features,
)

DATASET_SOURCE = "https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset"
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
AUTO_VERSION_PATTERN = re.compile(r"v(\d+)")


def fraction(text):
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("debe estar entre 0 y 1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=f"Entrena el modelo (variable objetivo: {TARGET_COL}) y guarda una versión nueva.",
    )
    parser.add_argument("--data", default="data/student_performance_dataset.csv", help="CSV de entrenamiento (default: %(default)s)")
    parser.add_argument("--output-dir", default="models", help="Carpeta de las versiones (default: %(default)s)")
    parser.add_argument("--version", help="Identificador de la versión, p. ej. v2 (default: siguiente vN libre)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de la partición (default: %(default)s)")
    parser.add_argument("--test-size", type=fraction, default=0.2, help="Proporción de prueba (default: %(default)s)")
    parser.add_argument("--alpha", type=float, default=0.1, help="Regularización de Lasso (default: %(default)s)")
    return parser.parse_args(argv)


def resolve_version(output_dir, version):
    """Devuelve la versión y su carpeta. Una versión que ya existe se rechaza, nunca se sobrescribe."""
    if version is None:
        existing = os.listdir(output_dir) if os.path.isdir(output_dir) else []
        numbers = [int(m.group(1)) for m in map(AUTO_VERSION_PATTERN.fullmatch, existing) if m]
        version = f"v{max(numbers, default=0) + 1}"
    elif not VERSION_PATTERN.fullmatch(version):
        raise InvalidInputError(f"Versión no válida: {version}. Usa letras, números, punto, guion o guion bajo")

    version_dir = os.path.join(output_dir, version)
    if os.path.exists(version_dir):
        raise InvalidInputError(f"La versión {version} ya existe en {version_dir}; usa otro --version")
    return version, version_dir


def validate_target(df):
    if TARGET_COL not in df.columns:
        raise InvalidInputError(f"Falta la variable objetivo {TARGET_COL}")
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any():
        raise InvalidInputError(f"La variable objetivo {TARGET_COL} tiene valores vacíos o no numéricos")
    return y


def train_and_evaluate(df, seed, test_size, alpha):
    """Separa antes de ajustar: el preprocesamiento aprende solo del entrenamiento."""
    X = prepare_features(df)
    y = validate_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    pipeline = build_pipeline(alpha).fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    metrics = {
        "model": compute_metrics(y_test, predict_scores(pipeline, X_test)),
        "model_unclipped": compute_metrics(y_test, pipeline.predict(X_test)),
        "baseline": compute_metrics(y_test, baseline.predict(X_test)),
    }
    return pipeline, metrics, len(X_train), len(X_test)


def code_version():
    # En Docker no hay .git: el commit llega como argumento de construcción
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_metadata(version, pipeline, args, n_rows, n_train, n_test):
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem_type": "regression",
        "target": TARGET_COL,
        "algorithm": "Lasso",
        "model_params": pipeline.named_steps["model"].get_params(),
        "seed": args.seed,
        "features": [{"name": col, "type": "float" if col in NUM_FEATURES else "category"} for col in FEATURE_COLS],
        "categories": learned_categories(pipeline),
        "missing_values": {"numeric": "median", "categorical": "most_frequent"},
        "output_range": [SCORE_MIN, SCORE_MAX],
        "data": {
            "source": DATASET_SOURCE,
            "file": os.path.basename(args.data),
            "sha256": file_sha256(args.data),
            "n_rows": n_rows,
        },
        "split": {
            "strategy": "train_test_split aleatorio; sin grupos, una fila por estudiante",
            "test_size": args.test_size,
            "n_train": n_train,
            "n_test": n_test,
        },
        "metric": "mae",
        "python_version": platform.python_version(),
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "requirements": "requirements.txt",
        "code_version": code_version(),
    }


def save_version(version_dir, pipeline, metrics, metadata):
    os.makedirs(version_dir)
    paths = {
        "Modelo": os.path.join(version_dir, MODEL_FILENAME),
        "Métricas": os.path.join(version_dir, "metrics.json"),
        "Metadatos": os.path.join(version_dir, "metadata.json"),
    }
    joblib.dump(pipeline, paths["Modelo"])
    for key, content in (("Métricas", metrics), ("Metadatos", metadata)):
        with open(paths[key], "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    return paths


def main(argv=None):
    args = parse_args(argv)
    try:
        version, version_dir = resolve_version(args.output_dir, args.version)
        df = load_dataset(args.data)
        pipeline, metrics, n_train, n_test = train_and_evaluate(df, args.seed, args.test_size, args.alpha)
        metadata = build_metadata(version, pipeline, args, len(df), n_train, n_test)
        paths = save_version(version_dir, pipeline, metrics, metadata)
    except (InvalidInputError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Versión {version}: {n_train} filas de entrenamiento, {n_test} de prueba (semilla {args.seed})")
    print(f"{'':18}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
    for name, label in (("model", "Lasso"), ("model_unclipped", "Lasso sin recorte"), ("baseline", "Modelo base")):
        m = metrics[name]
        print(f"{label:18}{m['mae']:8.4f}{m['rmse']:8.4f}{m['r2']:8.4f}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
