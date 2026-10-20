"""Entrena el pipeline de Fase 1 (Lasso) desde la línea de comandos y versiona el modelo.

Uso:
    python train.py
    python train.py --data data/student_performance_dataset.csv --alpha 0.1
"""

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split

from common import build_pipeline, compute_metrics, load_dataset, split_features_target

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metadata(model_dir):
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"latest": None, "versions": {}}


def next_version(metadata):
    existing = [int(v[1:]) for v in metadata["versions"] if v.startswith("v") and v[1:].isdigit()]
    return f"v{max(existing, default=0) + 1}"


def train_model(X_train, y_train, alpha):
    pipeline = build_pipeline(alpha=alpha)
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline, model_dir, version, alpha, metrics, n_train):
    os.makedirs(model_dir, exist_ok=True)
    filename = f"pipeline_modelo_{version}.joblib"
    model_path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, model_path)

    metadata = load_metadata(model_dir)
    metadata["latest"] = version
    metadata["versions"][version] = {
        "path": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "alpha": alpha,
        "metrics": metrics,
        "n_train": n_train,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena y versiona el pipeline predictivo.")
    parser.add_argument(
        "--data",
        default=os.path.join(SCRIPT_DIR, "data", "student_performance_dataset.csv"),
        help="Ruta al CSV de entrenamiento (default: data/student_performance_dataset.csv)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.path.join(SCRIPT_DIR, "modelo"),
        help="Carpeta donde se guarda el modelo versionado (default: modelo)",
    )
    parser.add_argument("--version", default=None, help="Etiqueta de versión, ej. v2 (default: autoincremental)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1, help="Parámetro alpha de Lasso (default: 0.1)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_dataset(args.data)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    pipeline = train_model(X_train, y_train, args.alpha)
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    metadata = load_metadata(args.model_dir)
    version = args.version or next_version(metadata)
    model_path = save_model(pipeline, args.model_dir, version, args.alpha, metrics, len(X_train))

    print(f"Versión entrenada: {version}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
