"""Genera predicciones de final_exam_score con el modelo versionado.

Uso:
    python predict.py --input data/sample.csv
    python predict.py --input data/sample.csv --output predicciones.csv
    python predict.py --input data/sample.csv --model modelo/pipeline_modelo_v1.joblib
"""

import argparse
import json
import os

import joblib

from common import FEATURE_COLS, load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_model_path(model_dir, explicit_model_path=None):
    if explicit_model_path:
        return explicit_model_path

    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No se encontró {metadata_path}. Entrena un modelo con train.py o pasa --model."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    latest = metadata.get("latest")
    if not latest or latest not in metadata.get("versions", {}):
        raise ValueError(f"metadata.json en {model_dir} no tiene una versión 'latest' válida.")

    return os.path.join(model_dir, metadata["versions"][latest]["path"])


def predict_dataframe(pipeline, df):
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el input: {missing}")
    return pipeline.predict(df[FEATURE_COLS])


def parse_args():
    parser = argparse.ArgumentParser(description="Predice final_exam_score a partir de un CSV.")
    parser.add_argument("--input", required=True, help="CSV con las columnas predictoras")
    parser.add_argument("--output", default=None, help="Ruta de salida (default: imprime a stdout)")
    parser.add_argument("--model", default=None, help="Ruta directa a un .joblib (omite metadata.json)")
    parser.add_argument(
        "--model-dir",
        default=os.path.join(SCRIPT_DIR, "modelo"),
        help="Carpeta del modelo versionado (default: modelo)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = resolve_model_path(args.model_dir, args.model)
    pipeline = joblib.load(model_path)

    df = load_dataset(args.input)
    predictions = predict_dataframe(pipeline, df)

    df_result = df.copy()
    df_result["predicted_final_exam_score"] = predictions

    if args.output:
        df_result.to_csv(args.output, index=False)
        print(f"Predicciones guardadas en: {args.output}")
    else:
        print(df_result.to_string(index=False))


if __name__ == "__main__":
    main()
