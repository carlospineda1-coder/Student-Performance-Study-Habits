"""Genera predicciones con una versión guardada del modelo, sin reentrenar.

Uso, desde fase-2:
    python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""

import argparse
import os
import sys

from common import InvalidInputError, learned_categories, load_dataset, load_model, predict_scores, prepare_features

PREDICTION_COL = "prediccion"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predice final_exam_score (0–100) para cada fila de un CSV.")
    parser.add_argument("--model", required=True, help="Carpeta de la versión (models/v1) o su modelo.joblib")
    parser.add_argument("--input", required=True, help="CSV con las 9 variables predictoras")
    parser.add_argument("--output", required=True, help="CSV de salida: la entrada más la columna prediccion")
    return parser.parse_args(argv)


def predict_file(model_path, input_path):
    """Devuelve el CSV de entrada, en su mismo orden, con la columna de predicción al final."""
    pipeline = load_model(model_path)
    df = load_dataset(input_path)
    X = prepare_features(df, learned_categories(pipeline))
    result = df.copy()
    result[PREDICTION_COL] = predict_scores(pipeline, X)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = predict_file(args.model, args.input)
        # Se escribe solo después de validar y predecir: si algo falla, no queda un archivo a medias
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_csv(args.output, index=False)
    except (InvalidInputError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"{len(result)} predicciones guardadas en {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
