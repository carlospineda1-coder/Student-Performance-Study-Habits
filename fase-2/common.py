"""Lógica compartida por train.py y predict.py: lectura y validación de datos,
construcción del pipeline, predicción, métricas y carga de versiones."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "final_exam_score"
NUM_FEATURES = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]
CAT_FEATURES = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
]
FEATURE_COLS = NUM_FEATURES + CAT_FEATURES
SCORE_MIN, SCORE_MAX = 0.0, 100.0
MODEL_FILENAME = "modelo.joblib"


class InvalidInputError(ValueError):
    """Error previsible de los datos o archivos de entrada. Los scripts lo muestran sin traceback."""


def load_dataset(csv_path):
    """Lee el CSV. Solo la celda vacía es nula: la cadena "None" de parental_education
    es una categoría (padres sin educación formal), no un faltante."""
    try:
        return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    except FileNotFoundError:
        raise InvalidInputError(f"No existe el archivo {csv_path}") from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise InvalidInputError(f"No se pudo leer {csv_path} como CSV: {error}") from None


def _csv_lines(mask):
    # +2: el índice empieza en 0 y la línea 1 del archivo es el encabezado
    return ", ".join(str(i + 2) for i in mask[mask].index[:10])


def prepare_features(df, categories=None):
    """Valida las columnas predictoras y las devuelve en el orden del entrenamiento.

    Las celdas vacías se dejan como NaN para que las impute el pipeline. Si se pasan
    `categories`, también rechaza valores no vistos en el entrenamiento.
    """
    if df.empty:
        raise InvalidInputError("El archivo no tiene filas")
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise InvalidInputError(f"Faltan columnas requeridas: {', '.join(missing)}")

    X = df[FEATURE_COLS].copy()
    for col in NUM_FEATURES:
        values = pd.to_numeric(X[col], errors="coerce")
        invalid = X[col].notna() & values.isna()
        if invalid.any():
            raise InvalidInputError(
                f"La columna {col} debe ser numérica; valores no válidos en las líneas {_csv_lines(invalid)}"
            )
        X[col] = values.astype(float)

    if categories:
        for col in CAT_FEATURES:
            unknown = X[col].notna() & ~X[col].isin(categories[col])
            if unknown.any():
                raise InvalidInputError(
                    f"La columna {col} tiene valores no vistos en el entrenamiento "
                    f"({', '.join(sorted(set(X.loc[unknown, col])))}) en las líneas {_csv_lines(unknown)}. "
                    f"Valores válidos: {', '.join(categories[col])}"
                )
    return X


def build_pipeline(alpha=0.1):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", Lasso(alpha=alpha))])


def learned_categories(pipeline):
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    return {col: [str(value) for value in values] for col, values in zip(CAT_FEATURES, encoder.categories_)}


def predict_scores(pipeline, X):
    """Predice y recorta a la escala del puntaje: si el valor real está en 0–100,
    recortar nunca aumenta el error."""
    return np.clip(pipeline.predict(X), SCORE_MIN, SCORE_MAX)


def compute_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_model(path):
    """Carga una versión a partir de su carpeta o de su archivo modelo.joblib."""
    model_path = os.path.join(path, MODEL_FILENAME) if os.path.isdir(path) else path
    if not os.path.isfile(model_path):
        raise InvalidInputError(f"No existe el modelo {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as error:
        raise InvalidInputError(f"No se pudo cargar el modelo {model_path}: {error}") from None
