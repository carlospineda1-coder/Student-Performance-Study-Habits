"""Lógica de datos y modelo compartida entre train.py y predict.py.

Reproduce exactamente las decisiones tomadas en fase-1/fase1_modelo_predictivo.ipynb:
columnas usadas, algoritmo (Lasso) y la lectura del CSV que conserva "None" como
categoría de parental_education en lugar de convertirla en nulo.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "final_exam_score"
EXCLUDED_COLS = ["student_id", "final_exam_score", "final_grade"]
NUM_FEATURES = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]
CAT_FEATURES = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
]
FEATURE_COLS = NUM_FEATURES + CAT_FEATURES


def load_dataset(csv_path):
    """Lee el CSV conservando la cadena 'None' como categoría, no como NaN."""
    import pandas as pd

    return pd.read_csv(csv_path, keep_default_na=False)


def split_features_target(df):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_pipeline(alpha=0.1):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", Lasso(alpha=alpha))])


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}
