"""API REST del modelo: consulta una versión guardada y entrena versiones nuevas.

Reutiliza common.py y train.py de fase-2, que deben estar en PYTHONPATH.

Uso, desde fase-3:
    PYTHONPATH=../fase-2 uvicorn api:app --port 8000
"""

import os
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field, create_model

import train
from common import (
    CAT_FEATURES,
    NUM_FEATURES,
    InvalidInputError,
    learned_categories,
    load_dataset,
    load_model,
    predict_scores,
    prepare_features,
)

FASE_2 = Path(__file__).resolve().parent.parent / "fase-2"

# strict: un número se envía como número y una categoría como texto, sin conversiones
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Category = Annotated[str, Field(strict=True)]

# Las nueve variables son obligatorias; null representa un dato faltante y se imputa
StudentRecord = create_model(
    "StudentRecord",
    **{col: (Number | None, ...) for col in NUM_FEATURES},
    **{col: (Category | None, ...) for col in CAT_FEATURES},
)

VALIDATION_MESSAGES = {
    "missing": "Falta el campo obligatorio {field}",
    "float_type": "El campo {field} debe ser un número o null",
    "finite_number": "El campo {field} debe ser un número finito o null",
    "string_type": "El campo {field} debe ser un texto o null",
    "json_invalid": "El cuerpo de la solicitud no es un JSON válido",
}
DEFAULT_VALIDATION_MESSAGE = "El cuerpo debe ser un objeto JSON con las variables predictoras"

app = FastAPI(title="Student Performance & Study Habits", version="3.0.0")


def models_dir():
    return os.environ.get("MODELS_DIR", str(FASE_2 / "models"))


def model_version():
    return os.environ.get("MODEL_VERSION", "v1")


def data_path():
    return os.environ.get("DATA_PATH", str(FASE_2 / "data" / "student_performance_dataset.csv"))


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    messages = [
        VALIDATION_MESSAGES.get(error["type"], DEFAULT_VALIDATION_MESSAGE).format(field=error["loc"][-1])
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "; ".join(dict.fromkeys(messages))})


def check_categories(record, categories):
    for col in CAT_FEATURES:
        value = record[col]
        if value is not None and value not in categories[col]:
            raise HTTPException(
                status_code=422,
                detail=f"El campo {col} tiene un valor no visto en el entrenamiento ({value}). "
                f"Valores válidos: {', '.join(categories[col])}",
            )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(record: StudentRecord):
    version = model_version()
    try:
        pipeline = load_model(os.path.join(models_dir(), version))
    except InvalidInputError as error:
        raise HTTPException(
            status_code=503, detail=f"La versión configurada ({version}) no está disponible: {error}"
        ) from None

    values = record.model_dump()
    check_categories(values, learned_categories(pipeline))
    X = prepare_features(pd.DataFrame([values]))
    return {"prediccion": float(predict_scores(pipeline, X)[0]), "version": version}


@app.post("/train", status_code=201)
def train_version():
    """Entrena con los datos y los valores por defecto de train.py (semilla 42, 80/20, alpha 0.1)
    y guarda la siguiente versión libre. No cambia la versión que usa /predict."""
    args = train.parse_args(["--data", data_path(), "--output-dir", models_dir()])
    try:
        version, version_dir = train.resolve_version(args.output_dir, args.version)
        df = load_dataset(args.data)
        pipeline, metrics, n_train, n_test = train.train_and_evaluate(df, args.seed, args.test_size, args.alpha)
        metadata = train.build_metadata(version, pipeline, args, len(df), n_train, n_test)
        train.save_version(version_dir, pipeline, metrics, metadata)
    except (InvalidInputError, OSError) as error:
        raise HTTPException(status_code=500, detail=f"No se pudo entrenar: {error}") from None
    return {"version": version, "metricas": metrics["model"]}
