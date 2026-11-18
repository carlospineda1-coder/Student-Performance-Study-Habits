"""La API de fase-3 con el registro de solicitudes activado.

Uso, desde fase-4:
    PYTHONPATH=../fase-2:../fase-3 uvicorn api_registro:app --port 8000
"""

import json
import logging

from fastapi import Response
from starlette.middleware.base import BaseHTTPMiddleware

import api
from registro import append_event

logger = logging.getLogger("uvicorn.error")


def record_event(status_code, version="", prediction=""):
    # Si el registro no se puede escribir, el cliente recibe igual su respuesta y el fallo queda en la consola
    try:
        append_event(status_code, version, prediction)
    except OSError:
        logger.exception("No se pudo escribir el registro de solicitudes")


async def log_predict_requests(request, call_next):
    """Registra cada solicitud a /predict cuando ya se conoce su resultado. Al envolver la
    respuesta, también quedan registradas las que la validación rechaza antes de llegar a la ruta."""
    if request.url.path != "/predict":
        return await call_next(request)
    # Un error no previsto también llega aquí, ya convertido en HTTP 500 por la API
    response = await call_next(request)
    if response.status_code != 200:
        record_event(response.status_code)
        return response

    # La versión y la predicción se toman de la respuesta: se registra lo que recibió el cliente
    body = b"".join([chunk async for chunk in response.body_iterator])
    result = json.loads(body)
    record_event(response.status_code, result["version"], result["prediccion"])
    return Response(content=body, status_code=response.status_code, headers=dict(response.headers))


# Envuelve la aplicación de fase-3 sin modificarla: quien importe api.app sigue sin registro
app = BaseHTTPMiddleware(api.app, dispatch=log_predict_requests)
