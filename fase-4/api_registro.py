"""La API de fase-3 con el registro de solicitudes activado.

Uso, desde fase-4:
    PYTHONPATH=../fase-2:../fase-3 uvicorn api_registro:app --port 8000
"""

import json

from fastapi import Response

from api import app
from registro import append_event


@app.middleware("http")
async def log_predict_requests(request, call_next):
    """Registra cada solicitud a /predict cuando ya se conoce su resultado. Al envolver la
    respuesta, también quedan registradas las que la validación rechaza antes de llegar a la ruta."""
    if request.url.path != "/predict":
        return await call_next(request)
    try:
        response = await call_next(request)
    except Exception:
        # Un error no previsto llega al cliente como HTTP 500 sin pasar por aquí: se registra antes
        append_event(500)
        raise
    if response.status_code != 200:
        append_event(response.status_code)
        return response

    # La versión y la predicción se toman de la respuesta: se registra lo que recibió el cliente
    body = b"".join([chunk async for chunk in response.body_iterator])
    result = json.loads(body)
    append_event(response.status_code, result["version"], result["prediccion"])
    return Response(content=body, status_code=response.status_code, headers=dict(response.headers))
