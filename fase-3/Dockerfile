# Se construye desde la raíz del repositorio, porque reutiliza el código de fase-2:
#   docker build -f fase-3/Dockerfile -t fase3-student-performance .
FROM python:3.11.18-slim

# La imagen conserva la estructura del repositorio: /app/fase-2 y /app/fase-3
WORKDIR /app/fase-3

COPY fase-2/requirements.txt /app/fase-2/
COPY fase-3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fase-2/common.py fase-2/train.py /app/fase-2/
COPY fase-3/api.py .

ENV PYTHONPATH=/app/fase-2

# En la imagen no hay .git: el commit se pasa al construirla y queda en los metadatos
ARG GIT_COMMIT=desconocido
ENV GIT_COMMIT=$GIT_COMMIT

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
