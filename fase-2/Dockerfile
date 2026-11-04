FROM python:3.11.18-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py train.py predict.py ./

# En la imagen no hay .git: el commit se pasa al construirla y queda en los metadatos
ARG GIT_COMMIT=desconocido
ENV GIT_COMMIT=$GIT_COMMIT

CMD ["python", "train.py", "--help"]
