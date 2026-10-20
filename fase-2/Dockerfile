FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py train.py predict.py ./
COPY data/ ./data/
COPY modelo/ ./modelo/

ENTRYPOINT ["python"]
CMD ["predict.py", "--help"]
