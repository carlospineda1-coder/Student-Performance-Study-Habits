# Fase 2 — Scripts y Docker

## Objetivo

Convertir el modelo entrenado en la Fase 1 en una aplicación reutilizable: entrenar y generar predicciones desde la línea de comandos, empaquetada en Docker, con el modelo versionado y pruebas básicas. No se entrena un modelo nuevo — `train.py` reproduce exactamente las decisiones fijadas en la Fase 1 (mismas columnas, mismo split 80/20 con `random_state=42`, algoritmo Lasso con `alpha=0.1`).

---

## Estructura

```
fase-2/
├── README.md
├── requirements.txt
├── common.py                   # columnas, pipeline y métricas compartidas por train.py y predict.py
├── train.py                    # entrena y versiona el modelo
├── predict.py                  # genera predicciones con el modelo versionado
├── Dockerfile
├── data/
│   ├── student_performance_dataset.csv   # dataset (copia del usado en fase-1)
│   └── sample.csv                         # 2 filas de ejemplo para predict.py
├── modelo/
│   ├── metadata.json            # historial de versiones entrenadas
│   └── pipeline_modelo_v1.joblib
└── tests/
    ├── conftest.py
    ├── test_common.py
    ├── test_train.py
    └── test_predict.py
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Entrenar (`train.py`)

```bash
python train.py
```

Entrena el pipeline (preprocesamiento + Lasso) sobre `data/student_performance_dataset.csv`, imprime MAE/RMSE/R² sobre el conjunto de prueba y guarda el modelo versionado en `modelo/`.

Opciones principales:

| Flag | Default | Descripción |
|------|---------|-------------|
| `--data` | `data/student_performance_dataset.csv` | CSV de entrenamiento |
| `--model-dir` | `modelo` | Carpeta donde se guarda el modelo |
| `--version` | autoincremental (`v1`, `v2`, ...) | Etiqueta de la versión a generar |
| `--alpha` | `0.1` | Parámetro del Lasso |
| `--test-size` | `0.2` | Proporción del split de prueba |
| `--random-state` | `42` | Semilla del split |

---

## Predecir (`predict.py`)

```bash
python predict.py --input data/sample.csv
```

Carga la versión más reciente registrada en `modelo/metadata.json`, predice `final_exam_score` para cada fila del CSV de entrada y agrega la columna `predicted_final_exam_score`. El CSV de entrada debe tener las 9 columnas predictoras (ver `common.FEATURE_COLS`).

Opciones principales:

| Flag | Descripción |
|------|-------------|
| `--input` | **Obligatorio.** CSV con las columnas predictoras |
| `--output` | Ruta de salida; si se omite, imprime a stdout |
| `--model` | Ruta directa a un `.joblib`, salta la resolución por `metadata.json` |
| `--model-dir` | Carpeta del modelo versionado (default `modelo`) |

---

## Versionado del modelo

Cada corrida de `train.py` guarda un archivo `modelo/pipeline_modelo_v{N}.joblib` nuevo (no sobrescribe el anterior) y registra la versión en `modelo/metadata.json`:

```json
{
  "latest": "v1",
  "versions": {
    "v1": {
      "path": "pipeline_modelo_v1.joblib",
      "created_at": "2026-...",
      "alpha": 0.1,
      "metrics": {"mae": 4.936, "rmse": 6.3659, "r2": 0.6235},
      "n_train": 800
    }
  }
}
```

`predict.py` usa por defecto la versión marcada como `latest`. Para fijar una versión específica: `python predict.py --input ... --model modelo/pipeline_modelo_v1.joblib`.

---

## Docker

Build (contexto = esta carpeta):

```bash
docker build -t fase2-student-performance .
```

Predecir usando el modelo ya empaquetado en la imagen:

```bash
docker run --rm fase2-student-performance predict.py --input data/sample.csv
```

Reentrenar dentro del contenedor (genera una nueva versión en `/app/modelo`, efímera salvo que se monte un volumen):

```bash
# macOS / Linux / PowerShell
docker run --rm -v "${PWD}/modelo:/app/modelo" fase2-student-performance train.py --version v2

# Git Bash en Windows: $(pwd) da una ruta estilo /c/Users/... que Docker Desktop
# no traduce bien para -v. Usar la ruta completa estilo Windows en su lugar:
docker run --rm -v "C:/Users/<usuario>/Student-Performance-Study-Habits/fase-2/modelo:/app/modelo" fase2-student-performance train.py --version v2
```

El `ENTRYPOINT` es `python`, así que cualquier script (`train.py` o `predict.py`) y sus flags se pasan como argumentos de `docker run`.

---

## Pruebas (`pytest`)

```bash
pytest tests
```

Cubren: construcción del pipeline y cálculo de métricas (`test_common.py`), entrenamiento y guardado versionado (`test_train.py`), y resolución del modelo + predicción (`test_predict.py`). Usan un dataset sintético pequeño, no el CSV completo, para que corran en segundos y no dependan de métricas exactas del dataset real.

---

## Troubleshooting

### `pip install` o `docker build` fallan con `SSLCertVerificationError` / `CERTIFICATE_VERIFY_FAILED`

Causa típica en Windows: el antivirus (ej. **Avast**, con su "Web Shield") hace inspección HTTPS e inyecta su propio certificado, que ni `pip` ni el contenedor Docker reconocen como válido — porque el tráfico de red de Docker Desktop en Windows pasa por la pila de red del host, donde el antivirus también lo intercepta. Se puede confirmar revisando las variables de entorno del sistema: si aparece algo como `NODE_EXTRA_CA_CERTS=...\Avast\wscert.pem`, esa es la causa.

**Solución:** desactivar temporalmente el Web Shield del antivirus (Avast: ícono en la bandeja → Protección → Protección principal → Web Shield → desactivar 10 min/1 h), reintentar el comando, y volver a activarlo al terminar. No hay que cambiar nada del código ni del Dockerfile — es un problema del entorno local, no del proyecto.

### El volumen de Docker (`-v`) no persiste cambios en el host (Windows / Git Bash)

En Git Bash, `$(pwd)` devuelve una ruta estilo `/c/Users/...` que Docker Desktop a veces no traduce correctamente al montar un volumen con `-v`, y el contenedor escribe en un directorio que no está realmente enlazado al host (no da error, pero los archivos "desaparecen" al salir del contenedor). Usar en su lugar la ruta completa estilo Windows (`C:/Users/<usuario>/.../fase-2/modelo:/app/modelo`), o ejecutar el comando desde PowerShell con `${PWD}`.
