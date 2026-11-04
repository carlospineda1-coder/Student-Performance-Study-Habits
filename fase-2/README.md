# Fase 2 — Scripts y Docker

Convierte el modelo de la Fase 1 en una aplicación: `train.py` entrena y guarda versiones identificables del modelo, y `predict.py` carga una versión y predice `final_exam_score` (0–100) sin reentrenar. Ambos corren en local y en Docker. Se conservan el problema, los datos y el modelo de la Fase 1: Lasso con `alpha=0.1` y partición 80/20 con semilla 42.

Todos los comandos se ejecutan desde la carpeta `fase-2`.

## Estructura

```
fase-2/
├── common.py          # lectura y validación de datos, pipeline, predicción, métricas y carga de versiones
├── train.py           # entrena y guarda una versión
├── predict.py         # predice con una versión guardada
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── data/
│   ├── student_performance_dataset.csv   # dataset de Kaggle (copia del de fase-1)
│   └── ejemplo_prediccion.csv            # 3 filas sin la variable objetivo, una con una celda vacía
├── models/<versión>/  # modelo.joblib, metrics.json y metadata.json de cada versión
├── outputs/           # predicciones generadas (no se versiona)
└── tests/
```

## Entorno

Python 3.11 (la imagen Docker usa 3.11.18).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos

[Student Performance & Study Habits — Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset). El archivo está versionado en `data/student_performance_dataset.csv`: no hace falta descargarlo ni tener credenciales.

El CSV de predicción necesita estas 9 columnas, en cualquier orden:

| Columna | Tipo | Valores en el entrenamiento |
|---------|------|-----------------------------|
| `study_time_hours` | numérica | 0.5 – 8.1 |
| `attendance_percent` | numérica | 54.8 – 100 |
| `sleep_hours` | numérica | 3.2 – 10 |
| `previous_grade` | numérica | 31.3 – 100 |
| `gender` | categórica | `Female`, `Male` |
| `parental_education` | categórica | `None`, `High School`, `Bachelors`, `Masters`, `PhD` |
| `internet_access` | categórica | `Yes`, `No` |
| `extracurricular_activities` | categórica | `Yes`, `No` |
| `part_time_job` | categórica | `Yes`, `No` |

`None` en `parental_education` es una categoría (padres sin educación formal), no un dato faltante.

## Entrenar

```bash
python train.py --help
python train.py --data data/student_performance_dataset.csv --output-dir models --version v1 --seed 42
```

La variable objetivo es `final_exam_score`, fija en `common.py`. Opciones adicionales: `--test-size` (default 0.2) y `--alpha` (default 0.1). Sin `--version` se usa el siguiente `vN` libre.

El script valida las columnas y la variable objetivo, separa entrenamiento y prueba antes de ajustar el preprocesamiento y evalúa el modelo y el modelo base (la media del entrenamiento) sobre la misma partición. Después guarda en `models/<versión>/`:

- `modelo.joblib`: el pipeline completo (imputación, escalado, codificación y Lasso).
- `metrics.json`: MAE, RMSE y R² del modelo, del modelo sin recorte y del modelo base.
- `metadata.json`: versión, fecha, variables y tipos, categorías, parámetros, semilla, datos (archivo, SHA-256 y filas), partición, versiones de Python y librerías, y commit del código.

Una versión que ya existe se rechaza con un error: nunca se sobrescribe.

Lasso es determinista: con los mismos datos, semilla y versiones de `requirements.txt` se obtienen las mismas métricas.

## Predecir

```bash
python predict.py --help
python predict.py --model models/v1 --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta la carpeta de la versión o su `modelo.joblib`. La salida es el CSV de entrada, en el mismo orden, con la columna `prediccion` al final: el puntaje predicho, recortado a 0–100.

| Caso | Qué pasa |
|------|----------|
| Celda vacía | Se imputa con lo aprendido en el entrenamiento: mediana en numéricas, valor más frecuente en categóricas |
| Columnas adicionales | Se ignoran para predecir y se conservan en la salida (sirven, por ejemplo, para un identificador) |
| Columna requerida ausente, valor no numérico o categoría no vista en el entrenamiento | Error con la columna y la línea del CSV, código de salida 1 y ningún archivo de salida |
| Modelo inexistente o ilegible | Error y código de salida 1 |

## Docker

La imagen no incluye datos ni modelos: entran por montaje de carpeta y lo que se genera queda en el equipo anfitrión.

Bash (Linux o macOS). `--user` hace que los archivos generados sean del usuario y no de root:

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase2-student-performance .
mkdir -p models outputs
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" fase2-student-performance python train.py --data data/student_performance_dataset.csv --output-dir models --version v-docker --seed 42
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" fase2-student-performance python predict.py --model models/v-docker --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

PowerShell (Windows):

```powershell
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase2-student-performance .
mkdir -Force models, outputs
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models" fase2-student-performance python train.py --data data/student_performance_dataset.csv --output-dir models --version v-docker --seed 42
docker run --rm -v "${PWD}/data:/app/data:ro" -v "${PWD}/models:/app/models:ro" -v "${PWD}/outputs:/app/outputs" fase2-student-performance python predict.py --model models/v-docker --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

En Git Bash sobre Windows, ver *El volumen de Docker no persiste* en Troubleshooting.

## Pruebas

```bash
python -m pytest
```

Usan datos sintéticos pequeños y carpetas temporales: no tocan `models/` ni descargan nada. Comprueban que:

- el entrenamiento genera un artefacto cargable, sus métricas y sus metadatos;
- el modelo cargado da una predicción por fila, en orden, igual a la del modelo antes de guardarlo;
- una celda vacía se imputa con el valor aprendido en el entrenamiento;
- una columna ausente, un valor no numérico, una categoría nueva o un modelo inexistente terminan con un error claro, código distinto de cero y sin archivo de salida;
- una segunda versión conserva intacta la primera, ambas cargan, y un identificador repetido no sobrescribe nada;
- la predicción se recorta a 0–100.

## Troubleshooting

### `pip install` o `docker build` fallan con `SSLCertVerificationError` / `CERTIFICATE_VERIFY_FAILED`

Causa típica en Windows: el antivirus (ej. **Avast**, con su "Web Shield") hace inspección HTTPS e inyecta su propio certificado, que ni `pip` ni el contenedor Docker reconocen como válido — porque el tráfico de red de Docker Desktop en Windows pasa por la pila de red del host, donde el antivirus también lo intercepta. Se puede confirmar revisando las variables de entorno del sistema: si aparece algo como `NODE_EXTRA_CA_CERTS=...\Avast\wscert.pem`, esa es la causa.

**Solución:** desactivar temporalmente el Web Shield del antivirus (Avast: ícono en la bandeja → Protección → Protección principal → Web Shield → desactivar 10 min/1 h), reintentar el comando, y volver a activarlo al terminar. No hay que cambiar nada del código ni del Dockerfile — es un problema del entorno local, no del proyecto.

### El volumen de Docker (`-v`) no persiste cambios en el host (Windows / Git Bash)

En Git Bash, `$(pwd)` devuelve una ruta estilo `/c/Users/...` que Docker Desktop a veces no traduce correctamente al montar un volumen con `-v`, y el contenedor escribe en un directorio que no está realmente enlazado al host (no da error, pero los archivos "desaparecen" al salir del contenedor). Usar en su lugar la ruta completa estilo Windows (`C:/Users/<usuario>/.../fase-2/modelo:/app/modelo`), o ejecutar el comando desde PowerShell con `${PWD}`.
