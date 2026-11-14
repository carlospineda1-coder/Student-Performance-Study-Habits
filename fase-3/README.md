# Fase 3 — API REST

Expone el modelo de la Fase 2 por HTTP: `GET /health` informa que la API está en ejecución, `POST /predict` devuelve el puntaje estimado de un estudiante y la versión que lo calculó, y `POST /train` entrena y guarda una versión nueva. Está hecha con FastAPI.

**Problema:** regresión. Se predice `final_exam_score` (0–100) a partir de hábitos de estudio, estilo de vida y datos demográficos.

**Integrantes:** Brahian Ocampo Garcia ([@Brahian2215](https://github.com/Brahian2215)) y Carlos Andres Pineda Ospina ([@HyperXFury34](https://github.com/HyperXFury34)).

**Versión entregada:** etiqueta de Git `entrega-3`.

## Cambios respecto a la Fase 2

El modelo, los datos y el preprocesamiento son los mismos. La API no copia código: importa `common.py` y `train.py` de `fase-2/` y usa sus carpetas `data/` y `models/`. Por eso la imagen de Docker se construye desde la raíz del repositorio.

## Estructura

```
fase-3/
├── api.py             # las tres rutas y la validación de entradas
├── cliente.py         # cliente de prueba: consulta /health y pide una predicción
├── requirements.txt   # las dependencias de fase-2 más FastAPI y Uvicorn
├── Dockerfile
└── tests/
```

## Configuración

La API se configura con variables de entorno. Se leen en cada solicitud.

| Variable | Qué define | Valor por defecto |
|----------|------------|-------------------|
| `MODEL_VERSION` | Versión que usa `/predict` | `v1` |
| `MODELS_DIR` | Carpeta de las versiones | `fase-2/models` |
| `DATA_PATH` | CSV con el que entrena `/train` | `fase-2/data/student_performance_dataset.csv` |

**Seleccionar una versión.** Las disponibles son las carpetas de `fase-2/models/` (`v1`, `v-docker` y `v2`, más las que cree `/train`). Para usar otra, se arranca la API con `MODEL_VERSION=<versión>`. Si esa versión no existe, `/predict` responde 503 y `/health` y `/train` siguen funcionando.

## Rutas

### `GET /health`

```json
{"status": "ok"}
```

### `POST /predict`

Recibe un objeto JSON con las nueve variables predictoras. Todas son obligatorias; la variable objetivo no se envía. Unidades y valores válidos: sección *Datos* de [fase-2/README.md](../fase-2/README.md).

Solicitud (primera fila del dataset, cuyo puntaje real es 100.0):

```json
{
  "study_time_hours": 4.0,
  "attendance_percent": 98.0,
  "sleep_hours": 6.5,
  "previous_grade": 76.9,
  "gender": "Male",
  "parental_education": "Bachelors",
  "internet_access": "Yes",
  "extracurricular_activities": "Yes",
  "part_time_job": "No"
}
```

Respuesta, HTTP 200:

```json
{"prediccion": 93.72052191273534, "version": "v1"}
```

| Caso | Respuesta |
|------|-----------|
| Campo con `null` | 200. Se imputa con la regla guardada en la Fase 2: mediana en numéricas, valor más frecuente en categóricas |
| Campos adicionales, incluida la variable objetivo | 200. Se ignoran |
| Falta un campo | 422, `Falta el campo obligatorio sleep_hours` |
| Tipo incompatible (texto donde va un número, o al revés) | 422, `El campo sleep_hours debe ser un número o null` |
| Categoría no vista en el entrenamiento | 422, con el campo y los valores válidos |
| El cuerpo no es un objeto JSON | 422 |
| La versión configurada no existe | 503, `La versión configurada (v9) no está disponible: ...` |

Los errores llegan como `{"detail": "<mensaje>"}`. El tipo es estricto: `"7"` no se acepta como número. `"None"` en `parental_education` es una categoría (padres sin educación formal); `null` es un dato faltante.

### `POST /train`

No necesita cuerpo. Entrena con `DATA_PATH` y la configuración de la Fase 2 (semilla 42, partición 80/20, Lasso con `alpha=0.1`), guarda la siguiente versión libre (`v2`, `v3`…) en `MODELS_DIR` con su `modelo.joblib`, `metrics.json` y `metadata.json`, y responde cuando los archivos ya están guardados. Las versiones anteriores no se tocan.

Respuesta, HTTP 201:

```json
{"version": "v2", "metricas": {"mae": 4.865736449463372, "rmse": 6.294585894007378, "r2": 0.6318939850653977}}
```

`/predict` sigue usando `MODEL_VERSION`. Para activar la versión nueva hay que reiniciar la API con `MODEL_VERSION=v2`.

## Docker

Desde la **raíz del repositorio**. La imagen no incluye datos ni modelos: se montan las carpetas de `fase-2/`, y las versiones que crea `/train` quedan en `fase-2/models/` del equipo anfitrión. La API queda en el puerto 8000.

Bash (Linux o macOS):

```bash
docker build -f fase-3/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase3-student-performance .
docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 -e MODEL_VERSION=v1 -v "$PWD/fase-2/data:/app/fase-2/data:ro" -v "$PWD/fase-2/models:/app/fase-2/models" fase3-student-performance
```

PowerShell (Windows):

```powershell
docker build -f fase-3/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase3-student-performance .
docker run --rm -p 8000:8000 -e MODEL_VERSION=v1 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" fase3-student-performance
```

En otra terminal:

```bash
curl http://localhost:8000/health
python fase-3/cliente.py
curl -X POST http://localhost:8000/train
ls fase-2/models
```

`cliente.py` solo usa la biblioteca estándar de Python y acepta `--url` si la API está en otra dirección y `--omitir CAMPO` para enviar el registro sin un campo y ver el rechazo. La documentación interactiva queda en `http://localhost:8000/docs`.

## Ejecución local y pruebas

Python 3.11, desde la carpeta `fase-3`.

Bash:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
PYTHONPATH=../fase-2 uvicorn api:app --port 8000
```

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
$env:PYTHONPATH = "../fase-2"; uvicorn api:app --port 8000
```

Las pruebas usan el cliente de pruebas de FastAPI y entrenan una versión en una carpeta temporal: no tocan `fase-2/models/` ni necesitan la API en ejecución. Comprueban que:

- `/health` responde 200;
- una entrada válida devuelve la predicción y la versión;
- una entrada sin un campo obligatorio se rechaza con el nombre del campo;
- un tipo incompatible y una categoría nueva se rechazan;
- un `null` se imputa con el valor aprendido en el entrenamiento;
- la variable objetivo no hace falta y, si llega, se ignora;
- una versión no disponible responde 503;
- `/train` guarda la siguiente versión, conserva la anterior y no cambia la que usa `/predict`.

## Evidencia de ejecución

En local (Bash, Python 3.11.2):

```
$ python -m pytest
11 passed in 0.44s

$ python cliente.py
GET /health -> 200 {"status": "ok"}
POST /predict
{
  "study_time_hours": 4.0,
  ...
}
-> 200 {"prediccion": 93.72052191273534, "version": "v1"}

$ curl -X POST http://localhost:8000/train
{"version":"v2","metricas":{"mae":4.865736449463372,"rmse":6.294585894007378,"r2":0.6318939850653977}}

$ curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"study_time_hours": 4.0, "attendance_percent": 98.0}'
{"detail":"Falta el campo obligatorio sleep_hours; Falta el campo obligatorio previous_grade; ..."}
```

En Docker (Bash, con los comandos de la sección Docker):

```
$ curl http://localhost:8000/health
{"status":"ok"}

$ python fase-3/cliente.py
...
-> 200 {"prediccion": 93.72052191273534, "version": "v1"}

$ curl -X POST http://localhost:8000/train
{"version":"v2","metricas":{"mae":4.865736449463372,"rmse":6.294585894007378,"r2":0.6318939850653977}}

$ ls fase-2/models
v1  v2  v-docker
```

`fase-2/models/v2/` es la versión que creó esa llamada a `/train` dentro del contenedor. Quedó en el equipo anfitrión y está versionada en el repositorio.

## Fuentes

- Documentación de FastAPI: [cuerpo de la solicitud](https://fastapi.tiangolo.com/tutorial/body/), [manejo de errores](https://fastapi.tiangolo.com/tutorial/handling-errors/) y [pruebas](https://fastapi.tiangolo.com/tutorial/testing/)
- Documentación de Pydantic: [modo estricto](https://docs.pydantic.dev/latest/concepts/strict_mode/)
- Documentación de Uvicorn: [opciones de arranque](https://www.uvicorn.org/settings/)
