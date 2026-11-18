# Fase 4 — Monitoreo

Agrega a la API de la Fase 3 un registro de las solicitudes a `/predict` y un script que calcula dos métricas de operación: el total de solicitudes y el porcentaje de fallidas.

**Integrantes:** Brahian Ocampo Garcia ([@Brahian2215](https://github.com/Brahian2215)) y Carlos Andres Pineda Ospina ([@HyperXFury34](https://github.com/HyperXFury34)).

**Versión entregada:** etiqueta de Git `entrega-3`.

## Cambios respecto a la Fase 3

Las rutas, la validación y las respuestas son las de `fase-3/api.py`, sin copiar: `api_registro.py` envuelve esa API con el registro, sin modificarla. El cliente de `fase-3/` ganó la opción `--omitir CAMPO` para enviar la solicitud inválida del ensayo.

## Estructura

```
fase-4/
├── registro.py            # escribe una fila del registro
├── api_registro.py        # la API de fase-3 con el registro activado
├── monitoreo.py           # total de solicitudes y porcentaje de fallidas
├── registro_muestra.csv   # registro de la ejecución del reporte
├── requirements.txt       # las mismas dependencias de fase-3
├── Dockerfile
├── logs/                  # registro de trabajo (no se versiona)
└── tests/
```

## El registro

Cada solicitud a `/predict` agrega una fila a un CSV cuando ya se conoce su resultado, incluidas las que la validación rechaza. Las demás rutas no se registran.

| Columna | Contenido |
|---------|-----------|
| `fecha_hora` | Momento de la respuesta, en UTC |
| `estado_http` | Código HTTP devuelto |
| `version` | Versión que predijo; vacía si no se usó ningún modelo |
| `prediccion` | Valor devuelto; vacío en un error |

El archivo se abre en modo de agregar, así que se conserva entre ejecuciones. Si no se puede escribir, la API responde igual y el error aparece en la consola del servidor. Su ubicación es `fase-4/logs/predicciones.csv` y se cambia con la variable de entorno `LOG_PATH`. Las demás variables (`MODEL_VERSION`, `MODELS_DIR`, `DATA_PATH`) son las de la [Fase 3](../fase-3/README.md#configuración).

## Docker

Desde la **raíz del repositorio**. El registro queda en `fase-4/logs/` del equipo anfitrión; los datos y los modelos son los de `fase-2/`. La API queda en el puerto 8000 y arranca con la versión `v2`.

Bash (Linux o macOS). La carpeta `fase-4/logs` se crea antes para que pertenezca al usuario y no a root:

```bash
mkdir -p fase-4/logs
docker build -f fase-4/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase4-student-performance .
docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 -e MODEL_VERSION=v2 -v "$PWD/fase-2/data:/app/fase-2/data:ro" -v "$PWD/fase-2/models:/app/fase-2/models" -v "$PWD/fase-4/logs:/app/fase-4/logs" fase4-student-performance
```

PowerShell (Windows):

```powershell
mkdir -Force fase-4/logs
docker build -f fase-4/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t fase4-student-performance .
docker run --rm -p 8000:8000 -e MODEL_VERSION=v2 -v "${PWD}/fase-2/data:/app/fase-2/data:ro" -v "${PWD}/fase-2/models:/app/fase-2/models" -v "${PWD}/fase-4/logs:/app/fase-4/logs" fase4-student-performance
```

## Ensayo y métricas

Con la API en ejecución y el registro vacío, en otra terminal y desde la raíz del repositorio: dos solicitudes válidas, una sin un campo obligatorio y el script de monitoreo.

```bash
python fase-3/cliente.py
python fase-3/cliente.py
python fase-3/cliente.py --omitir sleep_hours
python fase-4/monitoreo.py
```

`monitoreo.py` solo usa la biblioteca estándar. Sin argumentos lee `fase-4/logs/predicciones.csv`; también acepta la ruta de otro registro, por ejemplo `python fase-4/monitoreo.py fase-4/registro_muestra.csv`. Cuenta como fallidas las respuestas con código entre 400 y 599. Con un registro vacío o inexistente muestra:

```
Total de solicitudes: 0
Porcentaje de solicitudes fallidas: sin solicitudes
```

Ese porcentaje describe el funcionamiento del servicio, no la calidad del modelo: para medirla harían falta los puntajes reales de los estudiantes consultados.

## Ejecución local y pruebas

Python 3.11, desde la carpeta `fase-4`.

Bash:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
PYTHONPATH=../fase-2:../fase-3 MODEL_VERSION=v2 uvicorn api_registro:app --port 8000
```

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
$env:PYTHONPATH = "../fase-2;../fase-3"; $env:MODEL_VERSION = "v2"; uvicorn api_registro:app --port 8000
```

Las pruebas escriben el registro en una carpeta temporal y predicen con la versión `v1` de `fase-2/models/`, que solo se lee. Comprueban que:

- cada solicitud a `/predict` deja una sola fila, también la rechazada por validación;
- un 503 y un error interno (500) se registran sin versión ni predicción, y las demás rutas no se registran;
- el registro no cambia la respuesta que recibe el cliente, ni siquiera si no se puede escribir, y conserva el historial;
- la API de `fase-3/` sigue sin registro: solo se registra al arrancar `api_registro`;
- dos solicitudes válidas y una inválida dan 3 solicitudes y 33,3 % de fallos;
- los códigos 4xx y 5xx cuentan como fallos;
- un registro vacío o inexistente da total 0 y «sin solicitudes», y uno con una fila cortada da un error legible.

## Reporte

### Versiones

Las dos están en `fase-2/models/`, cada una con su `modelo.joblib`, `metrics.json` y `metadata.json`.

| Versión | Origen | Papel |
|---------|--------|-------|
| `v2` | `POST /train` dentro del contenedor de la Fase 3 | La que usa la API (`MODEL_VERSION=v2`) |
| `v1` | `train.py` en local, Fase 2 | La anterior |

Para volver a la anterior se detiene la API y se arranca con `-e MODEL_VERSION=v1` en lugar de `-e MODEL_VERSION=v2`; no se borra ni se mueve nada. La columna `version` del registro confirma cuál respondió cada solicitud. Las dos se entrenaron con los mismos datos y la misma configuración, y Lasso es determinista: sus métricas de prueba coinciden (MAE 4.866).

### Política de reentrenamiento

El modelo se revisa cuando se reúnan 100 estudiantes nuevos con su puntaje real del examen final. Un integrante del equipo reserva 30 de esos casos, agrega los otros 70 al CSV de entrenamiento y ejecuta `POST /train` para obtener una candidata. La candidata se activa con `MODEL_VERSION` solo si su MAE sobre los 30 casos reservados es menor que el de la versión activa sobre esos mismos casos; si no, se conserva la activa.

## Fuentes

- Documentación de FastAPI: [middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- Documentación de Python: [csv](https://docs.python.org/3/library/csv.html)
