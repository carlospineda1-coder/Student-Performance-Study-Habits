# Fase 1 — Modelo Predictivo

## Student Performance & Study Habits Dataset

### Integrantes
- Brahian Ocampo Garcia — [@Brahian2215](https://github.com/Brahian2215)
- Carlos Andres Pineda Ospina — [@carlospineda1-coder](https://github.com/carlospineda1-coder)

---

## Problema

Predecir el puntaje del examen final (`final_exam_score`, escala 0–100) de un estudiante a partir de sus hábitos de estudio, factores de estilo de vida y datos demográficos. Es un problema de **regresión**. El propósito es identificar de forma temprana a estudiantes en riesgo de bajo rendimiento.

**Dataset:** [Student Performance & Study Habits — Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset) — 1000 observaciones, 9 variables predictoras. El archivo está versionado en esta carpeta.

---

## Algoritmo, métrica y resultados

**Algoritmo:** Lasso (`alpha=0.1`), dentro de un pipeline con escalado y codificación one-hot. Se comparó contra Ridge, regresión lineal, árbol de decisión, Random Forest y Gradient Boosting mediante validación cruzada de 5 particiones sobre el conjunto de entrenamiento. Los tres modelos lineales quedaron empatados dentro del margen de error, y se fijó Lasso por ser el más simple de ellos: anula los coeficientes que no aportan.

**Métrica principal:** MAE, porque se expresa en puntos del examen y se interpreta directamente. Se acompaña de RMSE y R².

**Resultados sobre el conjunto de prueba (200 estudiantes, 20%):**

| | Modelo base (media) | Lasso |
|---|---|---|
| MAE | 8.5847 | **4.9360** |
| RMSE | 10.5013 | **6.3659** |
| R² | −0.0245 | **0.6235** |

El modelo reduce el error absoluto medio un 42.5% frente al modelo base y explica el 62% de la variabilidad del puntaje. La variable de mayor peso son las horas de estudio semanales.

---

## Variables del dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `student_id` | int | Identificador (excluido del modelo) |
| `gender` | categórica | Género del estudiante |
| `study_time_hours` | numérica | Horas de estudio semanales |
| `attendance_percent` | numérica | Porcentaje de asistencia a clase |
| `sleep_hours` | numérica | Horas de sueño promedio |
| `parental_education` | categórica | Nivel educativo de los padres (cinco niveles, incluido `None`) |
| `internet_access` | categórica | Acceso a internet (Yes/No) |
| `extracurricular_activities` | categórica | Actividades extracurriculares (Yes/No) |
| `part_time_job` | categórica | Trabajo de medio tiempo (Yes/No) |
| `previous_grade` | numérica | Calificación previa |
| **`final_exam_score`** | **numérica** | **Variable objetivo (0–100)** |
| `final_grade` | categórica | Calificación en letras (excluida: derivada del objetivo) |

---

## Valores faltantes

Al leer el archivo, pandas reporta 102 nulos (10.2%) en `parental_education`. El análisis del notebook muestra que no son datos ausentes: son la cadena literal `None`, que pandas convierte a nulo por estar en su lista de valores nulos por defecto, y que en esta variable significa *padres sin educación formal* — el nivel más bajo de la escala. Se conservan como categoría y no se imputan. El archivo no contiene ningún campo vacío.

---

## Prevención de fuga de información

1. `final_grade` se excluye del modelo porque se deriva de `final_exam_score`.
2. La división train/test se realiza **antes** de cualquier transformación.
3. Todas las transformaciones viven dentro del pipeline, cuyo `.fit()` se ejecuta únicamente sobre los datos de entrenamiento.

---

## Estructura de la carpeta

```
fase-1/
├── fase1_modelo_predictivo.ipynb   # Notebook ejecutable
├── student_performance_dataset.csv # Datos
├── modelo/
│   └── pipeline_modelo.joblib      # Pipeline completo (preprocesamiento + modelo)
└── README.md
```

---

## Cómo ejecutar

```bash
pip install -r ../requirements.txt
jupyter notebook fase1_modelo_predictivo.ipynb
```

El notebook se ejecuta de principio a fin sin intervención manual.

---

## Uso del modelo guardado

```python
import joblib
import pandas as pd

pipeline = joblib.load('modelo/pipeline_modelo.joblib')

estudiante = pd.DataFrame({
    'study_time_hours': [5.0],
    'attendance_percent': [90.0],
    'sleep_hours': [7.0],
    'previous_grade': [80.0],
    'gender': ['Female'],
    'parental_education': ['Bachelors'],
    'internet_access': ['Yes'],
    'extracurricular_activities': ['Yes'],
    'part_time_job': ['No']
})

print(f"Puntaje predicho: {pipeline.predict(estudiante)[0]:.1f}")
```
