# Fase 1 — Modelo Predictivo

## Student Performance & Study Habits Dataset

### Integrantes
- Brahian Ocampo Garcia
- Carlos Andres Pineda Ospina

---

## Descripción

Notebook ejecutable que construye un modelo de regresión para predecir el puntaje del examen final (`final_exam_score`) de estudiantes, a partir de hábitos de estudio y factores demográficos.

**Dataset:** [Student Performance & Study Habits — Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset)

---

## Variables del dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `student_id` | int | Identificador (excluido del modelo) |
| `gender` | categórica | Género del estudiante |
| `study_time_hours` | numérica | Horas de estudio semanales |
| `attendance_percent` | numérica | Porcentaje de asistencia a clase |
| `sleep_hours` | numérica | Horas de sueño promedio |
| `parental_education` | categórica | Nivel educativo de los padres (~10% nulos) |
| `internet_access` | categórica | Acceso a internet (Yes/No) |
| `extracurricular_activities` | categórica | Actividades extracurriculares (Yes/No) |
| `part_time_job` | categórica | Trabajo de medio tiempo (Yes/No) |
| `previous_grade` | numérica | Calificación previa |
| **`final_exam_score`** | **numérica** | **Variable objetivo (0–100)** |
| `final_grade` | categórica | Calificación en letras (excluida: derivada del target) |

---

## Estructura de la carpeta

```
fase-1/
├── fase1_modelo_predictivo.ipynb   # Notebook ejecutable
├── modelo/
│   └── pipeline_modelo.joblib      # Pipeline completo (preprocesamiento + modelo)
└── README.md
```

---

## Prevención de fuga de información

1. `final_grade` fue excluida del modelo porque se deriva directamente de `final_exam_score`.
2. La división train/test se realizó **antes** de cualquier preprocesamiento.
3. El pipeline de preprocesamiento se ajusta (`.fit()`) únicamente sobre datos de entrenamiento.

---

## Cómo ejecutar

1. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset).
2. Colocar el archivo `student_performance_dataset.csv` en esta carpeta.
3. Instalar dependencias:
   ```bash
   pip install -r ../requirements.txt
   ```
4. Ejecutar el notebook:
   ```bash
   jupyter notebook fase1_modelo_predictivo.ipynb
   ```

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
