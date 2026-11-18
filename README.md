# Student Performance & Study Habits — Proyecto Integrador

## Modelos y Simulación de Sistemas I — Universidad de Antioquia — 2026-2

### Integrantes

| Nombre | GitHub |
|--------|--------|
| Brahian Ocampo Garcia | [@Brahian2215](https://github.com/Brahian2215) |
| Carlos Andres Pineda Ospina | [@HyperXFury34](https://github.com/HyperXFury34) |

---

## Descripción

Proyecto integrador que lleva un modelo de Machine Learning desde un notebook hasta un prototipo desplegable. El objetivo es predecir el puntaje del examen final (`final_exam_score`) de estudiantes universitarios a partir de sus hábitos de estudio, factores de estilo de vida y datos demográficos.

**Dataset:** [Student Performance & Study Habits — Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset)

| Característica | Detalle |
|----------------|---------|
| Observaciones | 1000 estudiantes |
| Variables predictoras | 9 (numéricas y categóricas) |
| Variable objetivo | `final_exam_score` (0–100) |
| Tipo de problema | Regresión |

---

## Estructura del repositorio

```
├── .dockerignore
├── .gitignore
├── README.md
├── requirements.txt
├── fase-1/                              ← Modelo predictivo
│   ├── README.md
│   ├── fase1_modelo_predictivo.ipynb
│   ├── student_performance_dataset.csv
│   └── modelo/
│       └── pipeline_modelo.joblib
├── fase-2/                              ← Scripts y Docker
│   ├── README.md
│   ├── requirements.txt
│   ├── common.py
│   ├── train.py
│   ├── predict.py
│   ├── Dockerfile
│   ├── data/
│   ├── models/
│   └── tests/
├── fase-3/                              ← API REST
│   ├── README.md
│   ├── requirements.txt
│   ├── api.py
│   ├── cliente.py
│   ├── Dockerfile
│   └── tests/
└── fase-4/                              ← Monitoreo
    ├── README.md
    ├── requirements.txt
    ├── registro.py
    ├── api_registro.py
    ├── monitoreo.py
    ├── registro_muestra.csv
    ├── Dockerfile
    └── tests/
```

---

## Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| Fase 1 | Modelo predictivo | ✅ Completada |
| Fase 2 | Scripts y Docker | ✅ Completada |
| Fase 3 | API REST | ✅ Completada |
| Fase 4 | Monitoreo básico | ✅ Completada |

---

## Instalación y ejecución

### Requisitos previos
- Python 3.10 o superior

### Pasos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/carlospineda1-coder/Student-Performance-Study-Habits.git
   cd Student-Performance-Study-Habits
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar el notebook:
   ```bash
   cd fase-1
   jupyter notebook fase1_modelo_predictivo.ipynb
   ```

Las fases 2, 3 y 4 tienen su propio entorno e instrucciones en [fase-2/README.md](fase-2/README.md), [fase-3/README.md](fase-3/README.md) y [fase-4/README.md](fase-4/README.md).

---

## Recursos

- [Curso: IA para las Ciencias e Ingenierías](https://rramosp.github.io/ai4eng.v1/)
- [Scripts de referencia — sklearn_scripts](https://github.com/rramosp/sklearn_scripts)
