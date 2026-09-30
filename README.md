# Student Performance & Study Habits — Proyecto Integrador

## Modelos y Simulación de Sistemas I — Universidad de Antioquia — 2026-2

### Integrantes

| Nombre | GitHub |
|--------|--------|
| Brahian Ocampo Garcia | [@usuario1](https://github.com/usuario1) |
| Carlos Andres Pineda Ospina | [@usuario2](https://github.com/HyperXFury34) |

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
├── .gitignore
├── README.md
├── requirements.txt
├── fase-1/                          ← Modelo predictivo
│   ├── README.md
│   ├── fase1_modelo_predictivo.ipynb
│   └── modelo/
│       └── pipeline_modelo.joblib
├── fase-2/                          ← Scripts y Docker (pendiente)
├── fase-3/                          ← API REST (pendiente)
└── fase-4/                          ← Monitoreo (pendiente)
```

---

## Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| Fase 1 | Modelo predictivo | ✅ Completada |
| Fase 2 | Scripts y Docker | ⬜ Pendiente |
| Fase 3 | API REST | ⬜ Pendiente |
| Fase 4 | Monitoreo básico | ⬜ Pendiente |

---

## Instalación y ejecución

### Requisitos previos
- Python 3.10 o superior

### Pasos

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/USUARIO/student-performance-project.git
   cd student-performance-project
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Descargar el dataset desde [Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset) y colocar `student_performance_dataset.csv` en la carpeta `fase-1/`.

4. Ejecutar el notebook:
   ```bash
   cd fase-1
   jupyter notebook fase1_modelo_predictivo.ipynb
   ```

---

## Recursos

- [Curso: IA para las Ciencias e Ingenierías](https://rramosp.github.io/ai4eng.v1/)
- [Scripts de referencia — sklearn_scripts](https://github.com/rramosp/sklearn_scripts)
