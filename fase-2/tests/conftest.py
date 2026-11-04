import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train  # noqa: E402


def make_dataset(n=80, seed=0):
    """Datos sintéticos con el esquema real; el puntaje depende sobre todo de horas de estudio y nota previa."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "study_time_hours": rng.uniform(1, 10, n).round(1),
            "attendance_percent": rng.uniform(50, 100, n).round(1),
            "sleep_hours": rng.uniform(4, 9, n).round(1),
            "previous_grade": rng.uniform(40, 100, n).round(1),
            "gender": rng.choice(["Female", "Male"], n),
            "parental_education": rng.choice(["None", "High School", "Bachelors", "Masters", "PhD"], n),
            "internet_access": rng.choice(["Yes", "No"], n),
            "extracurricular_activities": rng.choice(["Yes", "No"], n),
            "part_time_job": rng.choice(["Yes", "No"], n),
        }
    )
    score = 10 + 4 * df["study_time_hours"] + 0.5 * df["previous_grade"] + rng.normal(0, 3, n)
    df["final_exam_score"] = score.clip(0, 100).round(1)
    return df


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "datos.csv"
    make_dataset().to_csv(path, index=False)
    return path


@pytest.fixture
def models_dir(tmp_path, dataset_csv):
    """Carpeta temporal con la versión v1 ya entrenada."""
    output_dir = tmp_path / "models"
    assert train.main(["--data", str(dataset_csv), "--output-dir", str(output_dir), "--version", "v1"]) == 0
    return output_dir
