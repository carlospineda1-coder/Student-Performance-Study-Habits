import json

import pandas as pd

import train
from common import FEATURE_COLS, TARGET_COL, load_model
from conftest import make_dataset


def read_version(version_dir):
    return {name: (version_dir / name).read_bytes() for name in ("modelo.joblib", "metrics.json", "metadata.json")}


def test_entrenamiento_genera_artefacto_cargable_metricas_y_metadatos(models_dir):
    version_dir = models_dir / "v1"
    pipeline = load_model(str(version_dir))
    assert len(pipeline.predict(make_dataset(n=4)[FEATURE_COLS])) == 4

    metrics = json.loads((version_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"model", "model_unclipped", "baseline"}
    assert metrics["model"]["mae"] < metrics["baseline"]["mae"]

    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["version"] == "v1"
    assert metadata["target"] == TARGET_COL
    assert [feature["name"] for feature in metadata["features"]] == FEATURE_COLS
    assert metadata["seed"] == 42
    assert metadata["data"]["n_rows"] == 80
    assert metadata["split"]["n_train"] + metadata["split"]["n_test"] == 80
    assert "None" in metadata["categories"]["parental_education"]


def test_segunda_version_conserva_la_primera_y_ambas_cargan(models_dir, dataset_csv):
    before = read_version(models_dir / "v1")

    assert train.main(["--data", str(dataset_csv), "--output-dir", str(models_dir)]) == 0

    assert read_version(models_dir / "v1") == before
    assert (models_dir / "v2" / "modelo.joblib").exists()
    load_model(str(models_dir / "v1"))
    load_model(str(models_dir / "v2"))


def test_version_repetida_no_sobrescribe(models_dir, dataset_csv, capsys):
    before = read_version(models_dir / "v1")

    code = train.main(["--data", str(dataset_csv), "--output-dir", str(models_dir), "--version", "v1", "--seed", "7"])

    assert code != 0
    assert "ya existe" in capsys.readouterr().err
    assert read_version(models_dir / "v1") == before


def test_falta_variable_objetivo(tmp_path, dataset_csv, capsys):
    path = tmp_path / "sin_objetivo.csv"
    pd.read_csv(dataset_csv).drop(columns=[TARGET_COL]).to_csv(path, index=False)

    code = train.main(["--data", str(path), "--output-dir", str(tmp_path / "models")])

    assert code != 0
    assert TARGET_COL in capsys.readouterr().err
    assert not (tmp_path / "models").exists()
