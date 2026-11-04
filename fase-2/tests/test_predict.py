import numpy as np
import pandas as pd
import pytest

import predict
import train
from common import load_dataset, load_model, predict_scores, prepare_features
from conftest import make_dataset


def run_predict(models_dir, input_df, tmp_path):
    input_path, output_path = tmp_path / "entrada.csv", tmp_path / "salida" / "predicciones.csv"
    input_df.to_csv(input_path, index=False)
    code = predict.main(["--model", str(models_dir / "v1"), "--input", str(input_path), "--output", str(output_path)])
    return code, output_path


def test_prediccion_por_fila_igual_al_modelo_antes_de_guardar(tmp_path, dataset_csv):
    pipeline, _, _, _ = train.train_and_evaluate(load_dataset(dataset_csv), seed=42, test_size=0.2, alpha=0.1)
    new_rows = make_dataset(n=15, seed=1).drop(columns=["final_exam_score"])
    expected = predict_scores(pipeline, prepare_features(new_rows))

    train.save_version(str(tmp_path / "models" / "v1"), pipeline, {}, {})
    code, output_path = run_predict(tmp_path / "models", new_rows, tmp_path)

    assert code == 0
    result = pd.read_csv(output_path)
    assert len(result) == len(new_rows)
    np.testing.assert_allclose(result["prediccion"], expected, rtol=1e-9)
    pd.testing.assert_series_equal(result["study_time_hours"], new_rows["study_time_hours"])


def test_valores_faltantes_se_imputan_con_lo_aprendido_en_entrenamiento(models_dir, tmp_path):
    rows = make_dataset(n=3, seed=2).drop(columns=["final_exam_score"])
    rows["study_time_hours"] = rows["study_time_hours"].astype(object)
    rows.loc[0, "study_time_hours"] = None
    rows.loc[1, "gender"] = None

    code, output_path = run_predict(models_dir, rows, tmp_path)

    assert code == 0
    result = pd.read_csv(output_path, keep_default_na=False, na_values=[""])
    assert len(result) == 3 and result["prediccion"].notna().all()

    pipeline = load_model(str(models_dir / "v1"))
    num_imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    filled = rows.iloc[[0]].copy()
    filled["study_time_hours"] = num_imputer.statistics_[0]
    assert result.loc[0, "prediccion"] == pytest.approx(predict_scores(pipeline, prepare_features(filled))[0])


def test_columna_ausente_termina_con_error_claro_y_sin_archivo(models_dir, tmp_path, capsys):
    rows = make_dataset(n=3).drop(columns=["final_exam_score", "sleep_hours"])

    code, output_path = run_predict(models_dir, rows, tmp_path)

    assert code != 0
    assert "Faltan columnas requeridas: sleep_hours" in capsys.readouterr().err
    assert not output_path.exists()


@pytest.mark.parametrize(
    "column, value, message",
    [("attendance_percent", "alta", "debe ser numérica"), ("internet_access", "Si", "no vistos en el entrenamiento")],
)
def test_tipo_incompatible_o_categoria_nueva_termina_con_error(models_dir, tmp_path, capsys, column, value, message):
    rows = make_dataset(n=3).drop(columns=["final_exam_score"]).astype({column: object})
    rows.loc[1, column] = value

    code, output_path = run_predict(models_dir, rows, tmp_path)

    assert code != 0
    assert message in capsys.readouterr().err
    assert not output_path.exists()


def test_modelo_inexistente_termina_con_error(tmp_path, capsys):
    code, output_path = run_predict(tmp_path / "no-existe", make_dataset(n=2), tmp_path)

    assert code != 0
    assert "No existe el modelo" in capsys.readouterr().err
    assert not output_path.exists()


def test_prediccion_se_recorta_a_la_escala_del_puntaje(models_dir, tmp_path):
    rows = make_dataset(n=2).drop(columns=["final_exam_score"])
    rows["study_time_hours"] = [1000.0, -1000.0]

    code, output_path = run_predict(models_dir, rows, tmp_path)

    assert code == 0
    assert pd.read_csv(output_path)["prediccion"].tolist() == [100.0, 0.0]
