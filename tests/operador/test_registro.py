"""Tarea 1.6: nombres, tags y artefactos de los runs del operador."""
import numpy as np
import pandas as pd

from analysis import load as analysis_load
from operador import registro as G


def _log(state, rep, exp, etapa, seed, tipo="operador"):
    name = G.run_name(rep, exp, "aedmd", 1, seed)
    frames = {"predictions.parquet": pd.DataFrame({"timestamp": [1], "y_true": [1.0], "y_pred": [1.0]})}
    return G.log_variante(str(state), name, G.tags_base(rep, etapa, "ae-dmd", seed), {"tipo": tipo, "etapa": etapa},
                          {"sel_wrmse_media": 1.0, "nan": float("nan")}, frames, {"operator.npz": {"A": np.eye(2)}})


def test_artefactos_legibles_con_artifact_location_ajeno(estado_mlflow):
    run_id = _log(estado_mlflow, "hankel", "e11a", "1.1", 0)
    path = analysis_load.artifact_path(str(estado_mlflow), run_id, "predictions.parquet")
    assert path is not None and pd.read_parquet(path).shape == (1, 3)
    assert analysis_load.artifact_path(str(estado_mlflow), run_id, "operator.npz")
    run = analysis_load.run_by_id(str(estado_mlflow), run_id)
    assert run["tags"]["stage"] == G.STAGE and run["tags"]["modelo"] == "dmdop-hankel"
    assert "nan" not in run["metrics"]


def test_semillas_y_etapas_distinguibles(estado_mlflow):
    names = {G.run_name("conv1d", exp, "aedmd", 1, seed) for exp in ("e11d", "e12") for seed in (0, 1, 2)}
    assert len(names) == 6
    for exp, etapa in (("e11d", "1.1"), ("e12", "1.2")):
        for seed in (0, 1, 2):
            _log(estado_mlflow, "conv1d", exp, etapa, seed)
    runs = G.runs_de_etapa(str(estado_mlflow))
    assert len({r["run_name"] for r in runs}) == 6
    assert {(r["tags"]["etapa"], r["tags"]["seed"]) for r in runs} == {(e, str(s)) for e in ("1.1", "1.2") for s in (0, 1, 2)}


def test_oraculo_diagnostico_fuera_de_vistas_por_serie(estado_mlflow):
    run_id = _log(estado_mlflow, "conv1d", "e10", "1.0", 0, tipo="diagnostico")
    run = analysis_load.run_by_id(str(estado_mlflow), run_id)
    assert run["params"]["tipo"] == "diagnostico"
    assert not analysis_load.is_series_run(run)
    assert not analysis_load.is_ae_run(run)


def test_nombre_test_por_serie():
    assert G.run_name_test("Padre Las Casas II", "pm25", "hankel", "aedmd", 1, 0) == \
        "sinca-padre_las_casas_ii-pm25-dmdop-hankel-aedmd-v1-s0"
