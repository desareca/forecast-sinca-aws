"""Layouts de artefactos en S3 para los scripts de MLflow (arreglo de 11.3 y 11.1).

MLflow guarda los artefactos como `mlruns/<run>/` (el layout real de sinca-mlflow) o como
`mlruns/<exp>/<run>/` (layout con experimento). Cada script debe encontrar los dos.
Sin AWS: claves simuladas.
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "mlflow" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lgbm = _load("lgbm_sources")
maint = _load("maintenance")

RUN = {"run_id": "abc123", "experiment_id": "1"}
PRED_FLAT = "_mlflow/mlruns/abc123/artifacts/predictions.parquet"
PRED_EXP = "_mlflow/mlruns/1/abc123/artifacts/predictions.parquet"


# --- lgbm_sources ------------------------------------------------------------


def test_lgbm_prefijos_en_orden_plano_primero():
    assert lgbm.artifact_prefixes(RUN) == ["_mlflow/mlruns/abc123/", "_mlflow/mlruns/1/abc123/"]


def test_lgbm_encuentra_layout_plano():
    assert lgbm.resolve_predictions_prefix(RUN, {PRED_FLAT}) == "_mlflow/mlruns/abc123/"


def test_lgbm_encuentra_layout_con_experimento():
    assert lgbm.resolve_predictions_prefix(RUN, {PRED_EXP}) == "_mlflow/mlruns/1/abc123/"


def test_lgbm_sin_predictions_devuelve_none():
    assert lgbm.resolve_predictions_prefix(RUN, {"_mlflow/mlruns/abc123/artifacts/zone_icap.parquet"}) is None


def test_lgbm_comandos_incluyen_ambos_layouts():
    cmds = lgbm.s3_check_commands([RUN])
    assert len(cmds) == 2
    assert "mlruns/abc123/artifacts/predictions.parquet" in cmds[0]
    assert "mlruns/1/abc123/artifacts/predictions.parquet" in cmds[1]


# --- maintenance -------------------------------------------------------------


def test_maintenance_borrado_incluye_ambos_prefijos():
    run = {"run_id": "abc123", "experiment_id": "1", "name": "sinca-nielol-lightgbm-v99",
           "tags": {"modelo": "lightgbm"}, "params": {"version": "99"}}
    deletes = maint.build_deletes([run])
    assert deletes[0]["s3_prefixes"] == ["_mlflow/mlruns/abc123/", "_mlflow/mlruns/1/abc123/"]
