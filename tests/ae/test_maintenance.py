"""Tests del script de mantenimiento de MLflow (ae-dmd-prep, tarea 8.2). Sin AWS: MLflow SQLite temporal."""
import hashlib
import importlib.util
import json
import pathlib
import sqlite3

import pytest

pytest.importorskip("mlflow")

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "mlflow" / "maintenance.py"
EXPERIMENT = "sinca-baseline"


@pytest.fixture(scope="module")
def maint():
    spec = importlib.util.spec_from_file_location("maintenance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(name, tags, params):
    import mlflow

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
    return run.info.run_id


@pytest.fixture
def mlflow_dir(tmp_path):
    """MLflow temporal con duplicados de `ref-*`, `sinca-zona-ae_*`, un v99 y dos `lightgbm-v1` de una serie."""
    import mlflow

    state = tmp_path / "state"
    state.mkdir()
    mlflow.set_tracking_uri(f"sqlite:///{(state / 'mlflow.sqlite').as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    ids = {}
    ids["ref24"] = _run("ref-mean", {"modelo": "ref-mean", "tipo": "referencia"},
                        {"tipo": "referencia", "L": 24, "d": 16, "version": 1})
    ids["ref72"] = _run("ref-mean", {"modelo": "ref-mean", "tipo": "referencia"},
                        {"tipo": "referencia", "L": 72, "d": 16, "version": 1})
    base = {"arch": "conv1d", "version": 2, "d": 16, "aux_input": "True", "seed": 0}
    ids["conv_a"] = _run("sinca-zona-ae_conv1d-v2", {"modelo": "ae-conv1d", "stage": "ae-reconstruccion"},
                         {**base, "w_aux": 0.1})
    ids["conv_b"] = _run("sinca-zona-ae_conv1d-v2", {"modelo": "ae-conv1d", "stage": "ae-reconstruccion"},
                         {**base, "w_aux": 0.5})
    series = {"estacion": "Nielol", "pollutant": "pm25"}
    ids["v99"] = _run("sinca-nielol-lightgbm-v99", {"modelo": "lightgbm", "serie": "Nielol|pm25"},
                      {**series, "version": 99})
    ids["v1a"] = _run("sinca-nielol-lightgbm-v1", {"modelo": "lightgbm", "serie": "Nielol|pm25"},
                      {**series, "version": 1})
    ids["v1b"] = _run("sinca-nielol-lightgbm-v1", {"modelo": "lightgbm", "serie": "Nielol|pm25"},
                      {**series, "version": 1})
    return state, ids


def _names(state):
    con = sqlite3.connect(state / "mlflow.sqlite")
    try:
        return dict(con.execute("select run_uuid, name from runs").fetchall())
    finally:
        con.close()


def _lifecycle(state):
    con = sqlite3.connect(state / "mlflow.sqlite")
    try:
        return dict(con.execute("select run_uuid, lifecycle_stage from runs").fetchall())
    finally:
        con.close()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_no_modifica_el_sqlite(maint, mlflow_dir, tmp_path, capsys):
    state, _ = mlflow_dir
    db = state / "mlflow.sqlite"
    before_hash, before_mtime = _hash(db), db.stat().st_mtime_ns
    assert maint.main(["plan", "--state-dir", str(state), "--out", str(tmp_path / "plan.json")]) == 0
    assert _hash(db) == before_hash
    assert db.stat().st_mtime_ns == before_mtime


def test_plan_propone_renombres_unicos_y_no_toca_lightgbm_v1(maint, mlflow_dir):
    state, ids = mlflow_dir
    plan = maint.make_plan(str(state), EXPERIMENT)
    new = {r["run_id"]: r["new"] for r in plan["renames"]}
    assert new[ids["ref24"]] == "ref-mean-L24-v1"
    assert new[ids["ref72"]] == "ref-mean-L72-v1"
    assert new[ids["conv_a"]] != new[ids["conv_b"]]
    assert ids["v1a"] not in new and ids["v1b"] not in new


def test_plan_lista_v99_con_prefijo_s3(maint, mlflow_dir):
    state, ids = mlflow_dir
    plan = maint.make_plan(str(state), EXPERIMENT)
    deletes = {d["run_id"]: d for d in plan["deletes"]}
    assert set(deletes) == {ids["v99"]}
    assert deletes[ids["v99"]]["s3_prefixes"][0].startswith("_mlflow/mlruns/")
    assert ids["v1a"] not in deletes and ids["v1b"] not in deletes


def test_apply_yes_deja_nombres_unicos_y_borra_v99(maint, mlflow_dir, tmp_path):
    state, ids = mlflow_dir
    plan_path = tmp_path / "plan.json"
    assert maint.main(["plan", "--state-dir", str(state), "--out", str(plan_path)]) == 0
    assert maint.main(["apply", "--state-dir", str(state), "--plan", str(plan_path), "--yes"]) == 0

    names = _names(state)
    assert names[ids["ref24"]] == "ref-mean-L24-v1"
    assert names[ids["ref72"]] == "ref-mean-L72-v1"
    assert names[ids["conv_a"]] != names[ids["conv_b"]]
    assert _lifecycle(state)[ids["v99"]] == "deleted"
    assert _lifecycle(state)[ids["v1a"]] == "active"
    assert names[ids["v1a"]] == names[ids["v1b"]] == "sinca-nielol-lightgbm-v1"


def test_apply_sin_yes_no_escribe(maint, mlflow_dir, tmp_path, capsys):
    state, ids = mlflow_dir
    plan_path = tmp_path / "plan.json"
    maint.main(["plan", "--state-dir", str(state), "--out", str(plan_path)])
    before = _names(state)
    assert maint.main(["apply", "--state-dir", str(state), "--plan", str(plan_path)]) == 0
    assert _names(state) == before
    assert "No se escribió nada" in capsys.readouterr().out


def test_segundo_plan_sin_acciones(maint, mlflow_dir, tmp_path, capsys):
    state, _ = mlflow_dir
    plan_path = tmp_path / "plan.json"
    maint.main(["plan", "--state-dir", str(state), "--out", str(plan_path)])
    maint.main(["apply", "--state-dir", str(state), "--plan", str(plan_path), "--yes"])
    capsys.readouterr()
    assert maint.main(["plan", "--state-dir", str(state), "--out", str(tmp_path / "plan2.json")]) == 0
    assert "Sin acciones" in capsys.readouterr().out


def test_comandos_del_operador_usan_placeholder_de_profile(maint, mlflow_dir, tmp_path, capsys):
    state, _ = mlflow_dir
    plan_path = tmp_path / "plan.json"
    maint.main(["plan", "--state-dir", str(state), "--out", str(plan_path)])
    capsys.readouterr()
    maint.main(["apply", "--state-dir", str(state), "--plan", str(plan_path), "--yes", "--skip-renames"])
    out = capsys.readouterr().out
    assert "--profile <AWS_PROFILE>" in out
    assert "mlflow gc" in out and "--dryrun" in out
    assert json.loads(plan_path.read_text(encoding="utf-8"))["renames"]  # el plan conserva los renombres
