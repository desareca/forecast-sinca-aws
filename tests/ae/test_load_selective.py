"""Descarga selectiva, listado de runs y clasificación AE (change `ae-dmd-prep`, tarea 2.5).

Usa un cliente S3 falso en memoria construido desde el estado de fixture y un
`mlflow.sqlite` copiado a `tmp_path`. No toca AWS.
"""
import pathlib
import shutil
import sqlite3

import pytest
from botocore.exceptions import ClientError

from analysis import load
from train import mlflow_log

FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "pipeline" / "analysis" / "fixtures" / "state"
PREFIX = "_mlflow"


class FakePaginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        contents = [
            {"Key": key, "Size": len(data)}
            for key, data in sorted(self.objects.items())
            if key.startswith(Prefix)
        ]
        return [{"Contents": contents}] if contents else [{}]


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.downloads: list[str] = []

    def download_file(self, bucket, key, path):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        self.downloads.append(key)
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_bytes(self.objects[key])

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)


def _experiment_of(run_id: str) -> str:
    con = sqlite3.connect(FIXTURE / "mlflow.sqlite")
    try:
        return str(con.execute("select experiment_id from runs where run_uuid = ?", (run_id,)).fetchone()[0])
    finally:
        con.close()


@pytest.fixture
def store():
    """Bucket falso con el estado de fixture; un run tiene layout con experimento."""
    objects = {f"{PREFIX}/mlflow.sqlite": (FIXTURE / "mlflow.sqlite").read_bytes()}
    run_ids = sorted(p.name for p in (FIXTURE / "mlruns").iterdir() if (p / "artifacts").is_dir())
    for path in (FIXTURE / "mlruns").rglob("*"):
        if path.is_file():
            rel = path.relative_to(FIXTURE).as_posix()
            objects[f"{PREFIX}/{rel}"] = path.read_bytes()
    exp_run = run_ids[0]
    exp = _experiment_of(exp_run)
    objects[f"{PREFIX}/mlruns/{exp}/{exp_run}/meta.yaml"] = b"exp-layout"
    objects[f"{PREFIX}/mlruns/{exp}/{exp_run}/artifacts/extra.txt"] = b"x"
    return {"objects": objects, "exp_run": exp_run, "plain_run": run_ids[1], "run_ids": run_ids}


def _local_files(root: pathlib.Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# --- Descarga selectiva ------------------------------------------------------


def test_run_ids_vacio_baja_solo_el_sqlite(store, tmp_path):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    got = load.ensure_state(str(out), bucket="b", prefix=PREFIX, client=client, run_ids=[])
    assert got == str(out)
    assert client.downloads == [f"{PREFIX}/mlflow.sqlite"]
    assert _local_files(out) == {"mlflow.sqlite"}


def test_run_ids_baja_sqlite_y_solo_objetos_de_ese_run_con_experimento(store, tmp_path):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    mlflow_log.download_state("b", PREFIX, str(out), client, run_ids=[store["exp_run"]])
    exp = _experiment_of(store["exp_run"])
    assert all(
        key == f"{PREFIX}/mlflow.sqlite" or key.startswith(f"{PREFIX}/mlruns/{exp}/{store['exp_run']}/")
        for key in client.downloads
    )
    assert f"{PREFIX}/mlruns/{exp}/{store['exp_run']}/meta.yaml" in client.downloads
    # Ningún otro run se bajó.
    assert not any(store["plain_run"] in key for key in client.downloads)


def test_run_ids_usa_layout_sin_experimento_si_no_hay_con_experimento(store, tmp_path):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    mlflow_log.download_state("b", PREFIX, str(out), client, run_ids=[store["plain_run"]])
    run_keys = [k for k in client.downloads if store["plain_run"] in k]
    assert run_keys, "debe bajar los artefactos del run"
    assert all(k.startswith(f"{PREFIX}/mlruns/{store['plain_run']}/") for k in run_keys)
    assert (out / "mlruns" / store["plain_run"] / "artifacts").is_dir()


def test_segunda_llamada_no_rebaja_archivos_del_mismo_tamano(store, tmp_path):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    mlflow_log.download_state("b", PREFIX, str(out), client, run_ids=[store["plain_run"]])
    first = len(client.downloads)
    mlflow_log.download_state("b", PREFIX, str(out), client, run_ids=[store["plain_run"]])
    second_run_downloads = [k for k in client.downloads[first:] if store["plain_run"] in k]
    assert second_run_downloads == []  # solo el sqlite se vuelve a bajar


def test_run_ids_none_baja_todo(store, tmp_path):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    got = mlflow_log.download_state("b", PREFIX, str(out), client)
    assert set(got) == set(store["objects"])


def test_run_sin_objetos_emite_warning_y_no_baja_nada(store, tmp_path, caplog):
    client = FakeS3(store["objects"])
    out = tmp_path / "state"
    with caplog.at_level("WARNING"):
        mlflow_log.download_state("b", PREFIX, str(out), client, run_ids=["no-existe"])
    assert "no-existe" in caplog.text
    assert not any("no-existe" in k for k in client.downloads)


# --- Parseo de --mlflow-runs --------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, None),
        ("", None),
        ("all", None),
        ("none", []),
        ("abc,def", ["abc", "def"]),
    ],
)
def test_parse_mlflow_runs(text, expected):
    assert mlflow_log.parse_mlflow_runs(text) == expected


# --- Listado de runs -----------------------------------------------------------


@pytest.fixture
def state_copy(tmp_path):
    out = tmp_path / "state"
    out.mkdir()
    shutil.copy(FIXTURE / "mlflow.sqlite", out / "mlflow.sqlite")
    return out


def test_list_runs_omite_borrados_y_los_incluye_con_flag(state_copy):
    all_runs = load.list_runs(str(state_copy), experiment_name=None)
    victim = all_runs[0]["run_id"]
    con = sqlite3.connect(state_copy / "mlflow.sqlite")
    con.execute("update runs set lifecycle_stage = 'deleted' where run_uuid = ?", (victim,))
    con.commit()
    con.close()

    active = {r["run_id"] for r in load.list_runs(str(state_copy), experiment_name=None)}
    assert victim not in active
    with_deleted = {
        r["run_id"]: r for r in load.list_runs(str(state_copy), experiment_name=None, include_deleted=True)
    }
    assert victim in with_deleted
    assert with_deleted[victim]["lifecycle_stage"] == "deleted"


def test_list_runs_expone_fechas_y_stage(state_copy):
    run = load.list_runs(str(state_copy), experiment_name=None)[0]
    assert {"start_time", "end_time", "lifecycle_stage"}.issubset(run)


def test_run_by_id_devuelve_forma_de_list_runs(state_copy):
    first = load.list_runs(str(state_copy), experiment_name=None)[0]
    got = load.run_by_id(str(state_copy), first["run_id"])
    assert got is not None
    assert set(got) == set(first)
    assert load.run_by_id(str(state_copy), "no-existe") is None


# --- Clasificación AE -----------------------------------------------------------


def test_is_ae_run_reconoce_conjunto_y_no_el_dmd():
    assert load.is_ae_run({"tags": {"modelo": "ae-conv1d"}, "params": {}})
    assert not load.is_ae_run({"tags": {"modelo": "ae-conv1d-dmd"}, "params": {}})
    assert not load.is_ae_run({"tags": {"modelo": "ae-conv1d"}, "params": {"estacion": "Nielol"}})


def test_is_series_run_excluye_ae_conjunto_y_acepta_dmd_por_serie():
    conjunto = {"tags": {"modelo": "ae-conv1d", "stage": "ae-reconstruccion"}, "params": {}}
    dmd_serie = {
        "tags": {"modelo": "ae-conv1d-dmd", "stage": "ae-dmd-prep"},
        "params": {"estacion": "Nielol", "pollutant": "pm25"},
    }
    assert not load.is_series_run(conjunto)
    assert load.is_series_run(dmd_serie)
