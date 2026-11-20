"""Tarea 1.7: publicación del estado local con S3 falso."""
import datetime as dt
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from operador import registro as G

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("publish_state", ROOT / "scripts" / "mlflow" / "publish_state.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

T0 = dt.datetime(2026, 11, 18, 4, 14, 52, tzinfo=dt.timezone.utc)


class FakeS3:
    def __init__(self):
        self.last_modified = T0
        self.uploads: list[str] = []
        self.copies: list[str] = []

    def head_object(self, Bucket, Key):
        return {"LastModified": self.last_modified}

    def copy_object(self, Bucket, Key, CopySource):
        self.copies.append(Key)

    def upload_file(self, filename, bucket, key):
        self.uploads.append(key)
        if key.endswith("mlflow.sqlite"):
            self.last_modified = self.last_modified + dt.timedelta(minutes=5)


def _run(state, seed):
    return G.log_variante(str(state), G.run_name("hankel", "e11a", "aedmd", 1, seed), G.tags_base("hankel", "1.1", "ae-dmd", seed),
                          {"tipo": "operador"}, {"m": 1.0}, {"predictions.parquet": pd.DataFrame({"a": [1]})},
                          {"operator.npz": {"A": np.eye(2)}})


def _meta(state, value):
    (pathlib.Path(state) / ".download.json").write_text(json.dumps({"last_modified": value, "published_runs": []}))


def test_last_modified_distinto_no_sube(estado_mlflow):
    _run(estado_mlflow, 0)
    _meta(estado_mlflow, "2020-01-01T00:00:00+00:00")
    s3 = FakeS3()
    with pytest.raises(SystemExit):
        P.apply(str(estado_mlflow), s3)
    assert s3.uploads == [] and s3.copies == []


def test_apply_actualiza_meta_y_sube_solo_nuevos(estado_mlflow):
    r0 = _run(estado_mlflow, 0)
    _meta(estado_mlflow, T0.isoformat())
    s3 = FakeS3()
    up = P.apply(str(estado_mlflow), s3, now=dt.datetime(2026, 11, 19, 10, 30))
    assert up[0] == "_mlflow/backups/mlflow-20261119-1030.sqlite"
    assert "_mlflow/mlflow.sqlite" in s3.uploads
    assert any(f"_mlflow/mlruns/{r0}/artifacts/predictions.parquet" == k for k in s3.uploads)
    meta = json.loads((estado_mlflow / ".download.json").read_text())
    assert meta["last_modified"] == s3.last_modified.isoformat() != T0.isoformat()
    assert meta["published_runs"] == [r0]
    r1 = _run(estado_mlflow, 1)
    s3.uploads.clear()
    P.apply(str(estado_mlflow), s3)
    runs_subidos = {k.split("/")[2] for k in s3.uploads if "/mlruns/" in k}
    assert runs_subidos == {r1}
    s3.uploads.clear()
    assert P.apply(str(estado_mlflow), s3) == [] and s3.uploads == []  # sin runs nuevos no escribe


def test_plan_falla_sin_artefactos(estado_mlflow):
    run_id = _run(estado_mlflow, 0)
    for f in (estado_mlflow / "mlruns" / run_id / "artifacts").iterdir():
        f.unlink()
    with pytest.raises(SystemExit, match="sin artefactos"):
        P.plan(str(estado_mlflow))


def test_cli_sin_profile_ni_yes(estado_mlflow):
    with pytest.raises(SystemExit):
        P.main(["plan", "--state-dir", str(estado_mlflow)])
    _run(estado_mlflow, 0)
    assert P.main(["apply", "--state-dir", str(estado_mlflow), "--profile", "x"]) == 1


def test_descartados_no_se_publican(estado_mlflow):
    from mlflow.tracking import MlflowClient

    r0 = _run(estado_mlflow, 0)
    r1 = _run(estado_mlflow, 1)
    MlflowClient().set_tag(r0, "descartado", "regla-vieja")
    assert [r for r, _, _ in P.plan(str(estado_mlflow))] == [r1]
