"""Tarea 1.2: publicación desde el job con resguardos (`train.mlflow_log.publish_new_runs`)."""
import numpy as np
import pandas as pd
import pytest

from sint_koopman import T0, FakeS3
from train import mlflow_log as ML


def _run(state, seed, stage="deep-koopman"):
    from operador import registro as G

    tags = {**G.tags_base("gru", "f1", "ae-dmd", seed), "stage": stage}
    return G.log_variante(str(state), f"sinca-zona-koop-gru-f1-aedmd-v1-s{seed}", tags, {"tipo": "koopman"},
                          {"m": 1.0}, {"predictions.parquet": pd.DataFrame({"a": [1]})}, {"koopman.npz": {"K": np.eye(2)}})


def test_conflicto_no_sube_y_copia(estado_mlflow, tmp_path):
    r0 = _run(estado_mlflow, 0)
    s3 = FakeS3()
    out = tmp_path / "conflicto"
    with pytest.raises(ML.PublishConflict):
        ML.publish_new_runs(str(estado_mlflow), "deep-koopman", s3, "2020-01-01T00:00:00+00:00", on_conflict_dir=str(out))
    assert s3.uploads == [] and s3.copies == []
    assert (out / "mlflow.sqlite").exists()
    assert (out / "mlruns" / r0 / "artifacts" / "predictions.parquet").exists()


def test_sin_conflicto_respaldo_y_solo_nuevos(estado_mlflow):
    r0 = _run(estado_mlflow, 0)
    _run(estado_mlflow, 9, stage="otro")
    s3 = FakeS3()
    res = ML.publish_new_runs(str(estado_mlflow), "deep-koopman", s3, T0.isoformat())
    assert res["backup"].startswith("_mlflow/backups/mlflow-") and s3.copies == [res["backup"]]
    assert res["runs"] == [r0]
    assert {k.split("/")[2] for k in s3.uploads if "/mlruns/" in k} == {r0}
    assert res["last_modified"] != T0.isoformat()
    r1 = _run(estado_mlflow, 1)
    s3.uploads.clear()
    res2 = ML.publish_new_runs(str(estado_mlflow), "deep-koopman", s3, res["last_modified"], published=[r0])
    assert res2["runs"] == [r1]
    assert ML.publish_new_runs(str(estado_mlflow), "deep-koopman", s3, res2["last_modified"],
                               published=[r0, r1])["uploaded"] == []
