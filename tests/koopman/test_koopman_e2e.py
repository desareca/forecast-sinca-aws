"""Tarea 6.1: CLI `koopman/run.py` de punta a punta (sintético, sin AWS)."""
import json

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from sint_koopman import T0, FakeS3, estado_referencias


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("koop_e2e")
    return (tmp, *estado_referencias(tmp))


def _argv(tmp, state, validated, *extra):
    return ["--fase", "smoke", "--encoder", "gru", "--seed", "0", "--hidden", "8", "--epocas", "2",
            "--max-filas", "300", "--bootstrap", "50", "--bootstrap-modos", "5", "--mlflow-dir", str(state),
            "--local-validated-dir", str(validated), "--model-dir", str(tmp / "model"),
            "--campana", "camp-test", *extra]


def test_smoke_no_upload_registra_run(refs):
    from analysis import load as analysis_load
    from koopman import run as R

    tmp, state, validated, _, _ = refs
    args = R.parse_args(_argv(tmp, state, validated, "--no-upload", "--checkpoint-dir", str(tmp / "ck1")))
    assert args.clave == "smoke-gru-s0"
    out = R.ejecutar(args, descargar=False)
    run = analysis_load.run_by_id(str(state), out["run_ids"][0])
    assert run["run_name"] == "sinca-zona-koop-gru-smoke-aedmd-v1-s0"
    assert run["tags"]["stage"] == "deep-koopman" and run["params"]["arranque"] == "caliente"
    for name in ("encoder.keras", "decoder.keras", "koopman.npz", "koopman_pesos.npz", "latent.parquet",
                 "predictions.parquet", "modos.parquet", "bootstrap_modos.parquet", "coherencia.parquet", "curvas.parquet"):
        assert analysis_load.artifact_path(str(state), run["run_id"], name), name
    assert run["metrics"]["epochs_run"] == 2 and "diff_71" in run["metrics"]


def test_dry_sin_mlflow_y_marcador(refs, tmp_path):
    from koopman import run as R

    tmp, state, validated, _, _ = refs
    s3 = FakeS3()
    n_antes = len(list((state / "mlruns").iterdir()))
    R.ejecutar(R.parse_args(_argv(tmp, state, validated, "--dry", "--clave", "f1-gru-s0")), s3=s3)
    assert "model-artifacts/orquestacion/deep-koopman/camp-test/jobs/f1-gru-s0.json" in s3.objects
    R.ejecutar(R.parse_args(["--fase", "compuerta", "--de", "f1", "--dry", "--rama-forzada", "b1",
                             "--campana", "camp-test"]), s3=s3)
    dec = json.loads(s3.objects["model-artifacts/orquestacion/deep-koopman/camp-test/f1.json"])
    assert dec["rama"] == "b1" and dec["seco"]
    assert len(list((state / "mlruns").iterdir())) == n_antes  # 0 escrituras en MLflow


def test_dry_ckpt_falla_y_retoma(tmp_path):
    from koopman import run as R

    s3 = FakeS3()
    base = ["--fase", "f1", "--dry", "--dry-ckpt", "--campana", "c2", "--clave", "f1-gru-s7",
            "--checkpoint-dir", str(tmp_path / "ck")]
    with pytest.raises(RuntimeError, match="época 2"):
        R.ejecutar(R.parse_args(base + ["--dry-fallar-en", "2"]), s3=s3)
    assert not any(k.endswith("f1-gru-s7.json") for k in s3.objects)
    info = R.ejecutar(R.parse_args(base), s3=s3)
    assert info["retomado_en"] == 2 and info["epochs_run"] == 4


def test_fallo_de_publicacion_no_deja_marcador(refs):
    from koopman import run as R
    from train import mlflow_log as ML

    tmp, state, validated, _, _ = refs
    s3 = FakeS3()
    args = R.parse_args(_argv(tmp, state, validated, "--checkpoint-dir", str(tmp / "ck2")))
    with pytest.raises(ML.PublishConflict):
        R.ejecutar(args, s3=s3, descargar=False)  # sin LastModified de descarga: conflicto
    assert not any("/jobs/" in k for k in s3.objects)
    assert (tmp / "model" / "conflicto" / "mlflow.sqlite").exists()
    (state / ".download.json").write_text(json.dumps({"last_modified": T0.isoformat()}))
    try:
        out = R.ejecutar(R.parse_args(_argv(tmp, state, validated, "--checkpoint-dir", str(tmp / "ck3"))), s3=s3,
                         descargar=False)
    finally:
        (state / ".download.json").unlink()
    m = json.loads(s3.objects["model-artifacts/orquestacion/deep-koopman/camp-test/jobs/smoke-gru-s0.json"])
    assert set(out["run_ids"]) <= set(m["publicado"])


def test_help(capsys):
    from koopman import run as R

    with pytest.raises(SystemExit):
        R.parse_args(["--help"])
    assert "--fase" in capsys.readouterr().out
