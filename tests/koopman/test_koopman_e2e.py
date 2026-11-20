"""Tareas 6.1 y 6.6: CLI `koopman/run.py` de punta a punta (sintético, sin AWS)."""
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


def _preparar():
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("preparar_ejecucion_e2e", root / "scripts" / "koopman" / "preparar_ejecucion.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv_job(job: dict) -> list[str]:
    out = []
    for k, v in job["hp"].items():
        if not k.startswith("sagemaker_"):
            out += [f"--{k}", v]
    return out


def test_dry_lista_completa_sin_mlflow(refs):
    """6.6: la lista fija de 48 jobs en modo seco, en orden: marcadores de todos, selección, cierre y test
    escriben su JSON, y 0 escrituras en MLflow."""
    from koopman import run as R

    tmp, state, validated, _, _ = refs
    P = _preparar()
    entrada = P.construir_entrada("camp-seca", "real", "s3://b/c.tar.gz", "img", "rol", "e1")
    s3 = FakeS3()
    n_antes = len(list((state / "mlruns").iterdir()))
    for fase, jobs in entrada["fases"].items():
        for job in jobs:
            R.ejecutar(R.parse_args(_argv_job(job) + ["--dry", "--mlflow-dir", str(state)]), s3=s3)
            assert job["marcador"] in s3.objects, job["clave"]
    for cual, key in entrada["resumenes"].items():
        assert json.loads(s3.objects[key])["seco"], cual
    assert json.loads(s3.objects[entrada["resumenes"]["seleccion"]])["runs"] == {"0": "seco-0", "1": "seco-1", "2": "seco-2"}
    assert len(list((state / "mlruns").iterdir())) == n_antes


def test_cierre_y_test_sin_seleccion_fallan_sin_runs(refs):
    from koopman import run as R

    tmp, state, validated, _, _ = refs
    s3 = FakeS3()
    n_antes = len(list((state / "mlruns").iterdir()))
    for fase in ("cierre", "test"):
        for extra in ([], ["--dry"]):
            with pytest.raises(SystemExit, match="requiere la selección"):
                R.ejecutar(R.parse_args(_argv(tmp, state, validated)[2:] + ["--fase", fase, *extra]), s3=s3,
                           descargar=False)
    assert not s3.objects and len(list((state / "mlruns").iterdir())) == n_antes


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
    assert set(out["run_ids"]) == set(m["publicado"])  # los runs del stage que ya había no se vuelven a subir


def test_help(capsys):
    from koopman import run as R

    with pytest.raises(SystemExit):
        R.parse_args(["--help"])
    assert "--fase" in capsys.readouterr().out


def test_seleccion_cierre_y_test_de_punta_a_punta(refs):
    """6.5/6.6: F1 (2 semillas) → selección (run + JSON) → cierre (lee la selección) → test, sin AWS."""
    from analysis import load as analysis_load
    from koopman import run as R

    tmp, state, validated, _, _ = refs
    base = _argv(tmp, state, validated, "--no-upload", "--campana", "camp-sel")[2:]
    # conv1d (en frío, latente de rango completo): con GRU hidden = 8 < d el arranque en frío de la semilla 1
    # da un LS mal condicionado y pronósticos no finitos (el fixture solo tiene AE v2 de la semilla 0)
    for s in (0, 1):
        R.ejecutar(R.parse_args(base + ["--fase", "f1", "--encoder", "conv1ds", "--seed", str(s), "--epocas", "1",
                                        "--checkpoint-dir", str(tmp / f"ckf1-{s}")]), descargar=False)
    out = R.ejecutar(R.parse_args(base + ["--fase", "seleccion", "--seeds", "0,1"]), descargar=False)
    sel = out["json"]["seleccion"]
    assert sel["configuracion"] == "f1-conv1ds" and set(sel["runs"]) == {"0", "1"} and sel["orden"] == ["f1-conv1ds"]
    run = analysis_load.run_by_id(str(state), out["run_ids"][0])
    assert run["run_name"] == "sinca-zona-koop-seleccion-aedmd-v1" and run["tags"]["fase"] == "seleccion"
    assert analysis_load.artifact_path(str(state), run["run_id"], "seleccion.json")
    ruta = tmp / "seleccion-camp-sel.json"
    ruta.write_text(json.dumps(sel), encoding="utf-8")
    out = R.ejecutar(R.parse_args(base + ["--fase", "cierre", "--seleccion", str(ruta), "--epocas", "1",
                                          "--checkpoint-dir", str(tmp / "ck-cierre")]), descargar=False)
    assert len(out["run_ids"]) == 2 and set(out["json"]["cierre"]["runs"]) == {"0", "1"}
    out = R.ejecutar(R.parse_args(base + ["--fase", "test", "--seleccion", str(ruta)]), descargar=False)
    t = out["json"]["test"]
    assert t["configuracion"] == "f1-conv1ds" and "veredicto_precision" in t and len(out["run_ids"]) == 12  # 6 series × 2
