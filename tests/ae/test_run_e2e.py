"""Test del entry point de punta a punta, sin AWS (tareas 5.6 y 5.7).

Corre `final` (un run por semilla), `reference` y `dmd` sobre datos validados
sintéticos en disco y un MLflow SQLite temporal.

Como el AE de prueba (2 épocas) no supera al promedio ni a PCA, la compuerta real
lo rechazaría. Los casos que necesitan el pase DMD fuerzan la **aprobación** con
`monkeypatch` sobre `metrics.gate_decision`; el caso de rechazo usa la compuerta real.
"""
import json
import pathlib

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from conftest import synthetic_validated  # noqa: E402


def _write_validated(root: pathlib.Path) -> pathlib.Path:
    data = synthetic_validated(start="2026-01-01", hours=24 * 280, gaps=True)
    for key, frame in (("sinca", data["sinca"]), ("open_meteo", data["open_meteo"]), ("feriados", data["feriados"])):
        target = root / key
        target.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target / "part-0.parquet", index=False)
    return root


def _args(tmp: pathlib.Path, validated: pathlib.Path, *extra: str) -> list[str]:
    return [
        "--local-validated-dir", str(validated),
        "--mlflow-dir", str(tmp / "mlflow"),
        "--model-dir", str(tmp / "model"),
        "--output-dir", str(tmp / "output"),
        "--experiment", "test-ae",
        "--no-upload",
        "--epochs", "2",
        "--patience", "1",
        "--hidden", "16",
        "--d", "8",
        "--length", "24",
        "--batch-size", "64",
        *extra,
    ]


def _force_gate_pass(monkeypatch) -> None:
    """Fuerza la compuerta relativa a aprobada (solo para tests de mecánica del DMD)."""
    from ae import metrics as aem

    real = aem.gate_decision

    def approved(*args, **kwargs):
        gate = real(*args, **kwargs)
        gate.update(gate_pass=True, relative=True)
        return gate

    monkeypatch.setattr(aem, "gate_decision", approved)


def _runs(tmp: pathlib.Path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow' / 'mlflow.sqlite').as_posix()}")
    exp = mlflow.get_experiment_by_name("test-ae")
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")


def _run_named(tmp: pathlib.Path, name: str):
    return next(r for r in _runs(tmp) if r.info.run_name == name)


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Final con compuerta aprobada (encadena DMD) + referencias, en un MLflow temporal."""
    from ae import run as aerun

    tmp = tmp_path_factory.mktemp("ae_e2e")
    validated = _write_validated(tmp / "validated")
    mp = pytest.MonkeyPatch()
    try:
        _force_gate_pass(mp)
        assert aerun.run(aerun.parse_args(_args(tmp, validated, "--mode", "final", "--arch", "mlp", "--seeds", "0"))) == 0
    finally:
        mp.undo()
    assert aerun.run(aerun.parse_args(_args(tmp, validated, "--mode", "reference", "--arch", "mlp"))) == 0
    return tmp, validated


def test_final_registra_run_del_ae_con_artefactos(pipeline_run):
    tmp, _ = pipeline_run
    names = {r.info.run_name: r for r in _runs(tmp)}
    name = "sinca-zona-ae_mlp-v1"
    assert name in names
    run = names[name]
    assert run.data.tags["stage"] == "ae-reconstruccion"
    assert run.data.tags["modelo"] == "ae-mlp"
    assert run.data.params["tipo"] == "entrenamiento"
    assert "estacion" not in run.data.params and "pollutant" not in run.data.params
    assert any(k.startswith("recon_rmse_") for k in run.data.metrics)
    assert "gate_beats_pca" in run.data.metrics
    assert "gate_beats_mean" in run.data.metrics
    assert run.data.metrics["gate_pass"] == 1.0  # compuerta forzada en este fixture


def test_final_guarda_modelos_y_tablas(pipeline_run):
    tmp, _ = pipeline_run
    seed_dir = next((tmp / "model").glob("sinca-zona-ae_mlp-v1-s0"))
    assert (seed_dir / "encoder.keras").exists()
    assert (seed_dir / "decoder_mp.keras").exists()
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    assert config["cfg"]["d_aux"] == 2
    assert isinstance(config["gate"]["gate_pass"], bool)
    assert isinstance(config["gate"]["numeric"], bool)  # la capa numérica siempre aplica (umbrales por defecto)
    assert config["gate"]["thresholds"]["max_rmse_vs_pca"] == 1.02
    assert "beats_mean" in config["gate"]


def test_final_registra_curva_de_aprendizaje(pipeline_run):
    """5.8: loss/val_loss por epoch (con step), best_epoch y epochs_run consistentes."""
    import mlflow
    from mlflow.tracking import MlflowClient

    tmp, _ = pipeline_run
    run = _run_named(tmp, "sinca-zona-ae_mlp-v1")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow' / 'mlflow.sqlite').as_posix()}")
    client = MlflowClient()
    epochs = int(run.data.metrics["epochs_run"])
    loss = client.get_metric_history(run.info.run_id, "loss")
    assert len(loss) == epochs and [m.step for m in loss] == list(range(1, epochs + 1))
    assert len(client.get_metric_history(run.info.run_id, "val_loss")) == epochs  # hay valid en el fixture
    assert 1 <= run.data.metrics["best_epoch"] <= epochs
    assert "runtime_tensorflow" in run.data.params  # versiones de runtime registradas (10.1)


def test_reference_registra_mean_y_pca(pipeline_run):
    tmp, _ = pipeline_run
    names = {r.info.run_name for r in _runs(tmp)}
    assert {"ref-mean", "ref-pca-d8"}.issubset(names)
    ref = _run_named(tmp, "ref-pca-d8")
    assert ref.data.params["tipo"] == "referencia"


def test_dmd_encadenado_tras_final_aprobado(pipeline_run):
    """5.7 (a): con compuerta aprobada, el pase DMD corre en el mismo job del final."""
    tmp, _ = pipeline_run
    names = {r.info.run_name for r in _runs(tmp)}
    assert any(n.endswith("ae-mlp-dmd-v1") for n in names)
    ae_run = _run_named(tmp, "sinca-zona-ae_mlp-v1")
    assert "dmd_latent_rmse_1" in ae_run.data.metrics
    assert "dmd_latent_rmse_24" in ae_run.data.metrics
    assert "dmd_trunc_latent_rmse_1" in ae_run.data.metrics


def test_dmd_standalone_desde_mlflow_sin_ae_dir(pipeline_run):
    """5.7 (b): `--ae-run-id` descarga encoder/decoder del estado de MLflow (sin --ae-dir)."""
    from ae import run as aerun

    tmp, validated = pipeline_run
    ae_run = _run_named(tmp, "sinca-zona-ae_mlp-v1")
    argv = _args(tmp, validated, "--mode", "dmd", "--ae-run-id", ae_run.info.run_id)
    assert aerun.run(aerun.parse_args(argv)) == 0
    refreshed = _run_named(tmp, "sinca-zona-ae_mlp-v1")
    assert "dmd_latent_rmse_24" in refreshed.data.metrics


def test_dmd_con_ae_dir_para_pruebas(pipeline_run):
    from ae import run as aerun

    tmp, validated = pipeline_run
    ae_run = _run_named(tmp, "sinca-zona-ae_mlp-v1")
    ae_dir = next((tmp / "model").glob("sinca-zona-ae_mlp-v1-s0"))
    argv = _args(tmp, validated, "--mode", "dmd", "--arch", "mlp", "--length", "24", "--d", "8",
                 "--ae-dir", str(ae_dir), "--ae-run-id", ae_run.info.run_id)
    assert aerun.run(aerun.parse_args(argv)) == 0


def test_compuerta_no_aprobada_no_encadena_ni_permite_dmd(tmp_path):
    """Compuerta real (no aprobada con el AE de prueba): sin pase encadenado y `dmd` se rehúsa."""
    from ae import run as aerun

    validated = _write_validated(tmp_path / "validated")
    assert aerun.run(aerun.parse_args(_args(tmp_path, validated, "--mode", "final", "--arch", "mlp", "--seeds", "0"))) == 0
    names = {r.info.run_name for r in _runs(tmp_path)}
    assert not any(n.endswith("ae-mlp-dmd-v1") for n in names)
    ae_run = _run_named(tmp_path, "sinca-zona-ae_mlp-v1")
    with pytest.raises(SystemExit):
        aerun.run(aerun.parse_args(_args(tmp_path, validated, "--mode", "dmd", "--ae-run-id", ae_run.info.run_id)))


def test_search_devuelve_mejor_configuracion_por_arquitectura(tmp_path):
    import pandas as pd

    from ae import data as aed
    from ae import run as aerun
    from conftest import TZ

    validated = synthetic_validated(start="2026-01-01", hours=24 * 120, gaps=False)
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    windows = [
        {
            "train_start": pd.Timestamp("2026-01-01", tz=TZ), "train_end": pd.Timestamp("2026-02-15", tz=TZ),
            "valid_start": pd.Timestamp("2026-02-15", tz=TZ), "valid_end": pd.Timestamp("2026-03-01", tz=TZ),
        },
    ]
    args = aerun.parse_args([
        "--mode", "search", "--archs", "mlp", "--search-trials", "1", "--search-windows", "1",
        "--search-epochs", "1", "--patience", "1", "--seed", "0", "--w-aux", "0.1",
    ])
    best = aerun.run_search(grid, args, windows=windows)
    assert set(best) == {"mlp"}
    assert best["mlp"]["best"] is not None
    assert "config" in best["mlp"]["best"]
    assert len(best["mlp"]["history"]) == 1
    assert "rmse_abs" in best["mlp"]["history"][0] and "score" in best["mlp"]["history"][0]


def test_search_tope_de_tiempo_no_lanza_trials_y_deja_json_valido(tmp_path):
    """5.9 (d)-(e): con tope 0 no se lanza ningún trial y el JSON incremental queda válido."""
    import pandas as pd

    from ae import data as aed
    from ae import run as aerun
    from conftest import TZ

    validated = synthetic_validated(start="2026-01-01", hours=24 * 120, gaps=False)
    grid = aed.build_grid(validated["sinca"], validated["open_meteo"], validated["feriados"])
    windows = [{
        "train_start": pd.Timestamp("2026-01-01", tz=TZ), "train_end": pd.Timestamp("2026-02-15", tz=TZ),
        "valid_start": pd.Timestamp("2026-02-15", tz=TZ), "valid_end": pd.Timestamp("2026-03-01", tz=TZ),
    }]
    args = aerun.parse_args(["--mode", "search", "--archs", "gru", "--search-windows", "1", "--search-time-budget", "0"])
    out = tmp_path / "ae_search.json"
    best = aerun.run_search(grid, args, windows=windows, out_path=str(out))
    assert best["gru"]["best"] is None and best["gru"]["history"] == []
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["gru"]["history"] == [] and not (tmp_path / "ae_search.json.tmp").exists()


def test_search_espacio_por_arquitectura_respeta_d6():
    """5.9 (a)-(b): GRU solo L=24 y hidden ≤ 64; conv1d batch 128; trials por defecto de D6."""
    import numpy as np

    from ae import run as aerun

    rng = np.random.default_rng(0)
    for _ in range(200):
        gru = aerun.sample_choice("gru", rng)
        assert gru["length"] == 24 and gru["hidden"] in (32, 64) and gru["batch_size"] == 128
        conv = aerun.sample_choice("conv1d", rng)
        assert conv["batch_size"] == 128 and conv["length"] in (24, 72)
        assert aerun.sample_choice("linear_decoder", rng)["length"] in (24, 72)
    assert aerun.DEFAULT_SEARCH_TRIALS == {"linear_decoder": 12, "mlp": 12, "conv1d": 6, "gru": 4}
