"""Test del entry point de punta a punta, sin AWS (tarea 5.6).

Corre `final` (un run por semilla), `reference` y `dmd` sobre datos validados
sintéticos en disco y un MLflow SQLite temporal.
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


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    from ae import run as aerun

    tmp = tmp_path_factory.mktemp("ae_e2e")
    validated = _write_validated(tmp / "validated")
    assert aerun.run(aerun.parse_args(_args(tmp, validated, "--mode", "final", "--arch", "mlp", "--seeds", "0"))) == 0
    assert aerun.run(aerun.parse_args(_args(tmp, validated, "--mode", "reference", "--arch", "mlp"))) == 0
    return tmp, validated


def _runs(tmp: pathlib.Path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow' / 'mlflow.sqlite').as_posix()}")
    exp = mlflow.get_experiment_by_name("test-ae")
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")


def test_final_registra_run_del_ae_con_artefactos(pipeline_run):
    tmp, _ = pipeline_run
    runs = {r.info.run_name: r for r in _runs(tmp)}
    name = "sinca-zona-ae_mlp-v1"
    assert name in runs
    run = runs[name]
    assert run.data.tags["stage"] == "ae-reconstruccion"
    assert run.data.tags["modelo"] == "ae-mlp"
    assert run.data.params["tipo"] == "entrenamiento"
    assert "estacion" not in run.data.params and "pollutant" not in run.data.params
    assert any(k.startswith("recon_rmse_") for k in run.data.metrics)
    assert "gate_beats_pca" in run.data.metrics


def test_final_guarda_modelos_y_tablas(pipeline_run):
    tmp, _ = pipeline_run
    seed_dir = next((tmp / "model").glob("sinca-zona-ae_mlp-v1-s0"))
    assert (seed_dir / "encoder.keras").exists()
    assert (seed_dir / "decoder_mp.keras").exists()
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    assert config["cfg"]["d_aux"] == 2
    assert config["gate"]["gate_pass"] is None  # sin umbrales: solo reporta


def test_reference_registra_mean_y_pca(pipeline_run):
    tmp, _ = pipeline_run
    names = {r.info.run_name for r in _runs(tmp)}
    assert {"ref-mean", "ref-pca-d8"}.issubset(names)
    ref = next(r for r in _runs(tmp) if r.info.run_name == "ref-pca-d8")
    assert ref.data.params["tipo"] == "referencia"


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


def test_dmd_adjunta_operador_y_runs_por_serie(pipeline_run):
    from ae import run as aerun

    tmp, validated = pipeline_run
    ae_run = next(r for r in _runs(tmp) if r.info.run_name == "sinca-zona-ae_mlp-v1")
    ae_dir = next((tmp / "model").glob("sinca-zona-ae_mlp-v1-s0"))
    argv = _args(tmp, validated, "--mode", "dmd", "--arch", "mlp", "--length", "24", "--d", "8",
                 "--ae-dir", str(ae_dir), "--ae-run-id", ae_run.info.run_id)
    assert aerun.run(aerun.parse_args(argv)) == 0
    names = {r.info.run_name for r in _runs(tmp)}
    assert any(n.endswith("ae-mlp-dmd-v1") for n in names)
    refreshed = next(r for r in _runs(tmp) if r.info.run_id == ae_run.info.run_id)
    assert "dmd_latent_rmse_1" in refreshed.data.metrics
    assert "dmd_latent_rmse_24" in refreshed.data.metrics
