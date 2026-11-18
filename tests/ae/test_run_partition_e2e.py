"""Test de punta a punta de la partición `ae-dmd` (ae-dmd-prep, tareas 4.5 y 5.4 b).

Mismo patrón y datos que `test_run_e2e.py` (280 días sintéticos, MLflow SQLite temporal, sin AWS).
La compuerta **no** se fuerza: el pase DMD de `ae-dmd` corre igual (D3).
"""
import json
import pathlib

import pandas as pd
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

from conftest import synthetic_validated  # noqa: E402

EXPERIMENT = "test-ae-partition"
AE_NAME = "sinca-zona-ae_mlp-aedmd-v1-s0"


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
        "--experiment", EXPERIMENT,
        "--no-upload",
        "--epochs", "2",
        "--patience", "1",
        "--hidden", "16",
        "--d", "8",
        "--length", "24",
        "--batch-size", "64",
        *extra,
    ]


def _client(tmp: pathlib.Path):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(f"sqlite:///{(tmp / 'mlflow' / 'mlflow.sqlite').as_posix()}")
    return MlflowClient()


def _runs(tmp: pathlib.Path):
    import mlflow

    client = _client(tmp)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    return client, mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")


def _read_artifact(tmp: pathlib.Path, run_id: str, name: str) -> pd.DataFrame:
    client = _client(tmp)
    local = client.download_artifacts(run_id, name, dst_path=str(tmp / "dl"))
    return pd.read_parquet(local)


@pytest.fixture(scope="module")
def partition_run(tmp_path_factory):
    """`final --partition ae-dmd` con una semilla; encadena el pase DMD sin mirar la compuerta."""
    from ae import run as aerun

    tmp = tmp_path_factory.mktemp("ae_partition_e2e")
    validated = _write_validated(tmp / "validated")
    argv = _args(tmp, validated, "--mode", "final", "--arch", "mlp", "--partition", "ae-dmd", "--seeds", "0")
    assert aerun.run(aerun.parse_args(argv)) == 0
    return tmp, validated


def _ae_run(tmp: pathlib.Path):
    _, runs = _runs(tmp)
    return next(r for r in runs if r.info.run_name == AE_NAME)


def test_run_ae_con_nombre_y_tags_de_particion(partition_run):
    tmp, _ = partition_run
    run = _ae_run(tmp)
    assert run.data.tags["particion"] == "ae-dmd"
    assert run.data.tags["stage"] == "ae-dmd-prep"
    assert run.data.tags["seed"] == "0"
    assert run.data.params["particion"] == "ae-dmd"
    assert "partition_bounds" in run.data.params
    assert "gate_pass" in run.data.metrics


def test_latent_tiene_los_cuatro_tramos(partition_run):
    tmp, _ = partition_run
    run = _ae_run(tmp)
    latent = _read_artifact(tmp, run.info.run_id, "latent.parquet")
    assert set(latent["split"].unique()) == {"train", "es", "sel", "test"}


def test_no_hay_metricas_de_test(partition_run):
    tmp, _ = partition_run
    run = _ae_run(tmp)
    assert not any(key.startswith("recon_") for key in run.data.metrics)
    assert not any("_test" in key for key in run.data.metrics)
    assert "sel_recon_rmse_mean" in run.data.metrics
    assert "gate_pass" in run.data.metrics


def test_reconstruccion_sin_filas_de_test(partition_run):
    tmp, _ = partition_run
    run = _ae_run(tmp)
    recon = _read_artifact(tmp, run.info.run_id, "reconstruction.parquet")
    assert set(recon["split"].unique()) <= {"es", "sel"}
    assert "test" not in set(recon["split"].unique())


def test_pase_dmd_encadenado_sobre_sel(partition_run):
    """5.4 (b): 6 runs por serie, pronóstico solo en sel (embargo incluido) y artefactos del AE."""
    tmp, _ = partition_run
    client, runs = _runs(tmp)
    dmd_runs = [r for r in runs if r.info.run_name.startswith("sinca-") and r.info.run_name.endswith("ae-mlp-dmd-aedmd-v1-s0")]
    assert len(dmd_runs) == 6
    names = [r.info.run_name for r in dmd_runs]
    assert len(set(names)) == 6  # 5.7: un nombre por serie, con contaminante
    assert all(("-pm25-" in n) or ("-pm10-" in n) for n in names)
    for run in dmd_runs:
        preds = _read_artifact(tmp, run.info.run_id, "predictions.parquet")
        assert set(preds["split"].unique()) == {"sel"}
        stamps = pd.to_datetime(preds["timestamp"])
        if stamps.dt.tz is not None:
            stamps = stamps.dt.tz_convert("America/Santiago").dt.tz_localize(None)
        assert stamps.max() < pd.Timestamp("2026-06-30 00:00")

    ae_run = _ae_run(tmp)
    assert "dmd_sel_latent_rmse_24" in ae_run.data.metrics
    assert "pydmd_version" in ae_run.data.tags
    artifacts = {a.path for a in client.list_artifacts(ae_run.info.run_id)}
    assert "dmd_operator.npz" in artifacts


def test_dmd_standalone_de_particion_sin_ae_dir(partition_run):
    """`--mode dmd --partition ae-dmd` usa el AE de MLflow (sin --ae-dir) y no exige la compuerta."""
    from ae import run as aerun

    tmp, validated = partition_run
    ae_run = _ae_run(tmp)
    argv = _args(tmp, validated, "--mode", "dmd", "--partition", "ae-dmd", "--mlflow-runs", "none",
                 "--ae-run-id", ae_run.info.run_id)
    assert aerun.run(aerun.parse_args(argv)) == 0


def test_normalizador_solo_usa_train(tmp_path, monkeypatch):
    """4.5 (e): la normalización se ajusta con filas de `train` únicamente."""
    from ae import data as aed
    from ae import run as aerun
    from features import partition as pt

    validated = _write_validated(tmp_path / "validated")
    masks: list = []
    real = aed.fit_normalizer

    def spy(grid, train_mask):
        masks.append(train_mask.copy())
        return real(grid, train_mask)

    monkeypatch.setattr(aed, "fit_normalizer", spy)
    argv = _args(tmp_path, validated, "--mode", "final", "--arch", "mlp", "--partition", "ae-dmd",
                 "--seeds", "0", "--dmd-after-final", "false")
    assert aerun.run(aerun.parse_args(argv)) == 0
    assert len(masks) == 1

    data = synthetic_validated(start="2026-01-01", hours=24 * 280, gaps=True)
    grid = aed.build_grid(data["sinca"], data["open_meteo"], data["feriados"])
    spans = aed.partition_spans(grid.index, pt.get_partition("ae-dmd"))
    mask = masks[0]
    lo, hi = spans["train"]
    assert mask[:lo].sum() == 0 and mask[hi:].sum() == 0
    assert mask.sum() == hi - lo  # toda la ventana de train, nada de es/sel/test
    assert not mask[spans["es"][0]:].any()


def test_early_stopping_off_entrena_epochs_completas(tmp_path):
    """13.1 (D11): con --early-stopping off se corren todas las épocas y el param queda registrado."""
    from ae import run as aerun

    validated = _write_validated(tmp_path / "validated")
    argv = _args(tmp_path, validated, "--mode", "final", "--arch", "mlp", "--partition", "ae-dmd",
                 "--seeds", "0", "--dmd-after-final", "false", "--early-stopping", "off", "--epochs", "3")
    assert aerun.run(aerun.parse_args(argv)) == 0
    run = _ae_run(tmp_path)
    assert run.data.metrics["epochs_run"] == 3
    assert run.data.params["early_stopping"] == "off"
    assert "es_recon_rmse_mean" in run.data.metrics  # `es` sigue como diagnóstico


def test_config_guarda_particion(partition_run):
    tmp, _ = partition_run
    seed_dir = tmp / "model" / AE_NAME
    config = json.loads((seed_dir / "config.json").read_text(encoding="utf-8"))
    assert json.loads(config["partition"])["name"] == "ae-dmd"
