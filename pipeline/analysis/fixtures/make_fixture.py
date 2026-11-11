"""Genera un estado MLflow sintético (fixture) para iterar el dashboard sin AWS.

Crea `pipeline/analysis/fixtures/state/` con `mlflow.sqlite` + `mlruns/`, igual
que el estado real de `s3://sinca-mlflow/_mlflow/`, pero con datos señuelo:
6 runs LightGBM (3 estaciones x pm10/pm25) + 1 run de persistencia, cada uno con
`predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet`.

Uso:
    python pipeline/analysis/fixtures/make_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_PIPELINE = _HERE.parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from config import STATIONS  # noqa: E402
from evaluate.evaluate import evaluate_predictions, zone_icap  # noqa: E402
from features.build import FEATURE_COLUMNS  # noqa: E402
from train import mlflow_log  # noqa: E402

FIXTURE_DIR = _HERE / "state"
TEST_START = "2026-07-01"
VALID_START = "2026-05-01"
TEST_HOURS = 24 * 30
VALID_HOURS = 24 * 14
TZ = "America/Santiago"


def _synthetic(station: str, pollutant: str, start: str, hours: int, rng, noise_scale: float = 8.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=hours, freq="h", tz=TZ)
    index = np.arange(hours)
    base = 60.0 if pollutant == "pm10" else 24.0
    daily = 10.0 * np.sin(2 * np.pi * index / 24)
    weekly = 5.0 * np.sin(2 * np.pi * index / (24 * 7))
    seasonal = 8.0 * np.sin(2 * np.pi * index / (24 * 365))
    noise = rng.normal(0.0, 6.0, hours)
    y_true = np.clip(base + daily + weekly + seasonal + noise, 1.0, None)
    y_pred = np.clip(y_true + rng.normal(0.0, noise_scale, hours) + 2.0, 1.0, None)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "estacion": station,
            "pollutant": pollutant,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def _importance(station: str, pollutant: str, rng) -> pd.DataFrame:
    gains = rng.lognormal(mean=12.0, sigma=1.2, size=len(FEATURE_COLUMNS))
    return pd.DataFrame(
        {
            "estacion": station,
            "pollutant": pollutant,
            "feature": list(FEATURE_COLUMNS),
            "gain": gains,
        }
    )


def _search_history(rng, base_score: float) -> tuple[list[dict], dict]:
    """20 candidatos señuelo del search; el ganador es el de menor score."""
    space = {
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 0.1, 1.0],
    }
    history = []
    for trial in range(20):
        params = {
            name: values[int(rng.integers(len(values)))] for name, values in space.items()
        }
        history.append(
            {
                "trial": trial,
                "params": params,
                "score": float(base_score + rng.normal(0.0, 1.4)),
            }
        )
    best = min(history, key=lambda item: item["score"])
    best["score"] = float(best["score"] - 1.0)
    return history, best["params"]


def _log_run(
    name: str,
    params: dict,
    metrics: dict,
    frames: dict[str, pd.DataFrame],
    json_artifacts: dict[str, dict] | None = None,
) -> None:
    import mlflow

    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"serie": f"{params.get('estacion')}|{params.get('pollutant')}", "tipo": params.get("tipo")})
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if np.isfinite(value)})
        with tempfile.TemporaryDirectory(prefix="sinca-fixture-") as tmp:
            for filename, frame in frames.items():
                frame.to_parquet(pathlib.Path(tmp) / filename, index=False)
            for filename, payload in (json_artifacts or {}).items():
                (pathlib.Path(tmp) / filename).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            mlflow.log_artifacts(tmp)


def _add_ae_runs() -> None:
    """Runs de autoencoder (change `ae-reconstruccion`): AE conjunto, referencia y DMD por serie.

    Usa su propio generador (semilla 11) para no alterar los runs LightGBM de arriba.
    """
    import mlflow

    rng = np.random.default_rng(11)
    arch, version, d, d_aux = "gru", 1, 16, 4
    name = f"sinca-zona-ae_{arch}-v{version}"
    ts = pd.date_range(TEST_START, periods=TEST_HOURS, freq="h", tz=TZ)

    recon_frames, diag_rows = [], []
    for station in STATIONS:
        for pollutant in ("pm10", "pm25"):
            base = 60.0 if pollutant == "pm10" else 24.0
            y_true = np.clip(base + 10 * np.sin(np.arange(TEST_HOURS) / 24 * 2 * np.pi) + rng.normal(0, 6, TEST_HOURS), 1, None)
            y_recon = np.clip(0.9 * y_true + 0.1 * y_true.mean() + rng.normal(0, 2, TEST_HOURS), 0, None)
            serie = f"{station}|{pollutant}"
            recon_frames.append(pd.DataFrame({"timestamp": ts, "serie": serie, "y_true": y_true, "y_recon": y_recon, "split": "test"}))
            diag = {"var_ratio": float(np.std(y_recon) / np.std(y_true)), "slope": 0.92, "rmse": float(np.sqrt(np.mean((y_true - y_recon) ** 2))),
                    "rmse_lt50": 4.0, "bias_lt50": 0.5, "rmse_50_100": 6.0, "bias_50_100": -1.0, "rmse_gt100": 12.0, "bias_gt100": -6.0}
            diag_rows.extend({"split": "test", "serie": serie, "metric": k, "value": v} for k, v in diag.items())

    z = rng.normal(0, 1, (TEST_HOURS, d))
    latent = pd.DataFrame(z, columns=[f"z_mp_{i}" for i in range(d - d_aux)] + [f"z_aux_{j}" for j in range(d_aux)])
    latent.insert(0, "timestamp", ts)
    latent.insert(1, "split", "test")

    metrics = {
        "recon_rmse_mean_test": 4.8, "recon_rmse_mean_valid": 5.1,
        "gate_beats_pca": 1.0, "ablation_zaux_delta_rmse_test": 0.37,
        "latent_eff_dim_pr": 9.1, "latent_dead_dims": 0.0,
        "dmd_latent_rmse_1": 0.12, "dmd_latent_rmse_24": 0.61,
        "dmd_valid_latent_rmse_1": 0.13, "dmd_valid_latent_rmse_24": 0.66,
        "dmd_trunc_latent_rmse_1": 0.19, "dmd_trunc_latent_rmse_24": 0.88,
    }
    params = {"arch": arch, "L": 72, "d": d, "d_aux": d_aux, "hidden": 64, "dropout": 0.1, "lr": 0.001,
              "w_mp": 1.0, "w_aux": 0.1, "aux_input": True, "seed": 0, "version": version, "tipo": "entrenamiento",
              "gate_thresholds": "null"}
    tags = {"modelo": f"ae-{arch}", "stage": "ae-reconstruccion"}

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory(prefix="sinca-fixture-ae-") as tmp:
            tmp_path = pathlib.Path(tmp)
            pd.concat(recon_frames, ignore_index=True).to_parquet(tmp_path / "reconstruction.parquet", index=False)
            latent.to_parquet(tmp_path / "latent.parquet", index=False)
            pd.DataFrame(diag_rows).to_parquet(tmp_path / "diagnostics.parquet", index=False)
            mlflow.log_artifacts(tmp)
        ae_run_id = run.info.run_id

    # Referencia PCA(d) del mismo tamaño (sin pronóstico): `tipo=referencia`.
    with mlflow.start_run(run_name=f"ref-pca-d{d}"):
        mlflow.set_tags({"modelo": f"ref-pca-d{d}", "tipo": "referencia", "stage": "ae-reconstruccion"})
        mlflow.log_params({"tipo": "referencia", "L": 72, "d": d, "version": version})
        mlflow.log_metrics({"recon_rmse_mean_test": 5.4, "recon_rmse_mean_valid": 5.6})

    # Pase DMD por serie: `ae-<arq>-dmd`, mismo contrato que LightGBM, tag `ae_run_id`.
    for station in STATIONS:
        for pollutant in ("pm10", "pm25"):
            test = _synthetic(station, pollutant, TEST_START, TEST_HOURS, rng, 9.0)
            test["split"] = "test"
            name_dmd = mlflow_log.run_name(station, f"ae-{arch}-dmd", version)
            with mlflow.start_run(run_name=name_dmd):
                mlflow.set_tags({"serie": f"{station}|{pollutant}", "modelo": f"ae-{arch}-dmd",
                                 "stage": "ae-dmd", "tipo": "entrenamiento", "ae_run_id": ae_run_id})
                mlflow.log_params({"estacion": station, "pollutant": pollutant, "version": version,
                                   "tipo": "entrenamiento", "rank": "full"})
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in evaluate_predictions(test).items()
                                    if np.isfinite(v)})
                with tempfile.TemporaryDirectory(prefix="sinca-fixture-dmd-") as tmp:
                    test.to_parquet(pathlib.Path(tmp) / "predictions.parquet", index=False)
                    zone_icap(test).to_parquet(pathlib.Path(tmp) / "zone_icap.parquet", index=False)
                    pd.DataFrame({"window_index": [], "rmse": []}).astype({"window_index": int, "rmse": float}).to_parquet(
                        pathlib.Path(tmp) / "rolling_scores.parquet", index=False)
                    mlflow.log_artifacts(tmp)


def main() -> int:
    import mlflow

    rng = np.random.default_rng(7)
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)

    mlflow_log.configure(str(FIXTURE_DIR), experiment=mlflow_log.EXPERIMENT_NAME)

    persistence_metrics: dict[str, float] = {}
    for station in STATIONS:
        for pollutant in ("pm10", "pm25"):
            # Varias familias/versiones por serie para poder comparar en el dashboard.
            for model, version, noise_scale in (
                ("lightgbm", 1, 8.0),
                ("lightgbm", 2, 5.0),
                ("dmd", 1, 11.0),
            ):
                test = _synthetic(station, pollutant, TEST_START, TEST_HOURS, rng, noise_scale)
                valid = _synthetic(station, pollutant, VALID_START, VALID_HOURS, rng, noise_scale)
                test["split"] = "test"
                valid["split"] = "valid"
                predictions = pd.concat([valid, test], ignore_index=True)

                metrics = {f"test_{key}": value for key, value in evaluate_predictions(test).items()}
                metrics["train_loss"] = float(rng.uniform(1.0, 8.0))
                rolling_scores = np.clip(rng.normal(18.0, 5.0, 42), 1.0, None)
                metrics["rolling_weighted_rmse_mean"] = float(rolling_scores.mean())
                metrics["rolling_weighted_rmse_std"] = float(rolling_scores.std())

                history, best_params = _search_history(rng, float(rolling_scores.mean()))

                _log_run(
                    mlflow_log.run_name(station, model, version),
                    {
                        "estacion": station,
                        "pollutant": pollutant,
                        "modelo": model,
                        "version": version,
                        "tipo": "entrenamiento",
                        "learning_rate": 0.1 if version == 1 else 0.05,
                        "num_leaves": 63,
                    },
                    metrics,
                    {
                        "predictions.parquet": predictions,
                        "feature_importance.parquet": _importance(station, pollutant, rng),
                        "zone_icap.parquet": zone_icap(test),
                        "rolling_scores.parquet": pd.DataFrame(
                            {
                                "window_index": np.arange(len(rolling_scores)),
                                "rmse": rolling_scores,
                            }
                        ),
                    },
                    json_artifacts={
                        f"{mlflow_log.run_name(station, model, version)}.json": {
                            "estacion": station,
                            "pollutant": pollutant,
                            "model": model,
                            "version": version,
                            "best_params": best_params,
                            "search_history": history,
                        }
                    },
                )

                if model == "lightgbm" and version == 1:
                    # Persistencia señuelo: predicción = y_true desplazado 1h (peor RMSE).
                    persist = test.copy()
                    persist["y_pred"] = persist["y_true"].shift(1).bfill() + rng.normal(
                        0.0, 3.0, len(persist)
                    )
                    for key, value in evaluate_predictions(persist).items():
                        persistence_metrics[f"test_{key}"] = value

    _log_run(
        "sinca-persistence-baseline",
        {"tipo": "persistencia", "alpha": 1.0},
        persistence_metrics,
        {},
    )

    _add_ae_runs()

    print("Fixture generado en:", FIXTURE_DIR)
    print("Runs:", len(mlflow.search_runs(experiment_names=[mlflow_log.EXPERIMENT_NAME])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
