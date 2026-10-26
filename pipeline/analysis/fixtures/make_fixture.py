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

    print("Fixture generado en:", FIXTURE_DIR)
    print("Runs:", len(mlflow.search_runs(experiment_names=[mlflow_log.EXPERIMENT_NAME])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
