"""Carga del estado de MLflow para el dashboard de análisis.

Reutiliza `train.mlflow_log` para bajar `mlflow.sqlite` + `mlruns/` a un dir
local y expone helpers para listar runs y leer los diagnósticos persistidos
(`predictions.parquet`, `feature_importance.parquet`, `zone_icap.parquet`).

Fuente de datos:
- `ensure_state(data_dir=...)`: usa un estado local ya descargado (fixture o
  `mlflow_log.download_state` previo) — no toca AWS.
- `ensure_state(data_dir=None)`: descarga el estado real desde S3 a un dir
  temporal.

Degradación: si un run no tiene `predictions.parquet` (corridas viejas), los
helpers devuelven `None` y el dashboard muestra "solo métricas" con aviso.
"""
from __future__ import annotations

import glob
import logging
import os
import pathlib
import sqlite3
import sys
import tempfile

import pandas as pd

# `pipeline/` es la raíz de importación (`config`, `train`, `evaluate`, ...).
_PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from train import mlflow_log  # noqa: E402

log = logging.getLogger(__name__)

EXPERIMENT_NAME = mlflow_log.EXPERIMENT_NAME
DIAGNOSTIC_FILES = ("predictions.parquet", "feature_importance.parquet", "zone_icap.parquet")


# --- Estado local ------------------------------------------------------------


def ensure_state(
    data_dir: str | None = None,
    bucket: str = mlflow_log.MLFLOW_BUCKET_DEFAULT,
    prefix: str = mlflow_log.MLFLOW_PREFIX_DEFAULT,
    client=None,
) -> str:
    """Devuelve un dir local con `mlflow.sqlite` + `mlruns/`.

    Si `data_dir` existe se usa tal cual (fixture o descarga previa); si no, se
    descarga el estado real desde S3 a un dir temporal.
    """
    if data_dir:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"No existe el directorio de estado: {data_dir}")
        return data_dir
    local = tempfile.mkdtemp(prefix="sinca-analysis-")
    downloaded = mlflow_log.download_state(bucket, prefix, local, client)
    log.info("Estado descargado desde s3://%s/%s: %d objetos", bucket, prefix, len(downloaded))
    return local


def _db_path(state_dir: str) -> str:
    return os.path.join(state_dir, mlflow_log.DB_FILENAME)


# --- Runs --------------------------------------------------------------------


def list_runs(state_dir: str, experiment_name: str | None = EXPERIMENT_NAME) -> list[dict]:
    """Lista runs del experimento desde `mlflow.sqlite`.

    Devuelve `[{run_id, run_name, status, experiment_id, params, metrics}]`.
    Usa un cursor por consulta (reusar el mismo cursor dentro de un loop corta
    la iteración externa en `sqlite3`).
    """
    db = _db_path(state_dir)
    if not os.path.exists(db):
        log.warning("No hay mlflow.sqlite en %s", state_dir)
        return []

    con = sqlite3.connect(db)
    try:
        if experiment_name:
            exp = con.execute(
                "select experiment_id from experiments where name = ?", (experiment_name,)
            ).fetchone()
            if not exp:
                log.warning("Experimento %s no encontrado; se listan todos los runs.", experiment_name)
                rows = con.execute(
                    "select run_uuid, name, status, experiment_id from runs order by name"
                ).fetchall()
            else:
                rows = con.execute(
                    "select run_uuid, name, status, experiment_id from runs "
                    "where experiment_id = ? order by name",
                    (exp[0],),
                ).fetchall()
        else:
            rows = con.execute(
                "select run_uuid, name, status, experiment_id from runs order by name"
            ).fetchall()

        runs: list[dict] = []
        for run_id, run_name, status, experiment_id in rows:
            params = dict(
                con.execute("select key, value from params where run_uuid = ?", (run_id,)).fetchall()
            )
            metrics = {
                key: float(value)
                for key, value in con.execute(
                    "select key, value from latest_metrics where run_uuid = ?", (run_id,)
                ).fetchall()
            }
            runs.append(
                {
                    "run_id": run_id,
                    "run_name": run_name,
                    "status": status,
                    "experiment_id": experiment_id,
                    "params": params,
                    "metrics": metrics,
                }
            )
        return runs
    finally:
        con.close()


def _artifact_dir(state_dir: str, run_id: str) -> str | None:
    base = os.path.join(state_dir, mlflow_log.ARTIFACTS_DIRNAME)
    direct = os.path.join(base, run_id, "artifacts")
    if os.path.isdir(direct):
        return direct
    hits = glob.glob(os.path.join(base, "**", run_id, "artifacts"), recursive=True)
    return hits[0] if hits else None


def load_artifacts(state_dir: str, run_id: str) -> dict[str, pd.DataFrame | None]:
    """Lee los 3 parquet de diagnóstico de un run (None si falta)."""
    directory = _artifact_dir(state_dir, run_id)
    out: dict[str, pd.DataFrame | None] = {name: None for name in DIAGNOSTIC_FILES}
    if not directory:
        return out
    for name in DIAGNOSTIC_FILES:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            out[name] = pd.read_parquet(path)
    return out


def has_diagnostics(state_dir: str, run_id: str) -> bool:
    directory = _artifact_dir(state_dir, run_id)
    if not directory:
        return False
    return os.path.exists(os.path.join(directory, "predictions.parquet"))


# --- Derivados ---------------------------------------------------------------


def zone_icap_from_runs(state_dir: str, runs: list[dict], split: str = "test") -> pd.DataFrame:
    """`ICAP_zona` real vs pred del test held-out, agregando las 6 series.

    `zone_icap` (evaluate) necesita todas las series para tomar el máximo por
    timestamp; por eso se concatenan las predicciones de todos los runs en vez
    de usar el `zone_icap.parquet` de un run aislado (que es por serie).
    """
    from evaluate.evaluate import zone_icap

    frames: list[pd.DataFrame] = []
    for run in runs:
        if run["params"].get("tipo") == "persistencia":
            continue
        artifacts = load_artifacts(state_dir, run["run_id"])
        predictions = artifacts["predictions.parquet"]
        if predictions is None or predictions.empty or "split" not in predictions.columns:
            continue
        chunk = predictions[predictions["split"] == split]
        if not chunk.empty:
            frames.append(chunk[["timestamp", "estacion", "pollutant", "y_true", "y_pred"]])

    if not frames:
        return pd.DataFrame(columns=["timestamp", "icap_true", "icap_pred"])
    return zone_icap(pd.concat(frames, ignore_index=True))


def rmse_summary(runs: list[dict]) -> pd.DataFrame:
    """RMSE de test por serie: LightGBM vs persistencia (solo métricas, sin parquet)."""
    persistence = next((run for run in runs if run["params"].get("tipo") == "persistencia"), None)
    persistence_metrics = persistence["metrics"] if persistence else {}

    rows: list[dict] = []
    for run in runs:
        params = run["params"]
        if params.get("tipo") == "persistencia":
            continue
        station = params.get("estacion")
        pollutant = params.get("pollutant")
        if not station or not pollutant:
            continue
        slug = mlflow_log.station_slug(station)
        key = f"{slug}_{pollutant}"
        rows.append(
            {
                "estacion": station,
                "pollutant": pollutant,
                "serie": f"{station} / {pollutant}",
                "rmse_lightgbm": run["metrics"].get("test_rmse"),
                "wrmse_lightgbm": run["metrics"].get("test_weighted_rmse"),
                "rmse_persistence": persistence_metrics.get(f"test_rmse_{key}"),
                "wrmse_persistence": persistence_metrics.get(f"test_weighted_rmse_{key}"),
            }
        )
    return pd.DataFrame(rows)


def series_options(runs: list[dict]) -> list[tuple[str, str]]:
    """Pares `(estacion, pollutant)` de los runs LightGBM."""
    options = {
        (run["params"].get("estacion"), run["params"].get("pollutant"))
        for run in runs
        if run["params"].get("tipo") != "persistencia"
        and run["params"].get("estacion")
        and run["params"].get("pollutant")
    }
    return sorted(options)
