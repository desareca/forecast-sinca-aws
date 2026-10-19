"""Evaluación del baseline (design.md §4/§5).

Produce las métricas que se loguean en MLflow a partir de un marco de
predicciones `[timestamp, estacion, pollutant, y_true, y_pred]`:

- **Loss**: RMSE ponderado por serie, `RMSE_ponderado = sqrt(Σ w_i·(y_i−ŷ_i)² / Σ w_i)`
  con `w_i = 1 + α·max(0, y_i − umbral_regular)` y `umbral_regular` = ancla
  ICAP 100 del contaminante (MP10 150 / MP2.5 50).
- **Diagnóstico por serie**: `mse` y `rmse` sin ponderar.
- **Bloque ICAP oficial** (sobre `ICAP_zona` derivado en post-proceso):
  `recall_alerta`, `recall_preemergencia`, `recall_emergencia`.
- **Bloque percentil propio**: `recall_top10pct`, `recall_top5pct`,
  `pr_auc_top10pct` (no depende de que ICAP tenga suficientes casos por nivel).

`ICAP_zona(t)` se deriva de las predicciones: ICAP por (estación, contaminante)
→ máximo entre contaminantes por estación → máximo entre estaciones.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import STATIONS
from icap.icap import ICAP_ANCHORS, icap_array

log = logging.getLogger(__name__)

# Umbrales oficiales del ICAP (D.S. 12/2011).
ICAP_ALERTA = 100.0
ICAP_PREEMERGENCIA = 200.0
ICAP_EMERGENCIA = 300.0

# Peso continuo del RMSE ponderado (design.md §4).
ALPHA_DEFAULT = 1.0

# Fracciones del bloque percentil propio.
PERCENTILES = (0.10, 0.05)


def _slug(station: str) -> str:
    return STATIONS.get(station, {}).get("slug", station.lower().replace(" ", "_"))


def regular_threshold(pollutant: str) -> float:
    """Umbral `umbral_regular` = ancla ICAP 100 del contaminante (MP10 150 / MP2.5 50)."""
    return float(ICAP_ANCHORS[pollutant][1][0])


def sample_weights(y_true, pollutant: str, alpha: float = ALPHA_DEFAULT, threshold: float | None = None) -> np.ndarray:
    """Pesos `w_i = 1 + α·max(0, y_i − umbral_regular)`, fila a fila."""
    y = np.asarray(y_true, dtype=float)
    thr = regular_threshold(pollutant) if threshold is None else float(threshold)
    return 1.0 + alpha * np.maximum(0.0, y - thr)


def _finite(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    return y[mask], p[mask], mask


def weighted_rmse(y_true, y_pred, weights=None) -> float:
    y, p, mask = _finite(y_true, y_pred)
    if y.size == 0:
        return float("nan")
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)[mask]
    total = w.sum()
    if total == 0:
        return float("nan")
    return float(np.sqrt(np.sum(w * (y - p) ** 2) / total))


def mse(y_true, y_pred) -> float:
    y, p, _ = _finite(y_true, y_pred)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - p) ** 2))


def rmse(y_true, y_pred) -> float:
    value = mse(y_true, y_pred)
    return float(np.sqrt(value)) if np.isfinite(value) else float("nan")


def series_metrics(pred_df: pd.DataFrame, alpha: float = ALPHA_DEFAULT) -> dict[str, float]:
    """`mse`/`rmse`/`weighted_rmse` por serie (nombre `mse_<estacion>_<contaminante>`)."""
    metrics: dict[str, float] = {}
    for (station, pollutant), group in pred_df.groupby(["estacion", "pollutant"]):
        base = f"{_slug(station)}_{pollutant}"
        weights = sample_weights(group["y_true"], pollutant, alpha)
        metrics[f"mse_{base}"] = mse(group["y_true"], group["y_pred"])
        metrics[f"rmse_{base}"] = rmse(group["y_true"], group["y_pred"])
        metrics[f"weighted_rmse_{base}"] = weighted_rmse(group["y_true"], group["y_pred"], weights)
    return metrics


def zone_icap(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `ICAP_zona` (real y predicho) por timestamp desde las 6 series."""
    work = pred_df.dropna(subset=["y_true", "y_pred"]).copy()
    if work.empty:
        return pd.DataFrame(columns=["timestamp", "icap_true", "icap_pred"])
    work["icap_true"] = np.nan
    work["icap_pred"] = np.nan
    for pollutant, group in work.groupby("pollutant"):
        work.loc[group.index, "icap_true"] = icap_array(group["y_true"], pollutant).to_numpy()
        work.loc[group.index, "icap_pred"] = icap_array(group["y_pred"], pollutant).to_numpy()
    per_station = work.groupby(["timestamp", "estacion"])[["icap_true", "icap_pred"]].max()
    zone = per_station.groupby(level="timestamp").max().reset_index()
    return zone


def _recall_at_least(actual: np.ndarray, predicted: np.ndarray, threshold: float) -> tuple[float, int]:
    """Fracción de timesteps con `actual >= threshold` que se predicen `>= threshold`."""
    mask = actual >= threshold
    total = int(mask.sum())
    if total == 0:
        return float("nan"), 0
    return float((predicted[mask] >= threshold).mean()), total


def icap_recall_metrics(zone: pd.DataFrame) -> dict[str, float]:
    if zone.empty:
        return {}
    actual = zone["icap_true"].to_numpy(dtype=float)
    predicted = zone["icap_pred"].to_numpy(dtype=float)
    metrics: dict[str, float] = {}
    for name, threshold in (
        ("alerta", ICAP_ALERTA),
        ("preemergencia", ICAP_PREEMERGENCIA),
        ("emergencia", ICAP_EMERGENCIA),
    ):
        recall, count = _recall_at_least(actual, predicted, threshold)
        metrics[f"recall_{name}"] = recall
        metrics[f"n_{name}"] = count
    return metrics


def _percentile_metrics(actual: np.ndarray, predicted: np.ndarray, top_frac: float) -> tuple[float, float]:
    if actual.size == 0:
        return float("nan"), float("nan")
    actual_thr = float(np.quantile(actual, 1.0 - top_frac))
    actual_set = actual >= actual_thr
    n_positive = int(actual_set.sum())
    if n_positive == 0:
        return float("nan"), float("nan")
    predicted_thr = float(np.quantile(predicted, 1.0 - top_frac))
    predicted_set = predicted >= predicted_thr
    recall = float((actual_set & predicted_set).sum() / n_positive)
    if n_positive == actual.size:
        pr_auc = float("nan")
    else:
        pr_auc = float(average_precision_score(actual_set.astype(int), predicted))
    return recall, pr_auc


def percentile_metrics(zone: pd.DataFrame, percentiles=PERCENTILES) -> dict[str, float]:
    """`recall_topXpct` (+ `pr_auc_top10pct`) sobre los timesteps más críticos."""
    if zone.empty:
        return {}
    actual = zone["icap_true"].to_numpy(dtype=float)
    predicted = zone["icap_pred"].to_numpy(dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    metrics: dict[str, float] = {}
    for frac in percentiles:
        label = f"top{int(round(frac * 100))}pct"
        recall, pr_auc = _percentile_metrics(actual, predicted, frac)
        metrics[f"recall_{label}"] = recall
        if abs(frac - 0.10) < 1e-9:
            metrics[f"pr_auc_{label}"] = pr_auc
    return metrics


def evaluate_predictions(
    pred_df: pd.DataFrame,
    alpha: float = ALPHA_DEFAULT,
    percentiles=PERCENTILES,
) -> dict[str, float]:
    """Bloque completo de métricas para un marco de predicciones."""
    clean = pred_df.dropna(subset=["y_true", "y_pred"])
    metrics: dict[str, float] = {}
    if clean.empty:
        log.warning("Sin predicciones evaluables (todas NaN).")
        return metrics

    weights = pd.Series(np.nan, index=clean.index, dtype=float)
    for pollutant, group in clean.groupby("pollutant"):
        weights.loc[group.index] = sample_weights(group["y_true"], pollutant, alpha)
    metrics["weighted_rmse"] = weighted_rmse(clean["y_true"], clean["y_pred"], weights.to_numpy())
    metrics["rmse"] = rmse(clean["y_true"], clean["y_pred"])
    metrics["mse"] = mse(clean["y_true"], clean["y_pred"])
    metrics["n"] = float(len(clean))

    metrics.update(series_metrics(clean, alpha))

    zone = zone_icap(clean)
    metrics.update(icap_recall_metrics(zone))
    metrics.update(percentile_metrics(zone, percentiles))
    return metrics
