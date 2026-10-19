"""Baseline de persistencia (naive).

Predicción a `t+24h` = el **último promedio móvil 24h observado** de la serie
(piso de referencia, sin entrenamiento). Se usa el mismo promedio móvil de 24h
que define el target (`icap.moving_average_24h`), calculado en `t`, que es la
información disponible al momento de predecir:

    ŷ(t) = mean(MP(t-23 … t))
    target(t) = mean(MP(t+1 … t+24))

Es la referencia trivial contra la que se mide el LightGBM.
"""
from __future__ import annotations

import logging

import pandas as pd

from features.build import ROLLING_TARGET_HOURS, TARGET_COLUMN
from icap.icap import moving_average_24h

log = logging.getLogger(__name__)


def regularize(series_df: pd.DataFrame, pollutant: str) -> pd.Series:
    """Serie horaria de un contaminante en una estación, en grilla completa."""
    work = (
        series_df[["timestamp", pollutant]]
        .dropna(subset=["timestamp"])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    series = work.set_index("timestamp")[pollutant].astype(float)
    if series.empty:
        return series
    index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    index.name = "timestamp"
    return series.reindex(index)


def persistence_predict_series(series: pd.Series) -> pd.Series:
    """Predicción naive sobre una serie horaria: promedio móvil 24h en `t`."""
    return moving_average_24h(series, min_hours=ROLLING_TARGET_HOURS)


def predict_frame(sinca: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Marco canónico de predicciones `[timestamp, estacion, pollutant, y_true, y_pred]`.

    Se calcula solo para las filas presentes en `features` (las que tienen
    target), reutilizando el `y_true` ya construido por el feature engineering.
    """
    keys = features[["timestamp", "estacion", "pollutant", TARGET_COLUMN]].copy()
    frames: list[pd.DataFrame] = []
    for (station, pollutant), group in keys.groupby(["estacion", "pollutant"]):
        station_df = sinca[sinca["estacion"] == station]
        series = regularize(station_df, pollutant)
        pred = persistence_predict_series(series).rename("y_pred").reset_index()
        pred.columns = ["timestamp", "y_pred"]
        merged = group.merge(pred, on="timestamp", how="left")
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={TARGET_COLUMN: "y_true"})[
        ["timestamp", "estacion", "pollutant", "y_true", "y_pred"]
    ]
