"""Cálculo del ICAP (Índice de Calidad del Aire referido a Partículas).

Fórmula oficial del D.S. 12/2011 (y su equivalente MP10), piecewise-lineal con
3 anclas por contaminante e interpolación lineal entre anclas consecutivas:

| Contaminante | ICAP 0 | ICAP 100 | ICAP 500 |
|--------------|--------|----------|----------|
| MP10         | 0 µg/m³| 150 µg/m³| 330 µg/m³|
| MP2.5        | 0 µg/m³| 50 µg/m³ | 170 µg/m³|

Se aplica sobre el **promedio móvil de 24h** de la concentración horaria. El
`ICAP_zona` se deriva en post-proceso como el máximo entre las 3 estaciones y
el peor contaminante (no se persiste como serie).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Anclas (concentración µg/m³, ICAP) por contaminante.
ICAP_ANCHORS: dict[str, list[tuple[float, float]]] = {
    "pm10": [(0.0, 0.0), (150.0, 100.0), (330.0, 500.0)],
    "pm25": [(0.0, 0.0), (50.0, 100.0), (170.0, 500.0)],
}

# Regla SINCA: el promedio móvil de 24h requiere al menos 75% de datos (18 h).
MIN_HOURS_24H = 18


def icap_from_concentration(concentration: float, pollutant: str) -> float:
    """ICAP para una concentración puntual. `pollutant` ∈ {pm10, pm25}."""
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return float("nan")
    anchors = ICAP_ANCHORS[pollutant]
    concentration = max(0.0, float(concentration))
    if concentration >= anchors[-1][0]:
        return 500.0
    for (c0, i0), (c1, i1) in zip(anchors, anchors[1:]):
        if c0 <= concentration <= c1:
            return i0 + (concentration - c0) * (i1 - i0) / (c1 - c0)
    return float("nan")


def icap_array(concentrations: pd.Series, pollutant: str) -> pd.Series:
    """Versión vectorizada de `icap_from_concentration` (clamp 0-500)."""
    anchors = ICAP_ANCHORS[pollutant]
    xs = np.array([a[0] for a in anchors], dtype=float)
    ys = np.array([a[1] for a in anchors], dtype=float)
    values = pd.to_numeric(concentrations, errors="coerce")
    result = np.interp(values.to_numpy(dtype=float), xs, ys, left=0.0, right=500.0)
    result = np.where(values.isna(), np.nan, result)
    return pd.Series(result, index=concentrations.index)


def moving_average_24h(series: pd.Series, min_hours: int = MIN_HOURS_24H) -> pd.Series:
    """Promedio móvil de 24h (mínimo `min_hours` valores no nulos)."""
    return series.rolling(window=24, min_periods=min_hours).mean()


def add_icap(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `icap_pm10`/`icap_pm25` (sobre el promedio móvil 24h) a un df ancho."""
    out = df.copy()
    for pollutant in ("pm10", "pm25"):
        column = f"icap_{pollutant}"
        if pollutant in out.columns:
            ma = moving_average_24h(pd.to_numeric(out[pollutant], errors="coerce"))
            out[column] = icap_array(ma, pollutant)
        else:
            out[column] = np.nan
    return out


def icap_zona(per_station: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Deriva `ICAP_zona(t)` = max sobre estaciones y contaminantes.

    `per_station`: {estación: df ancho con `timestamp`, `icap_pm10`, `icap_pm25`}.
    Devuelve un DataFrame [timestamp, icap_zona].
    """
    station_series: list[pd.Series] = []
    for df in per_station.values():
        if df is None or df.empty:
            continue
        cols = [c for c in ("icap_pm10", "icap_pm25") if c in df.columns]
        if not cols:
            continue
        worst = df[cols].max(axis=1)
        station_series.append(pd.Series(worst.to_numpy(), index=df["timestamp"], name="icap"))
    if not station_series:
        return pd.DataFrame(columns=["timestamp", "icap_zona"])
    combined = pd.concat(station_series, axis=1)
    zona = combined.max(axis=1)
    out = zona.rename("icap_zona").reset_index()
    out.columns = ["timestamp", "icap_zona"]
    return out.sort_values("timestamp").reset_index(drop=True)
