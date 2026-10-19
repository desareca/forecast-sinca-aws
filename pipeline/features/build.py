"""Feature engineering y particionado del baseline (Fase 3).

Lee los datasets validados de `sinca-data/validated/` (SINCA por estación,
Open-Meteo y feriados), construye el dataset supervisado por serie
(MP2.5/MP10 x 3 estaciones) y define el particionado (ventana deslizante de
tamaño fijo + test final held-out) descrito en `design.md` §2/§3.

Features por serie:

- Lags de la serie propia: `t-1h`, `t-24h`, `t-48h`, `t-72h`, `t-168h`.
- Meteorología (misma estación) con lag `t-24h`: temp, humedad, presión, viento
  (dir/vel) y radiación.
- Altura de capa límite (Open-Meteo): valor actual + lag `t-24h`.
- Feriados (flag binario), día de semana cyclic (sin/cos), ventana GEC (flag).
- Fourier (una armónica sin/cos por periodicidad): diaria (24h), semanal
  (168h) y anual (365.25 días).

Target por serie: promedio móvil de 24h a `t+24h`, `mean(MP(t+1 … t+24h))`.

Las series se reindexan a una grilla horaria completa antes de aplicar los
lags, de modo que `t-24h` sea literalmente 24 horas antes (y no 24 filas antes
si hay huecos en el histórico). LightGBM tolera los `NaN` resultantes.
"""
from __future__ import annotations

import glob
import io
import logging
import os
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import STATIONS, TARGET_VARS, TIMEZONE
from persist.s3 import get_client

log = logging.getLogger(__name__)

_TZ = ZoneInfo(TIMEZONE)

# --- Feature engineering -----------------------------------------------------

# Lags de la serie propia (horas).
LAGS_HOURS: tuple[int, ...] = (1, 24, 48, 72, 168)

# Meteorología (misma estación) con lag 24h.
MET_LAG_HOURS = 24
MET_COLUMNS: tuple[str, ...] = ("temp", "rhum", "pres", "wdir", "wspd", "glob")

# Altura de capa límite: valor actual + lag 24h.
BLH_LAG_HOURS = 24

# Target: promedio móvil de 24h a t+24h.
TARGET_HORIZON_HOURS = 24
ROLLING_TARGET_HOURS = 24

# Fourier: una armónica (sin/cos) por periodicidad, en horas.
FOURIER_PERIODS_HOURS: dict[str, float] = {
    "daily": 24.0,
    "weekly": 168.0,
    "annual": 365.25 * 24.0,
}

# Ventana GEC (gestión de episodios críticos): 1 abril - 15 septiembre.
# (mismo valor que `config.GEC_START`/`GEC_END`, duplicado acá para no acoplar
# el módulo de features a la constante de persistencia del pipeline de datos).
GEC_START = (4, 1)
GEC_END = (9, 15)

IDENTITY_COLUMNS: tuple[str, ...] = ("estacion", "pollutant", "timestamp")
TARGET_COLUMN = "target"

FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(f"lag_{h}h" for h in LAGS_HOURS)
    + tuple(f"met_{c}_lag24" for c in MET_COLUMNS)
    + ("blh", "blh_lag24")
    + ("is_holiday", "gec_window", "dow_sin", "dow_cos")
    + tuple(
        f"fourier_{name}_{fn}"
        for name in FOURIER_PERIODS_HOURS
        for fn in ("sin", "cos")
    )
)

# --- Particionado (design.md §3) --------------------------------------------

# Ventana deslizante sobre 2021-01-01 -> 2026-06-30.
ROLLING_START = date(2021, 1, 1)
ROLLING_END = date(2026, 6, 30)
TRAIN_YEARS = 2
VALID_MONTHS = 1
STEP_MONTHS = 1

# Test final held-out: 3 meses, nunca tocado en tuning.
TEST_START = date(2026, 7, 1)
TEST_END = date(2026, 10, 4)


# --- Lectura de datos validados ---------------------------------------------


def _iter_parquet_keys(client, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                yield key


def read_parquet_prefix(client, bucket: str, prefix: str) -> pd.DataFrame:
    """Lee (y concatena) todos los Parquet bajo `s3://bucket/prefix`."""
    frames: list[pd.DataFrame] = []
    for key in _iter_parquet_keys(client, bucket, prefix):
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        frames.append(pd.read_parquet(io.BytesIO(body)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_parquet_dir(path: str) -> pd.DataFrame:
    """Lee (y concatena) todos los Parquet bajo un directorio local."""
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True)
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_validated(
    bucket: str,
    layer: str = "validated",
    client=None,
    local_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Carga los tres datasets validados: SINCA, Open-Meteo y feriados.

    `local_dir` (opcional) lee de disco en vez de S3 (útil para pruebas locales).
    """
    if local_dir:
        return {
            "sinca": read_parquet_dir(os.path.join(local_dir, "sinca")),
            "open_meteo": read_parquet_dir(os.path.join(local_dir, "open_meteo")),
            "feriados": read_parquet_dir(os.path.join(local_dir, "feriados")),
        }
    client = client or get_client()
    return {
        "sinca": read_parquet_prefix(client, bucket, f"{layer}/sinca/"),
        "open_meteo": read_parquet_prefix(client, bucket, f"{layer}/open_meteo/"),
        "feriados": read_parquet_prefix(client, bucket, f"{layer}/feriados/"),
    }


# --- Construcción de features ------------------------------------------------


def _holiday_dates(feriados: pd.DataFrame | None) -> set:
    if feriados is None or feriados.empty or "fecha" not in feriados.columns:
        return set()
    return set(pd.to_datetime(feriados["fecha"]).dt.date)


def _blh_series(open_meteo: pd.DataFrame | None) -> pd.Series:
    if (
        open_meteo is None
        or open_meteo.empty
        or "boundary_layer_height" not in open_meteo.columns
    ):
        return pd.Series(dtype=float)
    series = open_meteo.set_index("timestamp")["boundary_layer_height"].astype(float)
    series = series[~series.index.duplicated()].sort_index()
    return series


def _to_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Horas desde epoch (UTC) para una `DatetimeIndex` tz-aware o naive."""
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.astype("int64").to_numpy() / 3.6e12


def _station_base(
    station_df: pd.DataFrame,
    blh: pd.Series,
    holiday_dates: set,
) -> pd.DataFrame:
    """Features independientes del contaminante, indexado por timestamp horario."""
    work = station_df[["timestamp", *MET_COLUMNS]].copy()
    work = work.dropna(subset=["timestamp"]).drop_duplicates("timestamp")
    work = work.sort_values("timestamp").set_index("timestamp")

    index = pd.date_range(work.index.min(), work.index.max(), freq="h")
    index.name = "timestamp"
    work = work.reindex(index)

    base = pd.DataFrame(index=index)
    for column in MET_COLUMNS:
        base[f"met_{column}_lag24"] = work[column].shift(MET_LAG_HOURS).astype(float)

    if blh.empty:
        blh_regular = pd.Series(np.nan, index=index)
    else:
        blh_regular = blh.reindex(index).astype(float)
    base["blh"] = blh_regular
    base["blh_lag24"] = blh_regular.shift(BLH_LAG_HOURS)

    ts = base.index
    local_dates = pd.Index(ts.date)
    base["is_holiday"] = local_dates.isin(holiday_dates).astype(int)

    month_day = ts.month * 100 + ts.day
    gec_start = GEC_START[0] * 100 + GEC_START[1]
    gec_end = GEC_END[0] * 100 + GEC_END[1]
    base["gec_window"] = ((month_day >= gec_start) & (month_day <= gec_end)).astype(int)

    dow_angle = 2.0 * np.pi * ts.dayofweek / 7.0
    base["dow_sin"] = np.sin(dow_angle)
    base["dow_cos"] = np.cos(dow_angle)

    hours = _to_hours(index)
    for name, period in FOURIER_PERIODS_HOURS.items():
        angle = 2.0 * np.pi * hours / period
        base[f"fourier_{name}_sin"] = np.sin(angle)
        base[f"fourier_{name}_cos"] = np.cos(angle)

    return base


def _series_rows(
    station_df: pd.DataFrame,
    base: pd.DataFrame,
    pollutant: str,
    station: str,
) -> pd.DataFrame:
    """Filas de una serie (estación x contaminante): lags + target."""
    series = (
        station_df[["timestamp", pollutant]]
        .dropna(subset=["timestamp"])
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")[pollutant]
        .astype(float)
        .reindex(base.index)
    )

    out = base.copy()
    for lag in LAGS_HOURS:
        out[f"lag_{lag}h"] = series.shift(lag)

    # mean(MP(t+1 … t+24h)): desplazar 24h hacia adelante y promediar 24 valores
    # completos (min_periods=24) para no usar promedios parciales.
    out[TARGET_COLUMN] = (
        series.shift(-TARGET_HORIZON_HOURS)
        .rolling(ROLLING_TARGET_HOURS, min_periods=ROLLING_TARGET_HOURS)
        .mean()
    )

    out["estacion"] = station
    out["pollutant"] = pollutant
    out = out[out[TARGET_COLUMN].notna()]
    return out.reset_index()


def build_features(
    sinca: pd.DataFrame,
    open_meteo: pd.DataFrame | None = None,
    feriados: pd.DataFrame | None = None,
    stations: list[str] | None = None,
    pollutants: list[str] | None = None,
) -> pd.DataFrame:
    """Construye el dataset supervisado (una fila por serie x timestamp).

    Devuelve un DataFrame largo con columnas de identidad (`estacion`,
    `pollutant`, `timestamp`), las `FEATURE_COLUMNS` y `target`.
    """
    if sinca is None or sinca.empty:
        raise ValueError("No hay datos SINCA validados para construir features.")

    stations = list(stations) if stations else list(STATIONS)
    pollutants = list(pollutants) if pollutants else list(TARGET_VARS)

    holiday_dates = _holiday_dates(feriados)
    blh = _blh_series(open_meteo)

    frames: list[pd.DataFrame] = []
    for station in stations:
        station_df = sinca[sinca["estacion"] == station].copy()
        if station_df.empty:
            log.warning("Sin filas validadas para la estación %s; se omite.", station)
            continue
        station_df["timestamp"] = pd.to_datetime(station_df["timestamp"])
        base = _station_base(station_df, blh, holiday_dates)
        for pollutant in pollutants:
            if pollutant not in station_df.columns:
                log.warning("Serie %s/%s ausente; se omite.", station, pollutant)
                continue
            frames.append(_series_rows(station_df, base, pollutant, station))

    if not frames:
        return pd.DataFrame(columns=[*IDENTITY_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN])

    out = pd.concat(frames, ignore_index=True)
    out["serie"] = out["estacion"].astype(str) + "|" + out["pollutant"].astype(str)
    return out.sort_values(["serie", "timestamp"]).reset_index(drop=True)


# --- Particionado ------------------------------------------------------------


def make_rolling_windows(
    rolling_start: date = ROLLING_START,
    rolling_end: date = ROLLING_END,
    train_years: int = TRAIN_YEARS,
    valid_months: int = VALID_MONTHS,
    step_months: int = STEP_MONTHS,
) -> list[dict]:
    """Ventanas deslizantes de tamaño fijo (train 2 años / valid 1 mes / step 1 mes).

    Cada ventana es `[train_start, train_end) = valid_start`, `[valid_start,
    valid_end)`. Devuelve los rangos (no los datos) para poder aplicarlos a
    cualquier serie.
    """
    start = pd.Timestamp(rolling_start, tz=_TZ)
    end = pd.Timestamp(rolling_end, tz=_TZ)
    origin = start + pd.DateOffset(years=train_years)

    windows: list[dict] = []
    while True:
        valid_start = origin
        valid_end = origin + pd.DateOffset(months=valid_months)
        if valid_end > end + pd.Timedelta(days=1):
            break
        windows.append(
            {
                "train_start": origin - pd.DateOffset(years=train_years),
                "train_end": valid_start,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )
        origin = origin + pd.DateOffset(months=step_months)
    return windows


def apply_window(series_df: pd.DataFrame, window: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorta una serie a `(train, valid)` según una ventana de `make_rolling_windows`."""
    ts = series_df["timestamp"]
    train = series_df[(ts >= window["train_start"]) & (ts < window["train_end"])]
    valid = series_df[(ts >= window["valid_start"]) & (ts < window["valid_end"])]
    return train, valid


def split_windows(series_df: pd.DataFrame, windows: list[dict]) -> list[dict]:
    """Aplica cada ventana a una serie, devolviendo `{**window, train, valid}`."""
    return [{**window, "train": t, "valid": v} for t, v in (apply_window(series_df, w) for w in windows)]


def final_holdout_split(
    series_df: pd.DataFrame,
    test_start: date = TEST_START,
    test_end: date = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa `(train, test)`: train = todo lo anterior al test; test = held-out."""
    start = pd.Timestamp(test_start, tz=_TZ)
    end = pd.Timestamp(test_end, tz=_TZ) + pd.Timedelta(days=1)
    ts = series_df["timestamp"]
    return series_df[ts < start], series_df[(ts >= start) & (ts < end)]


def series_list(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Pares `(estacion, pollutant)` presentes en el dataset, ordenados."""
    pairs = df[["estacion", "pollutant"]].drop_duplicates()
    return sorted(map(tuple, pairs.to_numpy().tolist()))
