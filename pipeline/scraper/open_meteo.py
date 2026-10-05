"""Altura de capa límite horaria desde Open-Meteo (JSON público, sin key).

- Para fechas históricas (backfill) se usa la Archive API.
- Para las últimas horas (incremental) se usa la Forecast API con `past_days`
  (la Archive API tiene un rezago de ~5 días).
- El rango se divide en la frontera entre ambas si es necesario.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    OPEN_METEO_ARCHIVE,
    OPEN_METEO_FORECAST,
    OPEN_METEO_LAT,
    OPEN_METEO_LON,
    TIMEZONE,
)

log = logging.getLogger(__name__)

_USER_AGENT = "forecast-sinca-aws/1.0 (+https://github.com/desareca/forecast-sinca-aws)"
_VARIABLE = "boundary_layer_height"
_ARCHIVE_LAG_DAYS = 5
_FORECAST_MAX_PAST_DAYS = 92

_TZ = ZoneInfo(TIMEZONE)


def _parse_hourly(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    values = hourly.get(_VARIABLE) or []
    rows = []
    for ts, value in zip(times, values):
        rows.append((pd.Timestamp(datetime.fromisoformat(ts), tz=_TZ), value))
    df = pd.DataFrame(rows, columns=["timestamp", _VARIABLE])
    if not df.empty:
        df = df.drop_duplicates("timestamp", keep="first")
    return df


def _fetch_archive(start: date, end: date, session: requests.Session) -> pd.DataFrame:
    params = {
        "latitude": OPEN_METEO_LAT,
        "longitude": OPEN_METEO_LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": _VARIABLE,
        "timezone": TIMEZONE,
    }
    resp = session.get(OPEN_METEO_ARCHIVE, params=params, timeout=120)
    resp.raise_for_status()
    return _parse_hourly(resp.json())


def _fetch_forecast(start: date, end: date, session: requests.Session) -> pd.DataFrame:
    today = date.today()
    past_days = min(max((today - start).days + 1, 0), _FORECAST_MAX_PAST_DAYS)
    forecast_days = max((end - today).days + 1, 1)
    params = {
        "latitude": OPEN_METEO_LAT,
        "longitude": OPEN_METEO_LON,
        "hourly": _VARIABLE,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    resp = session.get(OPEN_METEO_FORECAST, params=params, timeout=120)
    resp.raise_for_status()
    df = _parse_hourly(resp.json())
    if not df.empty:
        df = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)]
    return df


def fetch_boundary_layer_height(
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Altura de capa límite horaria [timestamp, boundary_layer_height] (metros)."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", _USER_AGENT)

    cutoff = date.today() - timedelta(days=_ARCHIVE_LAG_DAYS)
    frames: list[pd.DataFrame] = []

    if start <= cutoff:
        frames.append(_fetch_archive(start, min(end, cutoff), session))
    if end > cutoff:
        frames.append(_fetch_forecast(max(start, cutoff + timedelta(days=1)), end, session))

    if not frames:
        return pd.DataFrame(columns=["timestamp", _VARIABLE])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    return df
