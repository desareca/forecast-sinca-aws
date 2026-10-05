"""Feriados desde Nager.Date API (`date.nager.at`, JSON público, sin key).

Reemplaza al endpoint `apis.digital.gob.cl/fl/feriados` (caído). Se consulta
una vez por año del rango y se conservan los feriados nacionales
(`global=true`) más los regionales de La Araucanía (`counties` incluye
`CL-AR`), descartando los de otras regiones.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from config import NAGER_HOLIDAYS, NAGER_REGION_ARAUCANIA

log = logging.getLogger(__name__)

_USER_AGENT = "forecast-sinca-aws/1.0 (+https://github.com/desareca/forecast-sinca-aws)"


def fetch_holidays(
    years: list[int] | range,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Feriados relevantes para la zona saturada.

    Devuelve un DataFrame [fecha, nombre, tipo] con `tipo` ∈ {nacional, regional}.
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", _USER_AGENT)

    rows: list[dict] = []
    for year in sorted(set(int(y) for y in years)):
        resp = session.get(NAGER_HOLIDAYS.format(year=year), timeout=60)
        resp.raise_for_status()
        for holiday in resp.json():
            counties = holiday.get("counties") or []
            is_national = bool(holiday.get("global"))
            is_araucania = NAGER_REGION_ARAUCANIA in counties
            if not (is_national or is_araucania):
                continue
            rows.append(
                {
                    "fecha": pd.Timestamp(holiday["date"]).date(),
                    "nombre": holiday.get("localName") or holiday.get("name"),
                    "tipo": "nacional" if is_national else "regional",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["fecha", "nombre", "tipo"])
    df = pd.DataFrame(rows).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    return df


def years_in_range(start: date, end: date) -> range:
    return range(start.year, end.year + 1)
