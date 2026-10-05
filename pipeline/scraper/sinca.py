"""Scraper de SINCA (Sistema de Información Nacional de Calidad del Aire).

Estrategia
----------
1. Se descarga la página de la estación (`index.php/estacion/index/id/<id>`) y
   se descubren los parámetros horarios disponibles, extrayendo de cada link el
   `macropath` y el `macro`. Esto hace al scraper robusto a cambios de
   parámetros por estación (la cobertura meteorológica varía entre estaciones).
2. Se descarga el CSV horario por parámetro desde el endpoint tabular
   `apub.tsindico2.cgi?outtype=xcl`, construyendo el `macro` completo como
   `./<macropath>/<macro>.ic` (convención vigente del CGI de SINCA).
3. Se pivotea a formato ancho (una fila por hora, una columna por variable).

Formato del CSV (separador `;`, decimal `,`):
- Calidad del aire: `FECHA(YYMMDD);HORA(HHMM);validados;preliminares;no validados;`
- Meteorología:     `FECHA(YYMMDD);HORA(HHMM);valor;`
`NA`/vacío = sin dato. El estado se prioriza validados > preliminares > no validados.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import (
    CAL_VARS,
    MET_VARS,
    SINCA_CGI,
    SINCA_STATION_PAGE,
    STATIONS,
    TIMEZONE,
)

log = logging.getLogger(__name__)

_USER_AGENT = "forecast-sinca-aws/1.0 (+https://github.com/desareca/forecast-sinca-aws)"
_LINK_RE = re.compile(r'href="([^"]*macropath=[^"]*)"')
_MACROPATH_RE = re.compile(r"macropath=([^&]*)")
_MACRO_RE = re.compile(r"macro=([^&]*)")
_FROM_RE = re.compile(r"from=([^&]*)")
_TO_RE = re.compile(r"to=([^&]*)")

_TZ = ZoneInfo(TIMEZONE)


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _parse_yymmdd(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    if len(value) != 6 or not value.isdigit():
        return None
    return datetime.strptime(value, "%y%m%d").date()


def _classify(macropath: str) -> tuple[str, str, str] | None:
    """Clasifica un `macropath` (./RIX/902/Cal/PM25) -> (kind, variable, unidad)."""
    parts = macropath.strip("./").split("/")
    if len(parts) < 4:
        return None
    kind, code = parts[-2], parts[-1]
    if kind == "Cal":
        hit = CAL_VARS.get(code)
        return ("cal", hit[0], hit[1]) if hit else None
    if kind == "Met":
        hit = MET_VARS.get(code)
        return ("met", hit[0], hit[1]) if hit else None
    return None


def discover_parameters(station_id: str, session: requests.Session | None = None) -> list[dict]:
    """Descubre los parámetros horarios disponibles de una estación.

    Devuelve una lista de dicts con `macropath`, `macro`, `variable`, `unidad`,
    `kind` y la cobertura (`from`/`to`) del macro. Se descartan los macros
    diarios (solo interesa la granularidad horaria).
    """
    session = session or new_session()
    resp = session.get(SINCA_STATION_PAGE.format(id=station_id), timeout=60)
    resp.raise_for_status()

    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for match in _LINK_RE.finditer(resp.text):
        href = match.group(1).replace("&amp;", "&")
        macropath = (_MACROPATH_RE.search(href) or [None, None])[1]
        macro = (_MACRO_RE.search(href) or [None, None])[1]
        if not macropath or not macro:
            continue
        if "horario" not in macro:  # descartar series diarias
            continue
        key = (macropath, macro)
        if key in seen:
            continue
        seen.add(key)
        classified = _classify(macropath)
        if classified is None:
            continue
        kind, variable, unidad = classified
        found.append(
            {
                "macropath": macropath,
                "macro": macro,
                "variable": variable,
                "unidad": unidad,
                "kind": kind,
                "from": _parse_yymmdd((_FROM_RE.search(href) or [None, None])[1]),
                "to": _parse_yymmdd((_TO_RE.search(href) or [None, None])[1]),
            }
        )
    return found


def _to_float(value: str) -> float:
    value = (value or "").strip()
    if value in ("", "NA", "N/A", "-"):
        return float("nan")
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return float("nan")


def _pick_state(validados: str, preliminares: str, no_validados: str) -> tuple[float, str]:
    for raw, estado in (
        (validados, "validado"),
        (preliminares, "preliminar"),
        (no_validados, "no_validado"),
    ):
        val = _to_float(raw)
        if val == val:  # not NaN
            return val, estado
    return float("nan"), "sin_dato"


def parse_csv(text: str, kind: str) -> pd.DataFrame:
    """Parsea el CSV de SINCA a un DataFrame [timestamp, valor, estado]."""
    rows: list[tuple[datetime, float, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("FECHA"):
            continue
        fields = line.split(";")
        if len(fields) < 3:
            continue
        fecha, hora = fields[0].strip(), fields[1].strip()
        if not (fecha.isdigit() and hora.isdigit()):
            continue
        try:
            ts = datetime.strptime(fecha + hora.zfill(4), "%y%m%d%H%M").replace(tzinfo=_TZ)
        except ValueError:
            continue
        if kind == "cal":
            validados = fields[2] if len(fields) > 2 else ""
            preliminares = fields[3] if len(fields) > 3 else ""
            no_validados = fields[4] if len(fields) > 4 else ""
            valor, estado = _pick_state(validados, preliminares, no_validados)
        else:
            valor, estado = _to_float(fields[2]), "validado"
        rows.append((ts, valor, estado))

    df = pd.DataFrame(rows, columns=["timestamp", "valor", "estado"])
    if not df.empty:
        df = df.drop_duplicates("timestamp", keep="first")
    return df


def download_parameter(
    macropath: str,
    macro: str,
    start: date,
    end: date,
    kind: str,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Descarga y parsea un parámetro horario en el rango [start, end]."""
    session = session or new_session()
    # El `macro` incluye el macropath y la extensión `.ic`; se construye la URL
    # a mano para no URL-encodear las barras (el CGI las necesita literales).
    url = (
        f"{SINCA_CGI}?outtype=xcl&macro=./{macropath}/{macro}.ic"
        f"&from={start.strftime('%y%m%d')}&to={end.strftime('%y%m%d')}"
        f"&path=/usr/airviro/data/CONAMA/&lang=esp&rsrc=&macropath={macropath}"
    )
    resp = session.get(url, timeout=180)
    resp.raise_for_status()
    text = resp.text
    if text.lstrip().startswith("psgraph:"):
        raise RuntimeError(f"SINCA devolvió error para {macropath}/{macro}: {text.strip()[:200]}")
    return parse_csv(text, kind)


def scrape_station(
    station_name: str,
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame ancho por estación: `estacion`, `timestamp` + variables.

    Un parámetro puede tener varios macros con coberturas distintas (típico en
    meteorología); se consultan todos los que intersectan el rango y se
    concatenan priorizando el primer valor no nulo por timestamp.
    """
    session = session or new_session()
    station = STATIONS[station_name]
    parameters = discover_parameters(station["id"], session)
    log.info("Estación %s: %d parámetros horarios descubiertos", station_name, len(parameters))

    by_variable: dict[str, list[dict]] = {}
    for param in parameters:
        by_variable.setdefault(param["variable"], []).append(param)

    merged: pd.DataFrame | None = None
    for variable, params in by_variable.items():
        parts: list[pd.DataFrame] = []
        for param in params:
            p_from = max(start, param["from"] or start)
            p_to = min(end, param["to"] or end)
            if p_from > p_to:
                continue
            try:
                df = download_parameter(
                    param["macropath"], param["macro"], p_from, p_to, param["kind"], session
                )
            except Exception as exc:  # noqa: BLE001 - un parámetro caído no aborta la estación
                log.warning("Fallo %s %s (%s): %s", station_name, variable, param["macro"], exc)
                continue
            if not df.empty:
                parts.append(df)

        if not parts:
            continue
        var_df = pd.concat(parts, ignore_index=True)
        var_df = var_df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
        var_df = var_df.rename(columns={"valor": variable})[["timestamp", variable]]
        merged = var_df if merged is None else merged.merge(var_df, on="timestamp", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["estacion", "timestamp"])

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged.insert(0, "estacion", station_name)
    return merged


def scrape_all(
    start: date,
    end: date,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    """Scrapea las 3 estaciones. Devuelve {estacion: DataFrame ancho}."""
    session = session or new_session()
    return {name: scrape_station(name, start, end, session) for name in STATIONS}
