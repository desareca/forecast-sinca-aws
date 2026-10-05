"""Orquestador del pipeline de datos.

Flujo: scraper -> icap -> validate -> persist.

Modos:
- `incremental` (default): últimas 48 h (re-ingesta solapada para corregir
  registros preliminares del SINCA). Exige frescura (dimensión "oportunidad").
- `backfill`: rango `--from`/`--to` (por defecto 2021-01-01 -> hoy).

Uso:
    python entrypoint.py --mode incremental
    python entrypoint.py --mode backfill --from 2021-01-01 --to 2024-12-31
    python entrypoint.py --dry-run            # no escribe en S3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from config import (
    DATA_BUCKET_DEFAULT,
    MAX_QUARANTINE_RATE_DEFAULT,
    STATIONS,
    TARGET_VARS,
    TIMEZONE,
)
from icap.icap import add_icap
from persist.s3 import get_client, write_feriados, write_open_meteo, write_sinca
from scraper.feriados import fetch_holidays, years_in_range
from scraper.open_meteo import fetch_boundary_layer_height
from scraper.sinca import new_session, scrape_station
from validate.schemas import (
    build_feriados_schema,
    build_open_meteo_schema,
    build_sinca_schema,
    validate_batch,
)

log = logging.getLogger("pipeline")


def add_calendar_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    """Agrega día de semana, ventana GEC (1 abr-15 sep) y flag de feriado."""
    out = df.copy()
    ts = out["timestamp"]
    out["day_of_week"] = ts.dt.dayofweek
    month_day = ts.dt.month * 100 + ts.dt.day
    out["gec_window"] = ((month_day >= 401) & (month_day <= 915)).astype(int)
    out["is_holiday"] = ts.dt.date.isin(holiday_dates).astype(int)
    return out


def resolve_range(args: argparse.Namespace) -> tuple[date, date]:
    today = datetime.now(tz=ZoneInfo(TIMEZONE)).date()
    if args.mode == "backfill" or args.start is not None:
        start = args.start or date(2021, 1, 1)
        end = args.end or today
    else:
        end = args.end or today
        start = args.start or (end - timedelta(days=2))
    if start > end:
        raise SystemExit(f"Rango inválido: {start} > {end}")
    return start, end


def run(start: date, end: date, bucket: str, mode: str, dry_run: bool) -> int:
    session = new_session()
    log.info("Rango: %s -> %s (modo=%s, dry_run=%s)", start, end, mode, dry_run)

    # 1. Fuentes auxiliares -------------------------------------------------
    holidays = fetch_holidays(years_in_range(start, end), session=session)
    holiday_dates = set(holidays["fecha"]) if not holidays.empty else set()
    log.info("Feriados: %d registros", len(holidays))

    open_meteo = fetch_boundary_layer_height(start, end, session=session)
    log.info("Open-Meteo: %d filas de altura de capa límite", len(open_meteo))

    # 2. SINCA + features + ICAP -------------------------------------------
    raw_stations: dict[str, pd.DataFrame] = {}
    for name in STATIONS:
        df = scrape_station(name, start, end, session)
        for var in TARGET_VARS:
            if var not in df.columns:
                df[var] = float("nan")
        df = add_calendar_features(df, holiday_dates)
        df = add_icap(df)
        raw_stations[name] = df
        log.info("SINCA %s: %d filas", name, len(df))

    client = None if dry_run else get_client()

    # 3. Persistir raw ------------------------------------------------------
    if not dry_run:
        for df in raw_stations.values():
            write_sinca(df, bucket, "raw", client)
        write_open_meteo(open_meteo, bucket, "raw", client)
        write_feriados(holidays, bucket, "raw", client)

    # 4. Validar y persistir validated/quarantine ---------------------------
    sinca_schema = build_sinca_schema(enforce_freshness=(mode == "incremental"))
    max_rate = float(os.environ.get("MAX_QUARANTINE_RATE", MAX_QUARANTINE_RATE_DEFAULT))
    exceeded: list[str] = []

    for name, df in raw_stations.items():
        valid, rejected, reasons = validate_batch(df, sinca_schema)
        rate = len(rejected) / max(len(df), 1)
        log.info(
            "SINCA %s: %d válidas / %d rechazadas (%.1f%%) motivos=%s",
            name, len(valid), len(rejected), rate * 100, reasons,
        )
        if dry_run:
            continue
        if rate > max_rate:
            exceeded.append(name)
            log.error(
                "Estación %s supera el umbral de cuarentena (%.1f%% > %.1f%%); "
                "no se persiste validated.",
                name, rate * 100, max_rate * 100,
            )
        elif not valid.empty:
            write_sinca(valid, bucket, "validated", client)
        if not rejected.empty:
            write_sinca(rejected, bucket, "quarantine", client, reason=reasons or ["sin_motivo"])

    om_valid, om_rejected, om_reasons = validate_batch(open_meteo, build_open_meteo_schema())
    if not dry_run:
        if not om_valid.empty:
            write_open_meteo(om_valid, bucket, "validated", client)
        if not om_rejected.empty:
            write_open_meteo(om_rejected, bucket, "quarantine", client, reason=om_reasons or ["sin_motivo"])

    fer_valid, fer_rejected, fer_reasons = validate_batch(holidays, build_feriados_schema())
    if not dry_run:
        if not fer_valid.empty:
            write_feriados(fer_valid, bucket, "validated", client)
        if not fer_rejected.empty:
            write_feriados(fer_rejected, bucket, "quarantine", client, reason=fer_reasons or ["sin_motivo"])

    if exceeded:
        log.error("Corrida finalizada con estaciones sobre el umbral: %s", ", ".join(exceeded))
        return 1
    log.info("Corrida OK.")
    return 0


def load_dotenv(path: str = ".env") -> None:
    """Carga variables desde un `.env` local (stdlib, sin dependencias).

    No sobreescribe variables ya presentes en el entorno. En Fargate no hay
    `.env` (las variables las setea la task definition), así que es no-op.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de datos SINCA (Fase 2).")
    parser.add_argument("--mode", choices=["incremental", "backfill"], default="incremental")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, default=None,
                        help="Fecha inicial YYYY-MM-DD (backfill).")
    parser.add_argument("--to", dest="end", type=date.fromisoformat, default=None,
                        help="Fecha final YYYY-MM-DD (backfill).")
    parser.add_argument("--bucket", default=os.environ.get("DATA_BUCKET", DATA_BUCKET_DEFAULT))
    parser.add_argument("--dry-run", action="store_true", help="No escribe en S3.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    args = parse_args(argv)
    start, end = resolve_range(args)
    return run(start, end, args.bucket, args.mode, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
