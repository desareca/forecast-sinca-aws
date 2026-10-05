"""Persistencia en S3 (Parquet) con la convención `raw/` -> `validated/` -> `quarantine/`.

Estructura:
- `sinca-data/{layer}/sinca/estacion=<slug>/year=<yyyy>/month=<mm>/data.parquet`
- `sinca-data/{layer}/open_meteo/year=<yyyy>/month=<mm>/data.parquet`
- `sinca-data/{layer}/feriados/year=<yyyy>/data.parquet`

En `quarantine/` se adjunta `reason.json` con el motivo de rechazo (dimensión y
regla que falló), junto al Parquet rechazado.

Credenciales: usa el credential chain estándar de boto3. En Fargate (task role)
no hay profile; en local se respeta `AWS_PROFILE` explícito si está seteado.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re

import boto3
import pandas as pd

from config import STATIONS

log = logging.getLogger(__name__)


def get_client(profile: str | None = None):
    profile = profile or os.environ.get("AWS_PROFILE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _put_parquet(client, bucket: str, key: str, df: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def _put_reason(client, bucket: str, key: str, reason: list[str], rows: int, layer: str) -> None:
    payload = {"layer": layer, "filas": int(rows), "motivos": list(reason)}
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _write_partitioned(
    df: pd.DataFrame,
    bucket: str,
    layer: str,
    dataset: str,
    group_cols: list[str],
    path_builder,
    client,
    reason: list[str] | None = None,
) -> list[str]:
    if df.empty:
        return []
    work = df.copy()
    written: list[str] = []
    for keys, part in work.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        key = f"{layer}/{dataset}/{path_builder(*keys)}/data.parquet"
        out = part.drop(columns=[c for c in ("_year", "_month") if c in part.columns])
        _put_parquet(client, bucket, key, out)
        written.append(f"s3://{bucket}/{key}")
        if reason:
            _put_reason(client, bucket, key.replace("data.parquet", "reason.json"), reason, len(out), layer)
    return written


def write_sinca(
    df: pd.DataFrame,
    bucket: str,
    layer: str,
    client=None,
    reason: list[str] | None = None,
) -> list[str]:
    client = client or get_client()
    if df.empty:
        return []
    work = df.copy()
    work["_year"] = work["timestamp"].dt.year
    work["_month"] = work["timestamp"].dt.month

    def path(estacion: str, year: int, month: int) -> str:
        slug = STATIONS.get(estacion, {}).get("slug", _slugify(str(estacion)))
        return f"estacion={slug}/year={int(year)}/month={int(month):02d}"

    return _write_partitioned(work, bucket, layer, "sinca", ["estacion", "_year", "_month"], path, client, reason)


def write_open_meteo(
    df: pd.DataFrame,
    bucket: str,
    layer: str,
    client=None,
    reason: list[str] | None = None,
) -> list[str]:
    client = client or get_client()
    if df.empty:
        return []
    work = df.copy()
    work["_year"] = work["timestamp"].dt.year
    work["_month"] = work["timestamp"].dt.month

    def path(year: int, month: int) -> str:
        return f"year={int(year)}/month={int(month):02d}"

    return _write_partitioned(work, bucket, layer, "open_meteo", ["_year", "_month"], path, client, reason)


def write_feriados(
    df: pd.DataFrame,
    bucket: str,
    layer: str,
    client=None,
    reason: list[str] | None = None,
) -> list[str]:
    client = client or get_client()
    if df.empty:
        return []
    work = df.copy()
    work["_year"] = pd.to_datetime(work["fecha"]).dt.year

    def path(year: int) -> str:
        return f"year={int(year)}"

    return _write_partitioned(work, bucket, layer, "feriados", ["_year"], path, client, reason)
