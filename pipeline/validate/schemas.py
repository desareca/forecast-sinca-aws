"""Validación de calidad con pandera (5 dimensiones).

Dimensiones cubiertas (la "exactitud" contra fuente externa no aplica):

- **Validez**: tipos correctos y rangos plausibles por variable.
- **Completitud**: % de nulos de las series target (MP2.5/MP10) bajo el umbral.
- **Unicidad**: sin duplicados `(estacion, timestamp)`.
- **Oportunidad**: el último timestamp está dentro de las últimas N horas
  (solo se exige en modo incremental; en backfill no aplica).
- **Consistencia**: alineación horaria del `timestamp` y rangos de dirección de
  viento / humedad coherentes con sus unidades.

`validate_batch` devuelve `(valid, rejected, reasons)`: pandera permite separar
las filas que fallan un check de fila (van a cuarentena) de los checks de lote
(completitud/unicidad/oportunidad, que si fallan mandan el lote completo a
cuarentena).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pandera as pa

from config import (
    FRESHNESS_HOURS,
    MAX_NULL_RATE_TARGET,
    TARGET_VARS,
    TIMEZONE,
    VAR_RANGES,
)

log = logging.getLogger(__name__)

_TZ = ZoneInfo(TIMEZONE)


def _sinca_columns() -> dict[str, pa.Column]:
    columns: dict[str, pa.Column] = {
        "estacion": pa.Column(str),
        "timestamp": pa.Column(pd.DatetimeTZDtype(tz=_TZ)),
    }
    for var, (low, high) in VAR_RANGES.items():
        columns[var] = pa.Column(
            float,
            checks=pa.Check.in_range(low, high),
            nullable=True,
            required=False,
        )
    for icap_col in ("icap_pm10", "icap_pm25"):
        columns[icap_col] = pa.Column(
            float,
            checks=pa.Check.in_range(0.0, 500.0),
            nullable=True,
            required=False,
        )
    for flag in ("is_holiday", "gec_window", "day_of_week"):
        columns[flag] = pa.Column(nullable=True, required=False)
    return columns


def build_sinca_schema(enforce_freshness: bool = False) -> pa.DataFrameSchema:
    checks = [
        pa.Check(
            lambda df: not df.duplicated(["estacion", "timestamp"]).any(),
            error="unicidad: duplicados (estacion, timestamp)",
        ),
        pa.Check(
            lambda df: df["pm25"].isna().mean() <= MAX_NULL_RATE_TARGET,
            error=f"completitud: pm25 con mas de {MAX_NULL_RATE_TARGET:.0%} de nulos",
        ),
        pa.Check(
            lambda df: df["pm10"].isna().mean() <= MAX_NULL_RATE_TARGET,
            error=f"completitud: pm10 con mas de {MAX_NULL_RATE_TARGET:.0%} de nulos",
        ),
        pa.Check(
            lambda df: (
                df["timestamp"].dropna().dt.minute.eq(0).all()
                if not df["timestamp"].dropna().empty
                else True
            ),
            error="consistencia: timestamps no alineados a la hora",
        ),
    ]
    if enforce_freshness:
        deadline = datetime.now(tz=_TZ) - timedelta(hours=FRESHNESS_HOURS)
        checks.append(
            pa.Check(
                lambda df: (
                    not df["timestamp"].dropna().empty
                    and df["timestamp"].max() >= deadline
                ),
                error=f"oportunidad: ultimo timestamp anterior a {deadline.isoformat()}",
            )
        )
    return pa.DataFrameSchema(_sinca_columns(), checks=checks, strict=False, coerce=False)


def build_open_meteo_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column(pd.DatetimeTZDtype(tz=_TZ)),
            "boundary_layer_height": pa.Column(float, checks=pa.Check.ge(0.0), nullable=True),
        },
        checks=[
            pa.Check(
                lambda df: not df.duplicated(["timestamp"]).any(),
                error="unicidad: duplicados (timestamp)",
            ),
        ],
        strict=False,
        coerce=False,
    )


def build_feriados_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "fecha": pa.Column(pa.Date),
            "nombre": pa.Column(str),
            "tipo": pa.Column(str, checks=pa.Check.isin(["nacional", "regional"])),
        },
        checks=[
            pa.Check(
                lambda df: not df.duplicated(["fecha"]).any(),
                error="unicidad: duplicados (fecha)",
            ),
        ],
        strict=False,
        coerce=False,
    )


def validate_batch(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Valida un lote y lo divide en `(valid, rejected, reasons)`.

    - Fallos de checks de fila (rangos/tipos) -> esas filas van a `rejected`.
    - Fallos de checks de lote (completitud/unicidad/oportunidad) -> todo el
      lote va a `rejected` con el motivo.
    """
    if df.empty:
        return df, df, []

    work = df.reset_index(drop=True)
    try:
        schema.validate(work, lazy=True)
        return work, work.iloc[0:0].copy(), []
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases
        reasons: list[str] = []
        bad_indices: set[int] = set()
        for _, failure in failures.iterrows():
            column = failure.get("column")
            check = failure.get("check")
            reasons.append(f"{column or 'batch'}: {check}")
            index = failure.get("index")
            if index is None or (isinstance(index, float) and pd.isna(index)):
                bad_indices.update(int(i) for i in work.index)  # fallo de lote
            else:
                try:
                    bad_indices.add(int(index))
                except (TypeError, ValueError):
                    bad_indices.update(int(i) for i in work.index)
        bad = [i for i in bad_indices if i in work.index]
        rejected = work.loc[bad].copy()
        valid = work.drop(index=bad).reset_index(drop=True)
        return valid, rejected, sorted(set(reasons))
