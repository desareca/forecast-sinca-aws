"""Partición `ae-dmd` (change `ae-dmd-prep`, D1).

Tramos en fechas locales (`America/Santiago`), `[inicio, fin)`:

- `train`: entrenar AE y ajustar K.
- `es`: early stopping de los AE.
- `sel`: validación de selección (ganador de cada etapa).
- `test`: solo latentes en este change; confirma al final de `ae-dmd`.

Una fila con target (`target(t) = mean(MP(t+1..t+24))`) entra al tramo solo si
`inicio <= t < fin - EMBARGO_HOURS`, para que ningún target use datos del tramo siguiente.
`holdout` no se modela aquí: sigue en `ae/data.py::holdout_spans`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from features.build import TEST_END, TEST_START

TZ = "America/Santiago"
TRAMOS: tuple[str, ...] = ("train", "es", "sel", "test")
EMBARGO_HOURS = 24


@dataclass(frozen=True)
class Partition:
    """Partición con nombre y tramos `[inicio, fin)` en fechas locales."""

    name: str
    bounds: dict[str, tuple[date, date]]

    @property
    def token(self) -> str:
        """Nombre sin guiones, para nombres de run (`ae-dmd` -> `aedmd`)."""
        return self.name.replace("-", "")


PARTITIONS: dict[str, Partition] = {
    "ae-dmd": Partition(
        name="ae-dmd",
        bounds={
            "train": (date(2021, 1, 1), date(2026, 3, 1)),
            "es": (date(2026, 3, 1), date(2026, 4, 1)),
            "sel": (date(2026, 4, 1), date(2026, 7, 1)),
            "test": (TEST_START, TEST_END + timedelta(days=1)),
        },
    ),
}


def get_partition(name: str) -> Partition:
    """Devuelve la partición por nombre; `ValueError` si no existe."""
    if name not in PARTITIONS:
        raise ValueError(f"Partición desconocida {name!r}; válidas: {', '.join(sorted(PARTITIONS))}")
    return PARTITIONS[name]


def tramo_bounds(part: Partition, tramo: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Límites `(inicio, fin)` del tramo, localizados en `America/Santiago`."""
    if tramo not in part.bounds:
        raise ValueError(f"Tramo desconocido {tramo!r}; válidos: {', '.join(TRAMOS)}")
    start, stop = part.bounds[tramo]
    return pd.Timestamp(start, tz=TZ), pd.Timestamp(stop, tz=TZ)


def row_mask(
    timestamps,
    part: Partition,
    tramo: str,
    embargo_hours: int = EMBARGO_HOURS,
) -> np.ndarray:
    """Máscara booleana: `inicio <= t < fin - embargo`.

    Acepta timestamps naive (hora local) o con tz (se convierten a hora local de
    Santiago), igual que `ae.data._as_ts`: el resultado es el mismo para ambos.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is not None:
        ts = ts.tz_convert(TZ).tz_localize(None)
    start, stop = tramo_bounds(part, tramo)
    start = start.tz_localize(None)
    stop = stop.tz_localize(None) - pd.Timedelta(hours=embargo_hours)
    return np.asarray((ts >= start) & (ts < stop), dtype=bool)


def to_json(part: Partition) -> str:
    """Serialización estable de la partición (para `config.json` y params de MLflow)."""
    payload = {
        "name": part.name,
        "token": part.token,
        "tz": TZ,
        "embargo_hours": EMBARGO_HOURS,
        "bounds": {
            tramo: [start.isoformat(), stop.isoformat()]
            for tramo, (start, stop) in ((t, part.bounds[t]) for t in TRAMOS)
        },
    }
    return json.dumps(payload, sort_keys=True)
