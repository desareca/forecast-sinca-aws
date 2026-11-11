"""Preparación de datos para el autoencoder (change `ae-reconstruccion`, grupo 1).

Desde los datos validados construye una grilla horaria completa, con:

- **MP** (núcleo, 6 canales): pm25/pm10 × 3 estaciones.
- **Auxiliares** (12 canales): `pres`, `wspd`, `rain` × 3 estaciones, `temp` de
  Ñielol y Las Encinas, y `blh` (Open-Meteo).
- **Calendario** (10 canales, sin loss): Fourier diaria/semanal/anual sin/cos,
  `dow_sin/cos`, `is_holiday`, `gec_window`.

Reglas (design.md D3):

- Huecos de ≤ 6 h se interpolan linealmente. Se conserva la máscara de valores
  reales (`*_real`): loss y métricas se calculan solo sobre ellos.
- Ventana de MP con un hueco > 6 h se descarta. En auxiliares, un hueco > 6 h
  queda en 0 (tras normalizar) con una máscara de disponibilidad como entrada.
- `log1p` en pm25/pm10/rain/blh/wspd; z-score por canal con estadísticas solo
  del train de cada partición; recorte ±5 solo en auxiliares (nunca en MP).
- Una muestra pertenece a una partición solo si toda su ventana cae dentro.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import STATIONS, TARGET_VARS
from features.build import (
    FOURIER_PERIODS_HOURS,
    GEC_END,
    GEC_START,
    TEST_END,
    TEST_START,
    _blh_series,
    _holiday_dates,
    _to_hours,
)

GAP_MAX_HOURS = 6
CLIP_AUX = 5.0
LOG_VARS: tuple[str, ...] = ("pm25", "pm10", "rain", "blh", "wspd")
# Estaciones con `temp` válida (PLC II tiene 63% faltante: excluida).
AUX_TEMP_STATIONS: tuple[str, ...] = ("Nielol", "Las Encinas")

CALENDAR_CHANNELS: tuple[str, ...] = (
    "sin_daily", "cos_daily", "sin_weekly", "cos_weekly", "sin_annual", "cos_annual",
    "dow_sin", "dow_cos", "is_holiday", "gec_window",
)
CAL_DIM = len(CALENDAR_CHANNELS)  # 10
N_MP = len(TARGET_VARS) * len(STATIONS)  # 6
N_AUX = 12
# Entrada auxiliar: valores (12) + máscaras de disponibilidad (12) + calendario (10).
AUX_INPUT_DIM = N_AUX + N_AUX + CAL_DIM  # 34


# --- Canales -----------------------------------------------------------------


def station_slug(station: str) -> str:
    return STATIONS[station]["slug"]


def mp_channels() -> list[tuple[str, str, str]]:
    """Canales de MP `(nombre, estacion, pollutant)`, en orden fijo."""
    return [
        (f"{pollutant}_{station_slug(station)}", station, pollutant)
        for pollutant in TARGET_VARS
        for station in STATIONS
    ]


def aux_channels() -> list[tuple[str, str | None, str]]:
    """Canales auxiliares `(nombre, estacion|None, variable)`, en orden fijo."""
    out: list[tuple[str, str | None, str]] = []
    for station in STATIONS:
        for variable in ("pres", "wspd", "rain"):
            out.append((f"{variable}_{station_slug(station)}", station, variable))
    for station in AUX_TEMP_STATIONS:
        out.append((f"temp_{station_slug(station)}", station, "temp"))
    out.append(("blh", None, "blh"))
    return out


def channel_variable(name: str) -> str:
    """Variable física de un canal (`pm25_nielol` -> `pm25`, `blh` -> `blh`)."""
    return name if name == "blh" else name.split("_")[0]


def aux_input_columns(aux_input: bool) -> slice:
    """Columnas de la entrada auxiliar que usa la rama aux.

    Con `aux_input=True` se usan los 34 canales; apagado, solo el calendario
    (la rama aux recibe solo calendario, spec `modelado-ae`).
    """
    return slice(0, AUX_INPUT_DIM) if aux_input else slice(AUX_INPUT_DIM - CAL_DIM, AUX_INPUT_DIM)


# --- Grilla horaria ----------------------------------------------------------


@dataclass
class HourlyGrid:
    """Grilla horaria completa. Los `NaN` de `mp`/`aux` son huecos > 6 h."""

    index: pd.DatetimeIndex
    mp: pd.DataFrame        # valores (interpolados ≤6 h), NaN en huecos largos
    mp_real: pd.DataFrame   # bool: valor real medido (no interpolado)
    aux: pd.DataFrame       # idem a `mp`, para auxiliares
    aux_real: pd.DataFrame  # bool: valor real medido
    cal: pd.DataFrame       # calendario (sin NaN)


def fill_short_gaps(values: pd.Series, max_gap: int = GAP_MAX_HOURS) -> tuple[pd.Series, np.ndarray]:
    """Interpola linealmente huecos de ≤ `max_gap` horas.

    Devuelve `(serie, real)`: `serie` tiene los huecos cortos interpolados y los
    largos en `NaN`; `real` marca los valores medidos originalmente.
    """
    raw = values.to_numpy(dtype=float)
    real = np.isfinite(raw)
    fill = np.zeros(len(raw), dtype=bool)
    n = len(raw)
    i = 0
    while i < n:
        if real[i]:
            i += 1
            continue
        j = i
        while j < n and not real[j]:
            j += 1
        # Solo huecos internos (con medición a ambos lados) y cortos.
        if (j - i) <= max_gap and i > 0 and j < n:
            fill[i:j] = True
        i = j
    interpolated = values.interpolate(method="linear", limit_area="inside").to_numpy(dtype=float)
    out = raw.copy()
    out[fill] = interpolated[fill]
    return pd.Series(out, index=values.index), real


def _station_series(sinca: pd.DataFrame, station: str | None, variable: str, index: pd.DatetimeIndex) -> pd.Series:
    if station is None:
        return pd.Series(np.nan, index=index)
    frame = sinca[sinca["estacion"] == station]
    if frame.empty or variable not in frame.columns:
        return pd.Series(np.nan, index=index)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    series = pd.Series(frame[variable].astype(float).to_numpy(), index=stamps)
    series = series[~series.index.duplicated()].sort_index()
    return series.reindex(index)


def calendar_frame(index: pd.DatetimeIndex, feriados: pd.DataFrame | None) -> pd.DataFrame:
    """Canales de calendario (sin loss) en la grilla."""
    out = pd.DataFrame(index=index)
    hours = _to_hours(index)
    for name, period in FOURIER_PERIODS_HOURS.items():
        angle = 2.0 * np.pi * hours / period
        out[f"sin_{name}"] = np.sin(angle)
        out[f"cos_{name}"] = np.cos(angle)
    dow_angle = 2.0 * np.pi * index.dayofweek / 7.0
    out["dow_sin"] = np.sin(dow_angle)
    out["dow_cos"] = np.cos(dow_angle)
    holidays = _holiday_dates(feriados)
    out["is_holiday"] = pd.Index(index.date).isin(holidays).astype(float)
    month_day = index.month * 100 + index.day
    out["gec_window"] = (
        (month_day >= GEC_START[0] * 100 + GEC_START[1]) & (month_day <= GEC_END[0] * 100 + GEC_END[1])
    ).astype(float)
    return out[list(CALENDAR_CHANNELS)]


def build_grid(
    sinca: pd.DataFrame,
    open_meteo: pd.DataFrame | None = None,
    feriados: pd.DataFrame | None = None,
) -> HourlyGrid:
    """Arma la grilla horaria completa y aplica la interpolación ≤ 6 h."""
    if sinca is None or sinca.empty:
        raise ValueError("No hay datos SINCA validados para construir la grilla.")
    stamps = pd.to_datetime(sinca["timestamp"])
    index = pd.date_range(stamps.min(), stamps.max(), freq="h", name="timestamp")
    blh = _blh_series(open_meteo).reindex(index).astype(float) if open_meteo is not None else pd.Series(np.nan, index=index)

    mp_values, mp_reals = {}, {}
    for name, station, pollutant in mp_channels():
        filled, real = fill_short_gaps(_station_series(sinca, station, pollutant, index))
        mp_values[name], mp_reals[name] = filled, real

    aux_values, aux_reals = {}, {}
    for name, station, variable in aux_channels():
        raw = blh if station is None else _station_series(sinca, station, variable, index)
        filled, real = fill_short_gaps(raw)
        aux_values[name], aux_reals[name] = filled, real

    return HourlyGrid(
        index=index,
        mp=pd.DataFrame(mp_values, index=index),
        mp_real=pd.DataFrame(mp_reals, index=index).astype(bool),
        aux=pd.DataFrame(aux_values, index=index),
        aux_real=pd.DataFrame(aux_reals, index=index).astype(bool),
        cal=calendar_frame(index, feriados),
    )


# --- Transformación y normalización (fit solo en train) ----------------------


def _transform(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if channel_variable(column) in LOG_VARS:
            out[column] = np.log1p(out[column].clip(lower=0))
    return out


@dataclass
class Normalizer:
    """Media y desviación por canal (espacio transformado), ajustadas en train."""

    mean: pd.Series
    std: pd.Series


@dataclass
class NormalizedGrid:
    index: pd.DatetimeIndex
    mp_z: pd.DataFrame        # z-score sin recorte; NaN en huecos largos
    mp_real: pd.DataFrame
    aux_z_raw: pd.DataFrame   # z-score sin recorte; NaN en huecos largos
    aux_z: pd.DataFrame       # z-score recortado ±5; 0 en huecos largos
    aux_avail: pd.DataFrame   # bool: hay valor (real o interpolado) -> máscara de entrada
    aux_real: pd.DataFrame
    cal: pd.DataFrame
    normalizer: Normalizer


def fit_normalizer(grid: HourlyGrid, train_mask: np.ndarray) -> Normalizer:
    """Estadísticas (solo valores reales dentro de `train_mask`)."""
    frames_t = []
    for frame, real in ((grid.mp, grid.mp_real), (grid.aux, grid.aux_real)):
        transformed = _transform(frame)
        mask = real.to_numpy() & np.asarray(train_mask, dtype=bool)[:, None]
        frames_t.append(transformed.where(pd.DataFrame(mask, index=frame.index, columns=frame.columns)))
    both = pd.concat(frames_t, axis=1)
    mean = both.mean().fillna(0.0)
    std = both.std().fillna(1.0).clip(lower=1e-6)
    return Normalizer(mean=mean, std=std)


def apply_normalizer(grid: HourlyGrid, norm: Normalizer) -> NormalizedGrid:
    mp_z = (_transform(grid.mp) - norm.mean[grid.mp.columns]) / norm.std[grid.mp.columns]
    aux_z_raw = (_transform(grid.aux) - norm.mean[grid.aux.columns]) / norm.std[grid.aux.columns]
    aux_avail = aux_z_raw.notna()
    aux_z = aux_z_raw.clip(lower=-CLIP_AUX, upper=CLIP_AUX).fillna(0.0)
    return NormalizedGrid(
        index=grid.index,
        mp_z=mp_z,
        mp_real=grid.mp_real,
        aux_z_raw=aux_z_raw,
        aux_z=aux_z,
        aux_avail=aux_avail,
        aux_real=grid.aux_real,
        cal=grid.cal,
        normalizer=norm,
    )


def mp_to_ugm3(z: np.ndarray, norm: Normalizer) -> np.ndarray:
    """Deshace z-score y `log1p` de MP: `z[..., 6]` (orden de `mp_channels`) -> µg/m³."""
    columns = [name for name, _, _ in mp_channels()]
    mean = norm.mean[columns].to_numpy()
    std = norm.std[columns].to_numpy()
    return np.clip(np.expm1(np.asarray(z) * std + mean), 0.0, None)


# --- Ventanas y particiones --------------------------------------------------


def _as_ts(value, index: pd.DatetimeIndex) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if index.tz is None:
        return stamp.tz_localize(None) if stamp.tz is not None else stamp
    return stamp.tz_localize(index.tz) if stamp.tz is None else stamp.tz_convert(index.tz)


def span(index: pd.DatetimeIndex, start, stop) -> tuple[int, int]:
    """Posiciones `[lo, hi)` de la grilla entre `start` (incl.) y `stop` (excl.)."""
    return int(index.searchsorted(_as_ts(start, index))), int(index.searchsorted(_as_ts(stop, index)))


def holdout_spans(index: pd.DatetimeIndex) -> dict[str, tuple[int, int]]:
    """Partición held-out: train = todo lo anterior al test; test fijo (design §3)."""
    test_stop = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)
    return {
        "train": span(index, index[0], TEST_START),
        "test": span(index, TEST_START, test_stop),
    }


def rolling_spans(index: pd.DatetimeIndex, windows: list[dict]) -> list[dict[str, tuple[int, int]]]:
    """Particiones train/valid de cada ventana deslizante (`make_rolling_windows`)."""
    return [
        {
            "train": span(index, w["train_start"], w["train_end"]),
            "valid": span(index, w["valid_start"], w["valid_end"]),
        }
        for w in windows
    ]


def window_eligibility(grid: HourlyGrid, length: int) -> np.ndarray:
    """Por posición final `t`: la ventana `[t-L+1, t]` de MP no tiene huecos > 6 h."""
    bad = grid.mp.isna().any(axis=1).astype(float).to_numpy()
    counts = pd.Series(bad).rolling(length, min_periods=length).sum().to_numpy()
    return np.nan_to_num(counts, nan=1.0) == 0


def assign_samples(end_positions: np.ndarray, length: int, lo: int, hi: int) -> np.ndarray:
    """Máscara de muestras cuya ventana completa cae en `[lo, hi)`."""
    starts = end_positions - length + 1
    return (starts >= lo) & (end_positions < hi)


@dataclass
class Windows:
    """Muestras `[L × ·]` listas para la red (float32)."""

    length: int
    end_positions: np.ndarray  # posiciones finales t en la grilla
    index: pd.DatetimeIndex    # timestamps de t
    x_mp: np.ndarray           # [N, L, 6]
    mp_real: np.ndarray        # [N, L, 6] bool: valor real
    x_aux: np.ndarray          # [N, L, 34] (valores, máscaras, calendario)
    y_aux: np.ndarray          # [N, L, 12] objetivo auxiliar (z recortado)
    aux_real: np.ndarray       # [N, L, 12] bool

    def __len__(self) -> int:
        return len(self.end_positions)


def build_windows(ng: NormalizedGrid, length: int, end_positions: np.ndarray) -> Windows:
    """Corta ventanas `[L × ·]` en las posiciones finales dadas (sin filtrar)."""
    end_positions = np.asarray(end_positions, dtype=int)
    idx = (end_positions - length + 1)[:, None] + np.arange(length)[None, :]
    mp = ng.mp_z.fillna(0.0).to_numpy(dtype=np.float32)
    aux_vals = ng.aux_z.to_numpy(dtype=np.float32)
    avail = ng.aux_avail.to_numpy(dtype=np.float32)
    cal = ng.cal.to_numpy(dtype=np.float32)
    x_mp = mp[idx]
    aux_block = aux_vals[idx]
    x_aux = np.concatenate([aux_block, avail[idx], cal[idx]], axis=-1)
    return Windows(
        length=length,
        end_positions=end_positions,
        index=ng.index[end_positions],
        x_mp=x_mp,
        mp_real=ng.mp_real.to_numpy()[idx],
        x_aux=x_aux,
        y_aux=aux_block,
        aux_real=ng.aux_real.to_numpy()[idx],
    )


def select_samples(
    grid: HourlyGrid,
    ng: NormalizedGrid,
    length: int,
    lo: int,
    hi: int,
) -> Windows:
    """Muestras elegibles (MP sin huecos > 6 h) cuya ventana completa está en `[lo, hi)`."""
    eligible = window_eligibility(grid, length)
    positions = np.arange(len(grid.index))
    keep = eligible & assign_samples(positions, length, lo, hi)
    return build_windows(ng, length, positions[keep])


def survival_counts(grid: HourlyGrid, lengths: tuple[int, ...], spans: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Ventanas que sobreviven a la regla de descarte, por `L` y partición (tarea 9.5)."""
    rows = []
    positions = np.arange(len(grid.index))
    for length in lengths:
        eligible = window_eligibility(grid, length)
        for name, (lo, hi) in spans.items():
            rows.append({"L": length, "particion": name, "muestras": int((eligible & assign_samples(positions, length, lo, hi)).sum())})
    return pd.DataFrame(rows)


# --- Reporte de rangos (tarea 1.7) ------------------------------------------


def range_report(ng: NormalizedGrid, real_rows: np.ndarray | None = None) -> pd.DataFrame:
    """Mínimo, máximo, percentiles y fracción recortada por canal (espacio z)."""
    rows = []
    rows_mask = np.ones(len(ng.index), dtype=bool) if real_rows is None else np.asarray(real_rows, dtype=bool)
    specs = (
        ("mp", ng.mp_z, ng.mp_real, False),
        ("aux", ng.aux_z_raw, ng.aux_real, True),
    )
    for group, frame, real, is_aux in specs:
        for column in frame.columns:
            values = frame[column].to_numpy(dtype=float)[rows_mask]
            present = real[column].to_numpy()[rows_mask] & np.isfinite(values)
            v = values[present]
            if v.size == 0:
                continue
            clipped = float((np.abs(v) > CLIP_AUX).mean()) if is_aux else 0.0
            rows.append(
                {
                    "grupo": group,
                    "canal": column,
                    "min": float(v.min()),
                    "p01": float(np.quantile(v, 0.01)),
                    "p50": float(np.quantile(v, 0.50)),
                    "p99": float(np.quantile(v, 0.99)),
                    "max": float(v.max()),
                    "frac_recortada": clipped,
                }
            )
    return pd.DataFrame(rows)
