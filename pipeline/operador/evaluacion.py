"""Filas de evaluación, métricas con bootstrap emparejado y diagnósticos (D3, D8, D12).

- `construir_eval_rows`: filas comunes (por serie, `timestamp`) de los 36 DMD v2 ∩ LightGBM v2
  ∩ persistencia v2, con `y_true` y las dos referencias.
- `Evaluador`: RMSE ponderado por serie y su distribución bootstrap (bloques móviles de días,
  emparejados: mismos índices para toda variante y referencia).
- Toda variante debe cubrir las filas: si falta una, error con variante y serie (sin recorte).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ae import data as aed
from ae.metrics import CONCENTRATION_BINS
from analysis.sel_summary import common_rows
from evaluate.evaluate import ALPHA_DEFAULT, regular_threshold, sample_weights, weighted_rmse
from features.partition import TZ

KEYS = ["estacion", "pollutant", "timestamp"]
SERIES: list[tuple[str, str]] = [(station, pollutant) for _, station, pollutant in aed.mp_channels()]
REFS = ("lightgbm", "persistencia")
B_DEFAULT = 2000
DIAS_DEFAULT = 7


def serie_label(station: str, pollutant: str) -> str:
    return f"{station}|{pollutant}"


def canal_de(station: str, pollutant: str) -> int:
    return SERIES.index((station, pollutant))


# --- Filas ---------------------------------------------------------------------


def construir_eval_rows(dmd: dict[str, pd.DataFrame], lgbm: pd.DataFrame, pers: pd.DataFrame) -> pd.DataFrame:
    """Intersección por `(estacion, pollutant, timestamp)` con valores finitos en todas las etiquetas."""
    frames = {**dmd, "lightgbm": lgbm, "persistencia": pers}
    common = common_rows(frames)
    base = common["persistencia"][KEYS + ["y_true", "y_pred"]].rename(columns={"y_pred": "persistencia"}).copy()
    base["lightgbm"] = common["lightgbm"]["y_pred"].to_numpy()
    if not np.allclose(common["lightgbm"]["y_true"].to_numpy(), base["y_true"].to_numpy(), rtol=1e-9, atol=1e-9):
        raise ValueError("y_true difiere entre LightGBM y persistencia en las filas comunes")
    base["canal"] = [canal_de(s, p) for s, p in zip(base["estacion"], base["pollutant"])]
    return base.sort_values(["canal", "timestamp"]).reset_index(drop=True)


def interseccion(rows: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Filas de `rows` presentes en `frame` (para variantes con otra elegibilidad, `h = 48`)."""
    keys = pd.MultiIndex.from_frame(frame[KEYS])
    return rows[pd.MultiIndex.from_frame(rows[KEYS]).isin(keys)]


# --- Bootstrap -----------------------------------------------------------------


def block_counts(n_days: int, B: int = B_DEFAULT, dias: int = DIAS_DEFAULT, seed: int = 0) -> np.ndarray:
    """Conteos `[B, n_days]` de un bootstrap de bloques móviles de `dias` días consecutivos."""
    rng = np.random.default_rng(seed)
    b = min(dias, n_days)
    n_blocks = int(np.ceil(n_days / b))
    starts = rng.integers(0, n_days - b + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B, -1)[:, :n_days]
    counts = np.zeros((B, n_days), dtype=float)
    np.add.at(counts, (np.repeat(np.arange(B), n_days), idx.ravel()), 1.0)
    return counts


def intervalo(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    q = (1.0 - level) / 2.0
    return float(np.quantile(values, q)), float(np.quantile(values, 1.0 - q))


@dataclass
class Puntaje:
    """RMSE ponderado por serie: estimación puntual `[6]` y bootstrap `[B, 6]`."""

    point: np.ndarray
    boot: np.ndarray
    n: np.ndarray

    @property
    def media(self) -> float:
        return float(np.mean(self.point))

    @property
    def media_boot(self) -> np.ndarray:
        return self.boot.mean(axis=1)

    @staticmethod
    def promedio(items: list["Puntaje"]) -> "Puntaje":
        """Media entre semillas (D9: puntaje AE = media de las 3 semillas)."""
        return Puntaje(point=np.mean([p.point for p in items], axis=0), boot=np.mean([p.boot for p in items], axis=0),
                       n=items[0].n)


class Evaluador:
    def __init__(self, rows: pd.DataFrame, B: int = B_DEFAULT, dias: int = DIAS_DEFAULT, seed: int = 0,
                 alpha: float = ALPHA_DEFAULT):
        self.rows = rows.reset_index(drop=True)
        self.alpha = alpha
        self.y = self.rows["y_true"].to_numpy(dtype=float)
        self.canal = self.rows["canal"].to_numpy(dtype=int)
        self.w = np.empty(len(self.rows))
        for pollutant in self.rows["pollutant"].unique():
            m = (self.rows["pollutant"] == pollutant).to_numpy()
            self.w[m] = sample_weights(self.y[m], pollutant, alpha)
        ts = pd.DatetimeIndex(pd.to_datetime(self.rows["timestamp"]))
        local = ts.tz_convert(TZ).tz_localize(None) if ts.tz is not None else ts
        dates = local.normalize().to_numpy()
        self.days = np.unique(dates)
        self.dia = np.searchsorted(self.days, dates)
        self.C = block_counts(len(self.days), B, dias, seed)
        self.B = B
        self.stamps = ts
        self._refs = {name: self.puntaje(self.rows[name].to_numpy(dtype=float)) for name in REFS}

    # Alineación
    def alinear(self, nombre: str, stamps, forecast: np.ndarray, rows_mask: np.ndarray | None = None) -> np.ndarray:
        """`forecast [T, 6]` en `stamps` -> vector alineado con las filas (error si falta alguna)."""
        stamps = pd.DatetimeIndex(stamps)
        if stamps.tz is None and self.stamps.tz is not None:
            stamps = stamps.tz_localize(self.stamps.tz)
        ti = stamps.get_indexer(self.stamps)
        out = np.full(len(self.rows), np.nan)
        ok = ti >= 0
        out[ok] = np.asarray(forecast, dtype=float)[ti[ok], self.canal[ok]]
        need = np.ones(len(self.rows), bool) if rows_mask is None else np.asarray(rows_mask, bool)
        bad = need & ~np.isfinite(out)
        if bad.any():
            series = sorted({serie_label(*SERIES[c]) for c in self.canal[bad]})
            raise ValueError(f"Variante {nombre}: sin pronóstico en {int(bad.sum())} filas de eval_rows "
                             f"(series: {', '.join(series)})")
        return out

    # Puntajes
    def puntaje(self, yhat: np.ndarray, mask: np.ndarray | None = None) -> Puntaje:
        yhat = np.asarray(yhat, dtype=float)
        m = np.ones(len(self.y), bool) if mask is None else np.asarray(mask, bool)
        if not np.all(np.isfinite(yhat[m])):
            raise ValueError("Predicciones no finitas en filas evaluadas")
        nd = len(self.days)
        a = np.zeros((len(SERIES), nd))
        w = np.zeros((len(SERIES), nd))
        err = np.where(m, self.w * (self.y - np.nan_to_num(yhat)) ** 2, 0.0)
        np.add.at(a, (self.canal, self.dia), err)
        np.add.at(w, (self.canal, self.dia), np.where(m, self.w, 0.0))
        point = np.sqrt(a.sum(axis=1) / w.sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.sqrt((self.C @ a.T) / (self.C @ w.T))
        n = np.bincount(self.canal[m], minlength=len(SERIES))
        return Puntaje(point=point, boot=boot, n=n)

    def ref(self, name: str, mask: np.ndarray | None = None) -> Puntaje:
        if mask is None:
            return self._refs[name]
        return self.puntaje(self.rows[name].to_numpy(dtype=float), mask)

    def mejor_ref(self, mask: np.ndarray | None = None) -> list[str]:
        """Por serie, la referencia de menor RMSE ponderado puntual (D12)."""
        lg, pe = self.ref("lightgbm", mask), self.ref("persistencia", mask)
        return ["lightgbm" if lg.point[s] <= pe.point[s] else "persistencia" for s in range(len(SERIES))]

    def tabla(self, p: Puntaje, mask: np.ndarray | None = None) -> pd.DataFrame:
        """Por serie y media: RMSE ponderado, razones vs mejor referencia y cada una, skill vs persistencia."""
        refs = {name: self.ref(name, mask) for name in REFS}
        best = self.mejor_ref(mask)
        best_point = np.array([refs[b].point[s] for s, b in enumerate(best)])
        best_boot = np.stack([refs[b].boot[:, s] for s, b in enumerate(best)], axis=1)
        quant = {
            "wrmse": (p.point, p.boot),
            "razon_mejor": (p.point / best_point, p.boot / best_boot),
            "razon_lgbm": (p.point / refs["lightgbm"].point, p.boot / refs["lightgbm"].boot),
            "razon_pers": (p.point / refs["persistencia"].point, p.boot / refs["persistencia"].boot),
            "skill_pers": (1 - p.point / refs["persistencia"].point, 1 - p.boot / refs["persistencia"].boot),
        }
        rows = []
        for s, (station, pollutant) in enumerate(SERIES):
            row = {"serie": serie_label(station, pollutant), "n": int(p.n[s]), "mejor_ref": best[s]}
            for key, (pt_, bt) in quant.items():
                lo, hi = intervalo(bt[:, s])
                row.update({key: float(pt_[s]), f"{key}_lo": lo, f"{key}_hi": hi})
            rows.append(row)
        media = {"serie": "media", "n": int(p.n.min()), "mejor_ref": ""}
        for key, (pt_, bt) in quant.items():
            lo, hi = intervalo(bt.mean(axis=1))
            media.update({key: float(np.mean(pt_)), f"{key}_lo": lo, f"{key}_hi": hi})
        rows.append(media)
        return pd.DataFrame(rows)

    def predicciones(self, yhat: np.ndarray, mask: np.ndarray | None = None, split: str = "sel") -> pd.DataFrame:
        m = np.ones(len(self.y), bool) if mask is None else np.asarray(mask, bool)
        out = self.rows.loc[m, KEYS + ["y_true"]].copy()
        out["y_pred"] = np.asarray(yhat)[m]
        out["split"] = split
        return out.reset_index(drop=True)


def diferencia(a: Puntaje, b: Puntaje) -> dict[str, float]:
    """Diferencia emparejada de la media de 6 series `a − b` con intervalo (mismos remuestreos)."""
    d = a.media_boot - b.media_boot
    lo, hi = intervalo(d)
    return {"diff": a.media - b.media, "lo": lo, "hi": hi}


def significativa(diff: dict[str, float]) -> bool:
    """`a` mejor que `b` solo si todo el intervalo de `a − b` queda bajo 0 (D9)."""
    return diff["hi"] < 0


# --- Diagnósticos --------------------------------------------------------------


def horizonte(decoded: np.ndarray, positions: np.ndarray, grid: aed.HourlyGrid) -> pd.DataFrame:
    """RMSE (µg/m³, solo valores reales) de la hora `t+k` de la ventana decodificada, `k = 1…24`."""
    positions = np.asarray(positions, dtype=int)
    idx = positions[:, None] + np.arange(1, 25)[None, :]
    idx_c = np.clip(idx, 0, len(grid.index) - 1)
    truth = grid.mp.to_numpy(dtype=float)[idx_c]
    real = grid.mp_real.to_numpy()[idx_c] & (idx < len(grid.index))[..., None]
    rows = []
    for c, (station, pollutant) in enumerate(SERIES):
        for k in range(24):
            m = real[:, k, c] & np.isfinite(decoded[:, k, c])
            if m.any():
                rows.append({"serie": serie_label(station, pollutant), "k": k + 1, "n": int(m.sum()),
                             "rmse": float(np.sqrt(np.mean((decoded[m, k, c] - truth[m, k, c]) ** 2)))})
    return pd.DataFrame(rows)


def episodios(ev: Evaluador, yhat: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Filas con `y ≥` umbral ICAP100 del contaminante: n, RMSE, RMSE ponderado y sesgo por serie."""
    m0 = np.ones(len(ev.y), bool) if mask is None else np.asarray(mask, bool)
    rows = []
    for c, (station, pollutant) in enumerate(SERIES):
        m = m0 & (ev.canal == c) & (ev.y >= regular_threshold(pollutant))
        row = {"serie": serie_label(station, pollutant), "n": int(m.sum())}
        if m.any():
            e = yhat[m] - ev.y[m]
            row.update({"rmse": float(np.sqrt(np.mean(e**2))), "wrmse": weighted_rmse(ev.y[m], yhat[m], ev.w[m]),
                        "sesgo": float(np.mean(e))})
        rows.append(row)
    return pd.DataFrame(rows)


def tramos(ev: Evaluador, yhat: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Sesgo, RMSE y pendiente (pred sobre real) por tramo de concentración del target y global."""
    m0 = np.ones(len(ev.y), bool) if mask is None else np.asarray(mask, bool)
    rows = []
    bins = (("todo", -np.inf, np.inf), *CONCENTRATION_BINS)
    for c, (station, pollutant) in enumerate(SERIES):
        for label, lo, hi in bins:
            m = m0 & (ev.canal == c) & (ev.y >= lo) & (ev.y < hi)
            row = {"serie": serie_label(station, pollutant), "tramo": label, "n": int(m.sum())}
            if m.sum() >= 2:
                t, r = ev.y[m], yhat[m]
                var = float(np.var(t))
                row.update({"sesgo": float(np.mean(r - t)), "rmse": float(np.sqrt(np.mean((r - t) ** 2))),
                            "pendiente": float(np.cov(t, r, ddof=0)[0, 1] / var) if var > 0 else float("nan")})
            rows.append(row)
    return pd.DataFrame(rows)
