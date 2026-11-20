"""Representaciones del operador global (`dmd-operador-global`, D4).

Un `Estado` guarda el vector de estado en cada posición final `t` de la grilla horaria (las
mismas ventanas que el AE, `ae.data.build_windows`) y sabe decodificar un estado a la ventana
de MP de 24 h que termina en él (µg/m³). Constructores:

- `hankel`: ventana MP normalizada `L × 6` aplanada (`L = 24` o `48`); `meteo=True` agrega los
  12 canales meteo `× L` (`hankelmet`). Decodificación exacta (últimas 24 filas).
- `pca_state`: PCA(12) de la ventana MP + PCA(4) de la ventana auxiliar `34 × 24`, ajustados en
  las filas de ajuste. Decodificación lineal (PCA MP).
- `ae_state`: latente `latent.parquet` de un AE v2 + `decoder_mp.keras`.

La normalización (`Contexto`) se ajusta solo con los tramos de ajuste (`train`; en el cierre,
`train`+`es`+`sel` para Hankel/PCA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ae import data as aed
from ae import dmd as dmdm
from features import partition as pt

HORIZONTE = 24
N_MP = aed.N_MP
N_AUX = aed.N_AUX


# --- Contexto: grilla + normalización ----------------------------------------


@dataclass
class Contexto:
    grid: aed.HourlyGrid
    norm: aed.Normalizer
    ng: aed.NormalizedGrid
    spans: dict[str, tuple[int, int]]
    fit_tramos: tuple[str, ...]
    part: pt.Partition

    def fit_span(self) -> tuple[int, int]:
        return (min(self.spans[t][0] for t in self.fit_tramos), max(self.spans[t][1] for t in self.fit_tramos))


def contexto(grid: aed.HourlyGrid, part: pt.Partition, fit_tramos: tuple[str, ...] = ("train",)) -> Contexto:
    """Normalizador ajustado solo con las horas de `fit_tramos` (contiguos)."""
    spans = aed.partition_spans(grid.index, part)
    lo = min(spans[t][0] for t in fit_tramos)
    hi = max(spans[t][1] for t in fit_tramos)
    mask = np.zeros(len(grid.index), dtype=bool)
    mask[lo:hi] = True
    norm = aed.fit_normalizer(grid, mask)
    return Contexto(grid=grid, norm=norm, ng=aed.apply_normalizer(grid, norm), spans=spans,
                    fit_tramos=tuple(fit_tramos), part=part)


def posiciones(ctx: Contexto, length: int, tramos=pt.TRAMOS, union_fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones finales elegibles (ventana MP sin huecos > 6 h y completa dentro del tramo).

    Devuelve `(positions, split)`. Con `union_fit=True`, los tramos de ajuste se tratan como un
    solo bloque (ventanas que cruzan sus bordes), etiquetado con el primero (`fit`).
    """
    eligible = aed.window_eligibility(ctx.grid, length)
    allpos = np.arange(len(ctx.grid.index))
    pos, split = [], []
    blocks: list[tuple[str, int, int]] = []
    if union_fit:
        lo, hi = ctx.fit_span()
        blocks.append(("fit", lo, hi))
        blocks += [(t, *ctx.spans[t]) for t in tramos if t not in ctx.fit_tramos]
    else:
        blocks = [(t, *ctx.spans[t]) for t in tramos]
    for name, lo, hi in blocks:
        keep = eligible & aed.assign_samples(allpos, length, lo, hi)
        pos.append(allpos[keep])
        split.append(np.full(int(keep.sum()), name, dtype=object))
    p = np.concatenate(pos) if pos else np.zeros(0, dtype=int)
    s = np.concatenate(split) if split else np.zeros(0, dtype=object)
    order = np.argsort(p, kind="stable")
    return p[order], s[order]


# --- Estado -------------------------------------------------------------------


@dataclass
class Estado:
    """Estado `z [N, d]` en posiciones de la grilla, con decodificación a MP."""

    nombre: str
    z: np.ndarray
    positions: np.ndarray
    split: np.ndarray
    d_mp: int
    norm: aed.Normalizer
    decode_z: Callable[[np.ndarray], np.ndarray]           # z [M, d] -> mp_z [M, 24, 6]
    encode_fn: Callable[[np.ndarray], np.ndarray] | None = None  # posiciones -> z (ventana real)
    fit_split: tuple[str, ...] = ("train",)
    extra: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return int(self.z.shape[1])

    def fit_mask(self) -> np.ndarray:
        return np.isin(self.split, list(self.fit_split))

    def segments(self, mask: np.ndarray | None = None) -> list[np.ndarray]:
        mask = self.fit_mask() if mask is None else mask
        return dmdm.split_segments(self.z[mask], self.positions[mask])

    def segment_positions(self, mask: np.ndarray | None = None) -> list[np.ndarray]:
        mask = self.fit_mask() if mask is None else mask
        pos = self.positions[mask]
        if pos.size == 0:
            return []
        breaks = np.where(np.diff(pos) != 1)[0] + 1
        return list(np.split(pos, breaks))

    def lookup(self, positions) -> np.ndarray:
        """Fila de cada posición (-1 si no está)."""
        positions = np.asarray(positions, dtype=int)
        if len(self.positions) == 0:
            return np.full(len(positions), -1, dtype=int)
        idx = np.clip(np.searchsorted(self.positions, positions), 0, len(self.positions) - 1)
        return np.where(self.positions[idx] == positions, idx, -1)

    def at(self, positions) -> np.ndarray:
        """Estado en `positions`: guardado si existe; si no, codificado con la ventana real."""
        positions = np.asarray(positions, dtype=int)
        rows = self.lookup(positions)
        out = np.full((len(positions), self.d), np.nan)
        have = rows >= 0
        out[have] = self.z[rows[have]]
        if (~have).any():
            if self.encode_fn is None:
                raise KeyError(f"{self.nombre}: {int((~have).sum())} posiciones sin estado")
            out[~have] = self.encode_fn(positions[~have])
        return out

    def decode(self, z: np.ndarray) -> np.ndarray:
        """`z [M, d]` -> ventana MP `[M, 24, 6]` en µg/m³."""
        return aed.mp_to_ugm3(self.decode_z(np.asarray(z, dtype=float)), self.norm)

    def decode_log(self, z: np.ndarray) -> np.ndarray:
        """`z [M, d]` -> ventana MP `[M, 24, 6]` en `log1p(µg/m³)` (sin recorte)."""
        mean, std = mp_scale(self.norm)
        return self.decode_z(np.asarray(z, dtype=float)) * std + mean


def mp_scale(norm: aed.Normalizer) -> tuple[np.ndarray, np.ndarray]:
    columns = [name for name, _, _ in aed.mp_channels()]
    return norm.mean[columns].to_numpy(dtype=float), norm.std[columns].to_numpy(dtype=float)


# --- Hankel -------------------------------------------------------------------


def _hankel_z(ng: aed.NormalizedGrid, length: int, pos: np.ndarray, meteo: bool) -> np.ndarray:
    win = aed.build_windows(ng, length, pos)
    z = win.x_mp.astype(float).reshape(len(pos), length * N_MP)
    if meteo:
        z = np.concatenate([z, win.y_aux.astype(float).reshape(len(pos), length * N_AUX)], axis=1)
    return z


def hankel(ctx: Contexto, length: int = HORIZONTE, meteo: bool = False, tramos=pt.TRAMOS,
           union_fit: bool = False, nombre: str | None = None) -> Estado:
    """Hankel de MP (`L × 6`), opcionalmente con los 12 canales meteo (`L × 12`)."""
    if length < HORIZONTE:
        raise ValueError("L debe ser >= 24")
    pos, split = posiciones(ctx, length, tramos, union_fit)
    d_mp = length * N_MP

    def decode_z(z):
        return z[:, :d_mp].reshape(-1, length, N_MP)[:, -HORIZONTE:, :]

    def encode_fn(p):
        return _hankel_z(ctx.ng, length, np.asarray(p, dtype=int), meteo)

    fit_split = ("fit",) if union_fit else ctx.fit_tramos
    return Estado(nombre=nombre or ("hankelmet" if meteo else ("hankel" if length == 24 else f"hankel{length}")),
                  z=_hankel_z(ctx.ng, length, pos, meteo), positions=pos, split=split, d_mp=d_mp,
                  norm=ctx.norm, decode_z=decode_z, encode_fn=encode_fn, fit_split=fit_split,
                  extra={"length": length, "meteo": meteo})


def bloques_por_estacion(length: int = HORIZONTE) -> list[np.ndarray]:
    """Índices del estado Hankel MP por estación (para `hankelest`: un operador por estación)."""
    channels = aed.mp_channels()
    out = []
    for station in dict.fromkeys(s for _, s, _ in channels):
        cols = [c for c, (_, s, _) in enumerate(channels) if s == station]
        out.append(np.array([k * N_MP + c for k in range(length) for c in cols], dtype=int))
    return out


# --- PCA ----------------------------------------------------------------------


def pca_fit(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:k].T


def pca_state(ctx: Contexto, k_mp: int = 12, k_aux: int = 4, tramos=pt.TRAMOS, union_fit: bool = False) -> Estado:
    """PCA(`k_mp`) de la ventana MP (24 × 6) + PCA(`k_aux`) de la ventana auxiliar (24 × 34)."""
    pos, split = posiciones(ctx, HORIZONTE, tramos, union_fit)
    fit_split = ("fit",) if union_fit else ctx.fit_tramos
    fit = np.isin(split, list(fit_split))

    def flat(p):
        win = aed.build_windows(ctx.ng, HORIZONTE, np.asarray(p, dtype=int))
        return (win.x_mp.astype(float).reshape(len(p), -1), win.x_aux.astype(float).reshape(len(p), -1))

    x_mp, x_aux = flat(pos)
    m_mp, c_mp = pca_fit(x_mp[fit], k_mp)
    m_aux, c_aux = pca_fit(x_aux[fit], k_aux)

    def project(xm, xa):
        return np.concatenate([(xm - m_mp) @ c_mp, (xa - m_aux) @ c_aux], axis=1)

    def decode_z(z):
        return (z[:, :k_mp] @ c_mp.T + m_mp).reshape(-1, HORIZONTE, N_MP)

    def encode_fn(p):
        return project(*flat(p))

    return Estado(nombre="pca", z=project(x_mp, x_aux), positions=pos, split=split, d_mp=k_mp, norm=ctx.norm,
                  decode_z=decode_z, encode_fn=encode_fn, fit_split=fit_split,
                  extra={"m_aux": m_aux, "c_aux": c_aux, "m_mp": m_mp, "c_mp": c_mp,
                         "proj_aux": lambda xa: (xa - m_aux) @ c_aux})


# --- AE -----------------------------------------------------------------------


def _predict(model, x, batch_size: int = 2048) -> np.ndarray:
    return np.asarray(model.predict(x, batch_size=batch_size, verbose=0), dtype=float)


def ae_state(ctx: Contexto, latent: pd.DataFrame, decoder, encoder=None, aux_cols: slice | None = None,
             nombre: str = "ae", fit_split: tuple[str, ...] = ("train",)) -> Estado:
    """Latente de un AE (`latent.parquet`: `timestamp`, `split`, `z_mp_*`, `z_aux_*`).

    `decoder`/`encoder`: modelos Keras (`decoder_mp.keras`, `encoder.keras`). Sin `encoder`, una
    posición sin latente guardado es un error.
    """
    zcols = [c for c in latent.columns if c.startswith("z_mp_")] + [c for c in latent.columns if c.startswith("z_aux_")]
    d_mp = sum(c.startswith("z_mp_") for c in zcols)
    stamps = pd.DatetimeIndex(pd.to_datetime(latent["timestamp"]))
    index = ctx.grid.index
    if index.tz is not None and stamps.tz is None:
        stamps = stamps.tz_localize(index.tz)
    elif index.tz is not None:
        stamps = stamps.tz_convert(index.tz)
    pos = index.get_indexer(stamps)
    if (pos < 0).any():
        raise ValueError(f"{nombre}: {int((pos < 0).sum())} timestamps del latente fuera de la grilla")
    frame = pd.DataFrame({"pos": pos, "split": latent["split"].to_numpy()})
    z = latent[zcols].to_numpy(dtype=float)
    order = np.argsort(pos, kind="stable")
    pos, z, split = pos[order], z[order], frame["split"].to_numpy()[order]
    if np.any(np.diff(pos) == 0):
        raise ValueError(f"{nombre}: posiciones duplicadas en el latente")
    sl = aux_cols if aux_cols is not None else aed.aux_input_columns(True)

    def decode_z(zz):
        return _predict(decoder, zz.astype("float32"))[:, -HORIZONTE:, :]

    encode_fn = None
    if encoder is not None:
        def encode_fn(p):
            win = aed.build_windows(ctx.ng, HORIZONTE, np.asarray(p, dtype=int))
            return _predict(encoder, [win.x_mp, win.x_aux[..., sl]])

    return Estado(nombre=nombre, z=z, positions=pos, split=split, d_mp=d_mp, norm=ctx.norm, decode_z=decode_z,
                  encode_fn=encode_fn, fit_split=fit_split,
                  extra={"encoder": encoder, "decoder": decoder, "aux_cols": sl})
