"""Rollout del operador y pronóstico promedio de 24 h (`dmd-operador-global`, D6).

`autonomo`: `z_{t+24} = K²⁴ z_t` (o `A₂₄ z_t`) -> decodificación -> promedio de las 24 h de la
ventana `[t+1 … t+24]`. `oraculo`: decodifica el estado real en `t+24` (diagnóstico, 1.0).
"""
from __future__ import annotations

import numpy as np

from .estado import HORIZONTE, Estado
from .estimadores import Operador  # noqa: F401


def estado_final(op: Operador, z0: np.ndarray, pasos: int = HORIZONTE) -> np.ndarray:
    if op.pasos == pasos:
        return op.avanzar(z0, 1)
    if pasos % op.pasos:
        raise ValueError("pasos no es múltiplo del paso del operador")
    return op.avanzar(z0, pasos // op.pasos)


def autonomo(op: Operador, est: Estado, positions, pasos: int = HORIZONTE) -> tuple[np.ndarray, np.ndarray]:
    """Pronóstico `[N, 6]` (µg/m³) y ventana decodificada `[N, 24, 6]` desde las posiciones `t`."""
    z0 = est.at(positions)
    dec = est.decode(estado_final(op, z0, pasos))
    return dec.mean(axis=1), dec


def autonomo_log(op: Operador, est: Estado, positions, pasos: int = HORIZONTE) -> np.ndarray:
    """Ventana decodificada `[N, 24, 6]` en `log1p` (para Duan)."""
    return est.decode_log(estado_final(op, est.at(positions), pasos))


def oraculo(est: Estado, positions) -> tuple[np.ndarray, np.ndarray]:
    """Decodifica el estado real que termina en `t+24` y promedia (techo de la representación)."""
    dec = est.decode(est.at(np.asarray(positions, dtype=int) + HORIZONTE))
    return dec.mean(axis=1), dec


# --- Retransformación de log1p: smearing de Duan (1.2b, D7) ----------------------


def duan_factores(log_real: np.ndarray, log_pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """`S_{s,k} = mean(exp(ℓ − ℓ̂))` por paso `k` y serie `s` (`[24, 6]`), solo horas con valor real."""
    resid = np.where(real, np.asarray(log_real, float) - np.asarray(log_pred, float), np.nan)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.exp(resid), axis=0)
    return np.where(np.isfinite(out), out, 1.0)


def aplicar_duan(log_pred: np.ndarray, factores: np.ndarray) -> np.ndarray:
    """`ŷ = mean_k((1 + expm1(ℓ̂_k))·S_{s,k} − 1)` (µg/m³, recortado en 0). `log_pred [N, 24, 6]`."""
    y = np.clip(np.exp(np.asarray(log_pred, float)) * factores[None, :, :] - 1.0, 0.0, None)
    return y.mean(axis=1)


def ventanas_reales_log(grid, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`log1p(MP)` observado en `t+1 … t+24` (`[N, 24, 6]`) y máscara de valor real."""
    positions = np.asarray(positions, dtype=int)
    idx = positions[:, None] + np.arange(1, 25)[None, :]
    ok = idx < len(grid.index)
    idx = np.clip(idx, 0, len(grid.index) - 1)
    mp = grid.mp.to_numpy(dtype=float)[idx]
    real = grid.mp_real.to_numpy()[idx] & ok[..., None] & np.isfinite(mp)
    return np.log1p(np.clip(np.nan_to_num(mp), 0, None)), real


# --- Rollout exógeno (1.3b, D6): solo el bloque MP es estado ---------------------


def fuente_persistida(t_pos: np.ndarray, end: np.ndarray, length: int = HORIZONTE) -> tuple[np.ndarray, np.ndarray]:
    """Para la ventana que termina en `end`: horas `τ` y su fuente (`τ` si `τ ≤ t`; si no, `τ − 24`)."""
    tau = np.asarray(end, int)[:, None] - length + 1 + np.arange(length)[None, :]
    src = np.where(tau <= np.asarray(t_pos, int)[:, None], tau, tau - 24)
    return tau, src


def ventana_aux_persistida(ng, t_pos: np.ndarray, k: int, length: int = HORIZONTE) -> np.ndarray:
    """Ventana auxiliar `[N, L, 34]` que termina en `t+k`: meteo y máscaras por persistencia diurna, calendario real."""
    t_pos = np.asarray(t_pos, int)
    tau, src = fuente_persistida(t_pos, t_pos + k, length)
    vals = ng.aux_z.to_numpy(dtype=np.float32)[src]
    avail = ng.aux_avail.to_numpy(dtype=np.float32)[src]
    cal = ng.cal.to_numpy(dtype=np.float32)[np.clip(tau, 0, len(ng.index) - 1)]
    return np.concatenate([vals, avail, cal], axis=-1)


def entrada_hankel(ng, t_pos: np.ndarray, k: int) -> np.ndarray:
    """Entrada DMDc de Hankel en `t+k`: 12 canales meteo (persistencia diurna si `t+k > t`) + 10 de calendario."""
    t_pos = np.asarray(t_pos, int)
    tau = t_pos + k
    src = np.where(tau <= t_pos, tau, tau - 24)
    vals = ng.aux_z.to_numpy(dtype=float)[src]
    cal = ng.cal.to_numpy(dtype=float)[np.clip(tau, 0, len(ng.index) - 1)]
    return np.concatenate([vals, cal], axis=1)


def exogeno_estado_final(op: Operador, est: Estado, positions, aux_fn, aux0=None, pasos: int = HORIZONTE) -> np.ndarray:
    """`z_mp,t+k+1 = K_mm z_mp,t+k + K_ma z_aux,t+k`; `z_aux,t+k = aux_fn(t, k)` para `k ≥ 1`.

    `aux0`: entrada en `t` (por defecto, el bloque auxiliar del estado observado). Devuelve el estado
    en `t+24` como `[z_mp | z_aux]` (o solo `z_mp` si el estado no tiene bloque auxiliar).
    """
    positions = np.asarray(positions, int)
    z = est.at(positions)
    zmp = z[:, :est.d_mp]
    zaux = z[:, est.d_mp:] if aux0 is None else aux0
    kma = op.meta["K_ma"]
    for k in range(pasos):
        zmp = zmp @ op.A.T + zaux @ kma.T
        zaux = aux_fn(positions, k + 1)
    return np.concatenate([zmp, zaux], axis=1) if est.d > est.d_mp else zmp


def exogeno(op: Operador, est: Estado, positions, aux_fn, aux0=None, pasos: int = HORIZONTE):
    dec = est.decode(exogeno_estado_final(op, est, positions, aux_fn, aux0, pasos))
    return dec.mean(axis=1), dec
