"""Espectro, modos y selección de modos del operador (`dmd-operador-global`, D4, D8, D9).

Convención: un par conjugado cuenta como **un** modo (se reporta el de `arg λ ≥ 0`). Período en
horas `2π/|arg λ|` (∞ si real); vida media en horas `ln 0.5 / ln |λ|` (∞ si `|λ| ≥ 1`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOL_IMAG = 1e-10


def representantes(lam: np.ndarray, tol: float = TOL_IMAG) -> np.ndarray:
    """Índices de un representante por modo (`Im λ ≥ 0`; reales una vez)."""
    lam = np.asarray(lam, dtype=complex)
    return np.where(lam.imag >= -tol)[0]


def periodo_h(lam, paso_h: float = 1.0) -> np.ndarray:
    ang = np.abs(np.angle(np.asarray(lam, dtype=complex)))
    with np.errstate(divide="ignore"):
        return np.where(ang > TOL_IMAG, 2 * np.pi * paso_h / np.where(ang > TOL_IMAG, ang, 1.0), np.inf)


def vida_media_h(lam, paso_h: float = 1.0) -> np.ndarray:
    mod = np.abs(np.asarray(lam, dtype=complex))
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where((mod < 1) & (mod > 0), np.log(0.5) * paso_h / np.log(np.where((mod < 1) & (mod > 0), mod, 0.5)), np.inf)
    return np.where(mod == 0, 0.0, hl)


def espectro(a_or_lam, paso_h: float = 1.0) -> pd.DataFrame:
    """Tabla de modos: `lam_re`, `lam_im`, `abs`, `periodo_h`, `vida_media_h`, `tipo`.

    Acepta una matriz (se calculan sus autovalores) o un vector de autovalores.
    """
    arr = np.asarray(a_or_lam)
    lam = np.linalg.eigvals(arr) if arr.ndim == 2 else arr.astype(complex)
    rep = representantes(lam)
    lam = lam[rep]
    order = np.argsort(-np.abs(lam), kind="stable")
    lam = lam[order]
    return pd.DataFrame({
        "modo": np.arange(len(lam)),
        "lam_re": lam.real,
        "lam_im": lam.imag,
        "abs": np.abs(lam),
        "periodo_h": periodo_h(lam, paso_h),
        "vida_media_h": vida_media_h(lam, paso_h),
        "tipo": np.where(np.abs(lam.imag) > TOL_IMAG, "complejo", "real"),
    })
