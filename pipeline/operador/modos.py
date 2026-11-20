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


# --- Emparejamiento y bootstrap por modo (1.5, D8) --------------------------------

TOL_APARICION = 0.25


def coords(lam) -> np.ndarray:
    lam = np.asarray(lam, dtype=complex)
    return np.stack([np.log(np.maximum(np.abs(lam), 1e-300)), np.abs(np.angle(lam))], axis=1)


def emparejar(ref_lam, lam) -> tuple[np.ndarray, np.ndarray]:
    """Húngaro entre representantes (`arg λ ≥ 0`) en `(log|λ|, arg λ)`. Devuelve índices `(i_ref, j)`."""
    from scipy.optimize import linear_sum_assignment

    a, b = coords(ref_lam), coords(lam)
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return linear_sum_assignment(cost)


def _rel(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_inf = np.isinf(a) & np.isinf(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a - b) / np.abs(a)
    return np.where(both_inf, 0.0, np.where(np.isinf(a) | np.isinf(b), np.inf, r))


def aparece(ref_lam, lam, tol: float = TOL_APARICION) -> np.ndarray:
    """`True` si el modo emparejado tiene período y vida media a ≤ `tol` relativo (reales: solo vida media)."""
    ref_lam = np.asarray(ref_lam, complex)
    lam = np.asarray(lam, complex)
    real_ref = np.abs(ref_lam.imag) <= TOL_IMAG
    real_b = np.abs(lam.imag) <= TOL_IMAG
    ok_hl = _rel(vida_media_h(ref_lam), vida_media_h(lam)) <= tol
    ok_per = _rel(periodo_h(ref_lam), periodo_h(lam)) <= tol
    return np.where(real_ref, ok_hl & real_b, ok_hl & ok_per & ~real_b)


def bootstrap_modos(ref_lam, fit_fn, day_of_row: np.ndarray, B: int = 200, dias: int = 7, seed: int = 0,
                    paso_h: float = 1.0) -> pd.DataFrame:
    """Re-estima el operador `B` veces con bloques móviles de `dias` días del train (pesos = conteos)
    y empareja sus modos con `ref_lam`. `fit_fn(pesos_por_fila) -> autovalores`.

    Devuelve por modo de referencia: frecuencia de aparición e intervalos 95 % de período y vida media.
    """
    from .evaluacion import block_counts

    ref_lam = np.asarray(ref_lam, complex)
    ref = ref_lam[representantes(ref_lam)]
    ref = ref[np.argsort(-np.abs(ref), kind="stable")]
    days = np.unique(day_of_row, return_inverse=True)[1]
    counts = block_counts(int(days.max()) + 1, B, dias, seed)
    per = np.full((B, len(ref)), np.nan)
    hl = np.full((B, len(ref)), np.nan)
    hit = np.zeros((B, len(ref)), bool)
    for b in range(B):
        lam = np.asarray(fit_fn(counts[b, days]), complex)
        lam = lam[representantes(lam)]
        i, j = emparejar(ref, lam)
        per[b, i] = periodo_h(lam[j], paso_h)
        hl[b, i] = vida_media_h(lam[j], paso_h)
        hit[b, i] = aparece(ref[i], lam[j])
    out = espectro(ref, paso_h).copy()
    out["aparicion"] = hit.mean(axis=0)
    for name, arr in (("periodo_h", per), ("vida_media_h", hl)):
        fin = np.where(np.isfinite(arr), arr, np.nan)
        out[f"{name}_lo"] = np.nanquantile(fin, 0.025, axis=0) if np.isfinite(fin).any() else np.nan
        out[f"{name}_hi"] = np.nanquantile(fin, 0.975, axis=0) if np.isfinite(fin).any() else np.nan
    return out


# --- Selección de modos sin grilla (1.6, D9) ---------------------------------------

TOL_NUM = 1e-9


def _indices_de(lam_all: np.ndarray, conservar) -> np.ndarray:
    """Índices de `lam_all` que son (o son conjugados de) los representantes `conservar`."""
    conservar = np.asarray(conservar, complex).ravel()
    keep = np.zeros(len(lam_all), bool)
    for c in conservar:
        for target in (c, np.conj(c)):
            d = np.abs(lam_all - target)
            j = int(np.argmin(d))
            if d[j] <= 1e-8 * max(1.0, abs(target)):
                keep[j] = True
    return np.where(keep)[0]


def reconstruir(a: np.ndarray, conservar) -> np.ndarray:
    """`A` con solo los modos `conservar` (pares conjugados juntos), reconstrucción real."""
    lam, vecs = np.linalg.eig(a)
    w = np.linalg.inv(vecs)
    idx = _indices_de(lam, conservar)
    return np.real(vecs[:, idx] @ np.diag(lam[idx]) @ w[idx, :])


def operador_con_modos(op, conservar):
    """Copia del `Operador` con solo los modos `conservar` (lista de `[re, im]` o complejos)."""
    from .estimadores import Operador

    cons = np.asarray([complex(c[0], c[1]) if isinstance(c, (list, tuple)) else complex(c) for c in conservar])
    meta = {k: v for k, v in op.meta.items() if k != "eig_bloques"}
    meta["modos_conservados"] = np.stack([cons.real, cons.imag], axis=1) if len(cons) else np.zeros((0, 2))
    return Operador(A=reconstruir(op.A, cons), U=op.U, rank=op.rank, estimador=f"{op.estimador}+modos",
                    pasos=op.pasos, meta=meta)


def a_lista(lam) -> list[list[float]]:
    lam = np.asarray(lam, complex)
    return [[float(x.real), float(x.imag)] for x in lam]


def eliminacion_atras(reps, evaluar_fn, tol: float = TOL_NUM) -> tuple[list, pd.DataFrame]:
    """Quita de a un modo (representante; su conjugado va junto) el de menor empeoramiento del puntaje
    mientras el intervalo de `nuevo − actual` no quede entero sobre 0 (D9); se detiene en el primero
    significativo. `evaluar_fn(conservar) -> Puntaje`. Devuelve `(conservados, curva)`.
    """
    from .evaluacion import diferencia, intervalo

    actual = list(np.asarray(reps, complex))
    p_act = evaluar_fn(actual)
    lo, hi = intervalo(p_act.media_boot)
    curva = [{"n_modos": len(actual), "wrmse": p_act.media, "lo": lo, "hi": hi, "quitado_re": np.nan,
              "quitado_im": np.nan, "diff": 0.0, "diff_lo": 0.0, "diff_hi": 0.0, "aceptado": True}]
    while len(actual) > 1:
        mejor = None
        for i, lam in enumerate(actual):
            cand = actual[:i] + actual[i + 1:]
            p = evaluar_fn(cand)
            d = diferencia(p, p_act)
            if mejor is None or d["diff"] < mejor[1]["diff"]:
                mejor = (i, d, p, cand)
        i, d, p, cand = mejor
        signif = d["lo"] > tol
        lo, hi = intervalo(p.media_boot)
        curva.append({"n_modos": len(cand), "wrmse": p.media, "lo": lo, "hi": hi, "quitado_re": actual[i].real,
                      "quitado_im": actual[i].imag, "diff": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"],
                      "aceptado": not signif})
        if signif:
            break
        actual, p_act = cand, p
    return actual, pd.DataFrame(curva)
