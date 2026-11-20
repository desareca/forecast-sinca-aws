"""Identificabilidad de modos del Koopman (`deep-koopman`, D11, D13). Preregistro:

- Contribución del bloque `j` en `t`: parte del promedio de 24 h decodificado (espacio log1p normalizado)
  de `z_{t+24|t}` que viene del bloque `j` (decoder lineal: exacta; GRU: jacobiano del decoder en
  `z_{t+24|t}`). `B·u` va aparte (ya está dentro de `z_{t+24|t}` y se reparte por bloque).
- Aporte `A_j = Σ_t Σ_s w_s(t)·c_{j,s}(t)²` en `sel`; dominantes = mínimo conjunto (orden de `A_j`)
  con ≥ 80 % de `Σ A_j`. El diario no se exige como dominante.
- Bootstrap: DMD exacto sobre el latente de train, bloques de 7 días, B = 200, emparejado con `K`.
- Compuerta: cada dominante de cada semilla tiene pareja en las otras dos (húngaro, ±25 %; reales solo
  vida media), aparece en ≥ 80 % del bootstrap de cada semilla y sus intervalos se solapan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from operador import estimadores as estim
from operador import modos

from .modelo import HORIZONTE, KoopmanModelo, bloques, entradas, tipos_bloque, u_futuro

UMBRAL_DOMINANTES = 0.8
APARICION_MIN = 0.8


def z_horizonte(modelo: KoopmanModelo, ng, positions, lote: int = 4096) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    out = []
    for i in range(0, len(positions), lote):
        p = positions[i:i + lote]
        z = modelo.encode(entradas(ng, p))
        out.append(np.asarray(modelo.rollout(z, u_futuro(ng, p))[-1], dtype=float))
    return np.concatenate(out) if out else np.zeros((0, modelo.d))


def jacobiano_promedio(modelo: KoopmanModelo, z: np.ndarray) -> np.ndarray:
    """`∂ mean_24h(dec(z)) / ∂z` por fila: `[N, 6, d]`."""
    import tensorflow as tf

    zt = tf.constant(np.asarray(z, dtype="float32"))
    with tf.GradientTape() as tape:
        tape.watch(zt)
        y = tf.reduce_mean(modelo.decode(zt), axis=1)
    return np.asarray(tape.batch_jacobian(y, zt), dtype=float)


def contribuciones(modelo: KoopmanModelo, z: np.ndarray) -> np.ndarray:
    """`c [N, n_bloques, 6]` del promedio de 24 h por bloque de `z`."""
    bl = bloques(modelo.d)
    if modelo.config["decoder"] == "lineal":
        w = modelo.kernel_decoder().reshape(modelo.d, HORIZONTE, -1).mean(axis=1)  # d × 6
        return np.stack([z[:, b] @ w[b] for b in bl], axis=1)
    jac = jacobiano_promedio(modelo, z)
    return np.stack([np.einsum("nsd,nd->ns", jac[:, :, b], z[:, b]) for b in bl], axis=1)


def aportes(c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`A_j = Σ_t Σ_s w_s(t)·c_{j,s}(t)²` (`w [N, 6]`, 0 donde la serie no está)."""
    return np.einsum("ns,njs->j", np.asarray(w, float), np.asarray(c, float) ** 2)


def dominantes(a: np.ndarray, umbral: float = UMBRAL_DOMINANTES) -> list[int]:
    a = np.asarray(a, float)
    orden = np.argsort(-a, kind="stable")
    acc = np.cumsum(a[orden]) / max(a.sum(), 1e-300)
    n = int(np.searchsorted(acc, umbral - 1e-12) + 1)
    return [int(i) for i in orden[:n]]


def representantes_bloques(modelo: KoopmanModelo) -> np.ndarray:
    """Autovalor representante por bloque (`Im ≥ 0`); en `kden`, los representantes de `eig(K)`."""
    k = modelo.op.matriz_np().astype(float)
    if modelo.op.kden is not None:
        lam = np.linalg.eigvals(k)
        return lam[modos.representantes(lam)]
    out = []
    for b in bloques(modelo.d):
        lam = np.linalg.eigvals(k[np.ix_(b, b)])
        out.append(lam[np.argmax(lam.imag)] if len(b) == 2 else complex(lam[0]))
    return np.asarray(out, dtype=complex)


def tabla_modos(modelo: KoopmanModelo, a: np.ndarray, dom: list[int]) -> pd.DataFrame:
    reps = representantes_bloques(modelo)
    tipos = tipos_bloque(modelo.d) if modelo.op.kden is None else ["?"] * len(reps)
    total = max(float(np.sum(a)), 1e-300)
    return pd.DataFrame({
        "bloque": np.arange(len(reps)), "tipo": tipos[:len(reps)], "lam_re": reps.real, "lam_im": reps.imag,
        "periodo_h": modos.periodo_h(reps), "vida_media_h": modos.vida_media_h(reps),
        "aporte": a, "aporte_frac": np.asarray(a) / total, "dominante": [i in dom for i in range(len(reps))],
    })


# --- Latente, bootstrap y coherencia ----------------------------------------------------


def latente(modelo: KoopmanModelo, ng, positions, lote: int = 4096) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    return np.concatenate([np.asarray(modelo.encode(entradas(ng, positions[i:i + lote])), float)
                           for i in range(0, len(positions), lote)])


def segmentos(z: np.ndarray, positions) -> tuple[list[np.ndarray], list[int]]:
    positions = np.asarray(positions, dtype=int)
    breaks = np.where(np.diff(positions) != 1)[0] + 1
    segs = np.split(z, breaks)
    return segs, [len(s) for s in segs]


def bootstrap_latente(modelo: KoopmanModelo, z: np.ndarray, positions, dias: np.ndarray, B: int = 200,
                      seed: int = 0) -> pd.DataFrame:
    """Re-estima DMD exacto sobre el latente de train (pesos = conteos de bloques de 7 días) y empareja
    sus modos con los de `K`."""
    segs, lens = segmentos(z, positions)
    cortes = np.cumsum(lens)[:-1]

    def fit(w_rows):
        return estim.ponderado(segs, list(np.split(np.asarray(w_rows, float), cortes)), None).eigvals()

    return modos.bootstrap_modos(modelo.op.autovalores(), fit, dias, B=B, dias=7, seed=seed)


def coherencia(modelo: KoopmanModelo, z: np.ndarray, positions) -> pd.DataFrame:
    """Autovalores de `K` y del DMD exacto sobre el latente de train, emparejados (húngaro)."""
    segs, _ = segmentos(z, positions)
    lam_k = modelo.op.autovalores()
    lam_k = lam_k[modos.representantes(lam_k)]
    lam_d = estim.exacto(segs, None).eigvals()
    lam_d = lam_d[modos.representantes(lam_d)]
    i, j = modos.emparejar(lam_k, lam_d)
    return pd.DataFrame({
        "k_re": lam_k[i].real, "k_im": lam_k[i].imag, "dmd_re": lam_d[j].real, "dmd_im": lam_d[j].imag,
        "periodo_k_h": modos.periodo_h(lam_k[i]), "periodo_dmd_h": modos.periodo_h(lam_d[j]),
        "vida_media_k_h": modos.vida_media_h(lam_k[i]), "vida_media_dmd_h": modos.vida_media_h(lam_d[j]),
        "aparece": modos.aparece(lam_k[i], lam_d[j]),
    })


# --- Compuerta de identificabilidad ---------------------------------------------------


def _fila_boot(boot: pd.DataFrame, lam: complex) -> pd.Series:
    d = np.abs(boot["lam_re"].to_numpy() + 1j * boot["lam_im"].to_numpy() - lam)
    return boot.iloc[int(np.argmin(d))]


def _solapan(b0: pd.Series, b1: pd.Series, real: bool) -> bool:
    ok = True
    for col in ("vida_media_h",) + (() if real else ("periodo_h",)):
        lo0, hi0, lo1, hi1 = (float(b0[f"{col}_lo"]), float(b0[f"{col}_hi"]), float(b1[f"{col}_lo"]), float(b1[f"{col}_hi"]))
        if np.isfinite([lo0, hi0, lo1, hi1]).all():
            ok &= lo0 <= hi1 and lo1 <= hi0
    return bool(ok)


def compuerta(semillas: dict) -> tuple[bool, pd.DataFrame]:
    """`semillas = {seed: {"dominantes": [λ…], "modos": [λ de todos los bloques], "boot": DataFrame}}`.

    Aprueba si cada dominante de cada semilla tiene pareja en las otras (húngaro sobre sus modos),
    aparece en ≥ 80 % de su bootstrap y sus intervalos se solapan con los de la pareja."""
    seeds = sorted(semillas)
    filas = []
    for s in seeds:
        info = semillas[s]
        for lam in np.asarray(info["dominantes"], complex):
            real = abs(lam.imag) <= modos.TOL_IMAG
            b0 = _fila_boot(info["boot"], lam)
            fila = {"semilla": s, "lam_re": lam.real, "lam_im": lam.imag,
                    "periodo_h": float(modos.periodo_h([lam])[0]), "vida_media_h": float(modos.vida_media_h([lam])[0]),
                    "aparicion": float(b0["aparicion"])}
            pareja, solapa = True, True
            for s2 in seeds:
                if s2 == s:
                    continue
                otros = np.asarray(semillas[s2]["modos"], complex)
                i, j = modos.emparejar(np.array([lam]), otros)
                ok = bool(len(j) and modos.aparece(np.array([lam]), otros[j])[0])
                fila[f"pareja_s{s2}"] = ok
                pareja &= ok
                if ok:
                    solapa &= _solapan(b0, _fila_boot(semillas[s2]["boot"], otros[j[0]]), real)
            fila.update({"pareja": pareja, "boot_80": fila["aparicion"] >= APARICION_MIN, "solapan": solapa})
            fila["ok"] = bool(pareja and fila["boot_80"] and solapa)
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    return bool(len(tabla) and tabla["ok"].all()), tabla


# --- Eliminación de bloques (D13) ------------------------------------------------------


def eliminacion_bloques(modelo: KoopmanModelo, puntaje_fn) -> tuple[list[int], pd.DataFrame]:
    """Regla 1.6 enmendada sobre los bloques de `K` (diario protegido). `puntaje_fn(conservar_idx)` ->
    `Puntaje`. Devuelve los índices de bloque conservados y la curva."""
    reps = list(representantes_bloques(modelo))

    def a_indices(conservar) -> list[int]:
        libres = list(range(len(reps)))
        out = []
        for lam in conservar:
            j = min(libres, key=lambda i: abs(reps[i] - lam))
            libres.remove(j)
            out.append(j)
        return sorted(out)

    kept, curva = modos.eliminacion_atras(reps, lambda c: puntaje_fn(a_indices(c)), protegidos=[reps[0]])
    return a_indices(kept), curva
