"""Arranque del Koopman (`deep-koopman`, D5–D7).

- GRU en caliente desde `encoder.keras` v2 (`enc_mp`): kernel de entrada de MP, kernel recurrente y
  sesgo copiados; filas de meteo y máscaras en 0 (el estado oculto inicial coincide con v2); Dense
  final = `W_v2·V_d` (las `d` direcciones principales de la Dense 64→12 de v2).
- Decoder lineal por mínimos cuadrados del latente inicial de train a las ventanas MP.
- `K` desde los autovalores del `operator.npz` de `e15a` (7.1) con la regla de D7, o genérico (conv1d).
"""
from __future__ import annotations

import math

import numpy as np

from .modelo import (HORIZONTE, N_MP, RHO_MAX, KoopmanModelo, a_desde_rho, b_desde_theta, entradas, estructura)

TOL_REAL = 0.005
PERIODO_DIARIO = 24.0


def _gru_y_dense(sub):
    from keras import layers

    gru = next(l for l in sub.layers if isinstance(l, layers.GRU))
    dense = next(l for l in sub.layers if isinstance(l, layers.Dense))
    return gru, dense


def encoder_desde_v2(modelo: KoopmanModelo, encoder_v2) -> None:
    """Copia `enc_mp` de un AE v2 (GRU) al encoder GRU del Koopman (ver docstring del módulo)."""
    if modelo.config["encoder"] != "gru":
        raise ValueError("El arranque en caliente solo aplica al encoder GRU")
    sub = encoder_v2.get_layer("encoder_mp")
    gru_v2, dense_v2 = _gru_y_dense(sub)
    k_v2, rk_v2, b_v2 = gru_v2.get_weights()
    gru = modelo.enc.get_layer("enc_gru")
    k, rk, b = gru.get_weights()
    if rk.shape != rk_v2.shape:
        raise ValueError(f"GRU de tamaño distinto: v2 {rk_v2.shape}, Koopman {rk.shape}")
    k_new = np.zeros_like(k)
    k_new[:N_MP] = k_v2
    gru.set_weights([k_new, rk_v2, b_v2])
    w_v2, bias_v2 = dense_v2.get_weights()  # h × 12, 12
    _, _, vt = np.linalg.svd(w_v2, full_matrices=True)  # vt: 12 × 12 aunque h < 12
    v_d = vt[: modelo.d].T  # 12 × d
    modelo.enc.get_layer("enc_z").set_weights([w_v2 @ v_d, bias_v2 @ v_d])


def decoder_lineal_ls(modelo: KoopmanModelo, ng, positions, lote: int = 4096) -> None:
    """Decoder lineal por mínimos cuadrados (con sesgo) del latente inicial a las ventanas MP."""
    from .modelo import ventanas_mp

    positions = np.asarray(positions, dtype=int)
    zs, ys = [], []
    for i in range(0, len(positions), lote):
        p = positions[i:i + lote]
        zs.append(np.asarray(modelo.encode(entradas(ng, p)), dtype=float))
        ys.append(ventanas_mp(ng, p)[0].reshape(len(p), -1).astype(float))
    z, y = np.concatenate(zs), np.concatenate(ys)
    x = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    modelo.dec.get_layer("dec_lin").set_weights([coef[:-1].astype("float32"), coef[-1].astype("float32")])


def reescalar_bloques(modelo: KoopmanModelo) -> np.ndarray:
    """Reescalado por bloque tras el ajuste del decoder (D6, 4.2): `s_b = ‖W_b‖_F/√n_b`; columnas del
    decoder `/ s_b`, salida del encoder (kernel y sesgo de `enc_z`) y filas de `B` `· s_b`. `K` no cambia
    (escalar dentro del bloque); en `kden` se aplica `K ← S K S⁻¹`. Predicciones idénticas y `L_norma = 0`.
    Sin efecto con decoder GRU. Devuelve `s` por coordenada de `z`."""
    from .modelo import bloques

    s = np.ones(modelo.d)
    if modelo.config["decoder"] != "lineal":
        return s
    dec = modelo.dec.get_layer("dec_lin")
    W, bd = dec.get_weights()  # W: d × 144 (filas = coordenadas de z)
    for b in bloques(modelo.d):
        s[b] = np.sqrt(np.sum(W[b].astype(float) ** 2) / len(b))
    s = np.where(s > 1e-12, s, 1.0)
    dec.set_weights([(W / s[:, None]).astype("float32"), bd])
    ez = modelo.enc.get_layer("enc_z")
    k, be = ez.get_weights()  # k: hidden × d
    ez.set_weights([(k * s[None, :]).astype("float32"), (be * s).astype("float32")])
    op = modelo.op
    if op.B is not None:
        op.B.assign((np.asarray(op.B) * s[:, None]).astype("float32"))
    if op.kden is not None:
        op.kden.assign((s[:, None] * np.asarray(op.kden) / s[None, :]).astype("float32"))
    return s


def asignar_autovalores(lam, d: int = 10) -> dict:
    """Regla de D7: diario ← el más cercano a 24 h; libres ← los complejos no diarios de mayor `|λ|`;
    reales ← los reales (o `|arg λ| < 0.005`) de mayor `|λ|`; todo recortado al techo. Faltantes:
    valores genéricos. Devuelve `{"a", "b", "diario", "libres", "reales"}`."""
    from .modelo import OperadorBloques

    lam = np.asarray(lam, dtype=complex)
    rep = lam[lam.imag >= -1e-12]
    ang = np.abs(np.angle(rep))
    n_lib, n_real = estructura(d)
    gen = OperadorBloques.parametros_genericos(type("x", (), {"n_lib": n_lib, "n_real": n_real})())
    comp = np.where(ang >= TOL_REAL)[0]
    reales = np.where(ang < TOL_REAL)[0]
    with np.errstate(divide="ignore"):
        per = np.where(ang > 0, 2 * math.pi / np.where(ang > 0, ang, 1), np.inf)
    a = list(gen["a"])
    b = list(gen["b"])
    diario = libres = None
    usados = set()
    if len(comp):
        j = comp[np.argmin(np.abs(per[comp] - PERIODO_DIARIO))]
        diario = rep[j]
        usados.add(int(j))
        a[0] = a_desde_rho(min(abs(diario), RHO_MAX))
    otros = [int(j) for j in comp[np.argsort(-np.abs(rep[comp]), kind="stable")] if int(j) not in usados][:n_lib]
    for i, j in enumerate(otros):
        a[1 + i] = a_desde_rho(min(abs(rep[j]), RHO_MAX))
        b[i] = b_desde_theta(ang[j])
    rs = [int(j) for j in reales[np.argsort(-np.abs(rep[reales]), kind="stable")]][:n_real]
    for i, j in enumerate(rs):
        a[1 + n_lib + i] = a_desde_rho(min(abs(rep[j]), RHO_MAX))
    return {"a": a, "b": b, "diario": diario, "libres": [rep[j] for j in otros], "reales": [rep[j] for j in rs]}


def init_k(modelo: KoopmanModelo, lam=None) -> dict:
    """`K` desde autovalores de 7.1 (GRU) o genérico (`lam=None`, conv1d). No aplica a `kden`
    (que parte del `K` estructurado equivalente)."""
    if lam is None:
        params = modelo.op.parametros_genericos()
        info = {"origen": "generico"}
    else:
        params = asignar_autovalores(lam, modelo.d)
        info = {"origen": "e15a"}
    modelo.op.asignar(a=params["a"], b=params["b"])
    if modelo.op.kden is not None:
        import tensorflow as tf

        modelo.op.kden.assign(tf.constant(_estructurado(modelo).astype("float32")))
    return {**info, "a": [float(x) for x in params["a"]], "b": [float(x) for x in params["b"]]}


def _estructurado(modelo: KoopmanModelo) -> np.ndarray:
    kden, modelo.op.kden = modelo.op.kden, None
    try:
        return modelo.op.matriz_np()
    finally:
        modelo.op.kden = kden
