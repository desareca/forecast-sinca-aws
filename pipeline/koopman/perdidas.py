"""Pérdidas del Koopman entrenado (`deep-koopman`, D8).

Todas operan sobre un lote (dict de tensores, ver `entrenar.Datos.lote`):
`x0` entrada en `t`, `x1` entrada en `t+1` y `v1` (par válido), `mp`/`real` ventanas MP desde `t`
hasta `t+24` (`[N, 25, 24, 6]`: índice `k` = ventana que termina en `t+k`), `U` (`u_{t+1…t+24}`),
`y`/`ym` target 24 h en µg/m³ y su máscara, `w` pesos de la métrica, `iz`/`izw` ICAP_zona real y su peso.

`L_pred(H)` promedia sobre `k = 1…H` (no suma) para que la escala no dependa del currículo.
"""
from __future__ import annotations

import numpy as np

from evaluate.evaluate import ALPHA_DEFAULT, regular_threshold
from icap.icap import ICAP_ANCHORS

from .modelo import HORIZONTE, N_MP, bloques

TAU_ICAP = 5.0
LAMBDAS = {"rec": 1.0, "lin": 1.0, "pred": 1.0, "tgt": 1.0, "norma": 1.0, "cons": 1.0, "l1": 0.1}
SIN_NORMALIZAR = ("norma",)  # adimensional y parte en 0 (D8)


def _pollutants() -> list[str]:
    from ae import data as aed

    return [p for _, _, p in aed.mp_channels()]


def masked_mse(y_true, y_pred, mask):
    """MSE enmascarado y promediado por canal (como `ae.models.masked_channel_mse`), media del lote."""
    import tensorflow as tf

    num = tf.reduce_sum(mask * tf.square(y_true - y_pred), axis=1)
    den = tf.maximum(tf.reduce_sum(mask, axis=1), 1e-6)
    return tf.reduce_mean(num / den)


def l_rec(modelo, lote, training=True):
    z = modelo.encode(lote["x0"], training=training)
    return masked_mse(lote["mp"][:, 0], modelo.decode(z, training=training), lote["real"][:, 0]), z


def l_lin(modelo, lote, z0, training=True):
    import tensorflow as tf

    z1 = modelo.encode(lote["x1"], training=training)
    pred = modelo.paso(z0, lote["U"][:, 0, :])
    err = tf.reduce_mean(tf.square(z1 - pred), axis=1)
    v = lote["v1"]
    return tf.reduce_sum(v * err) / tf.maximum(tf.reduce_sum(v), 1.0)


def l_pred(modelo, lote, z0, H: int, training=True):
    import tensorflow as tf

    zs = modelo.rollout(z0, lote["U"], H)
    terms = [masked_mse(lote["mp"][:, k + 1], modelo.decode(zs[k], training=training), lote["real"][:, k + 1])
             for k in range(H)]
    return tf.add_n(terms) / float(H), zs


def promedio_ugm3(modelo, z24, mean, std, training=True):
    import tensorflow as tf

    lp = modelo.decode(z24, training=training) * std + mean
    return tf.reduce_mean(tf.maximum(tf.math.expm1(lp), 0.0), axis=1)  # [N, 6]


def l_tgt(pred, y, ym, w):
    """RMSE ponderado (pesos de la métrica) del promedio 24 h en µg/m³ sobre los targets presentes."""
    import tensorflow as tf

    ww = w * ym
    return tf.sqrt(tf.reduce_sum(ww * tf.square(pred - y)) / tf.maximum(tf.reduce_sum(ww), 1e-6))


def icap_tf(c, pollutant: str):
    import tensorflow as tf

    (c0, i0), (c1, i1), (c2, i2) = ICAP_ANCHORS[pollutant]
    c = tf.maximum(c, 0.0)
    tramo1 = i0 + (c - c0) * (i1 - i0) / (c1 - c0)
    tramo2 = i1 + (c - c1) * (i2 - i1) / (c2 - c1)
    return tf.where(c <= c1, tramo1, tf.where(c <= c2, tramo2, tf.fill(tf.shape(c), float(i2))))


def icap_zona_suave(pred, tau: float = TAU_ICAP):
    """ICAP_zona con máximo suave `τ·logsumexp(ICAP_s/τ)` sobre las 6 series (`pred [N, 6]` µg/m³)."""
    import tensorflow as tf

    pols = _pollutants()
    icaps = tf.stack([icap_tf(pred[:, s], pols[s]) for s in range(N_MP)], axis=1)
    return tau * tf.reduce_logsumexp(icaps / tau, axis=1)


def l_tgt_icap(pred, iz, izw, tau: float = TAU_ICAP):
    """RMSE ponderado (`1 + max(0, ICAP − 100)`) del ICAP_zona suave vs el real (filas con las 6 series)."""
    import tensorflow as tf

    soft = icap_zona_suave(pred, tau)
    return tf.sqrt(tf.reduce_sum(izw * tf.square(soft - iz)) / tf.maximum(tf.reduce_sum(izw), 1e-6))


def l_norma(modelo):
    """`Σ_b (‖W_b‖_F² / n_b − 1)²` sobre las columnas del decoder lineal de cada bloque de K (D6):
    fija la escala de `z` por bloque (evita el colapso `z → 0`); 0 con decoder GRU."""
    import tensorflow as tf

    if modelo.config["decoder"] != "lineal":
        return tf.constant(0.0)
    k = modelo.dec.get_layer("dec_lin").kernel  # d × 144 (Keras) = Wᵀ
    sq = tf.reduce_sum(tf.square(k), axis=1)  # ‖columna de W‖² por coordenada de z
    return tf.add_n([tf.square(tf.reduce_sum(tf.gather(sq, b)) / float(len(b)) - 1.0) for b in bloques(modelo.d)])


def l_cons(modelo, lote, z0, training=True):
    """B2: `‖enc([dec(z), meteo, máscaras]) − z‖²`."""
    import tensorflow as tf

    rec = modelo.decode(z0, training=training)
    x = tf.concat([rec, lote["x0"][:, :, N_MP:]], axis=-1)
    return tf.reduce_mean(tf.reduce_sum(tf.square(modelo.encode(x, training=training) - z0), axis=1))


def l_l1(modelo, z0):
    """B3: `mean_t Σ_bloques ‖z_bloque‖₂`."""
    import tensorflow as tf

    norms = [tf.norm(tf.gather(z0, b, axis=1) + 1e-12, axis=1) for b in bloques(modelo.d)]
    return tf.reduce_mean(tf.add_n(norms))


def terminos(modelo, lote, cfg: dict, escala_mp, training=True) -> dict:
    """Términos crudos activos según `cfg` (`fase`, `H`, `icap`, `cons`, `l1`).

    `H` puede ser un tensor (paso compilado, sin retrazar por currículo): entonces `H_max` (int) fija
    el desenrollado y los pasos `k ≥ H` se anulan con una máscara; el valor es el mismo que con `H` int."""
    import tensorflow as tf

    out = {}
    out["rec"], z0 = l_rec(modelo, lote, training)
    out["lin"] = l_lin(modelo, lote, z0, training)
    if modelo.config["decoder"] == "lineal":
        out["norma"] = l_norma(modelo)
    if cfg.get("cons"):
        out["cons"] = l_cons(modelo, lote, z0, training)
    if cfg.get("l1"):
        out["l1"] = l_l1(modelo, z0)
    H = cfg.get("H", 0)
    dinamico = tf.is_tensor(H)
    H_max = int(cfg.get("H_max", HORIZONTE)) if dinamico else int(H)
    if H_max > 0 or cfg.get("tgt"):
        zs = modelo.rollout(z0, lote["U"], HORIZONTE if cfg.get("tgt") else H_max)
        if H_max > 0:
            terms = [masked_mse(lote["mp"][:, k + 1], modelo.decode(zs[k], training=training), lote["real"][:, k + 1])
                     for k in range(H_max)]
            if dinamico:
                Hf = tf.cast(H, tf.float32)
                terms = [tf.cast(k < Hf, tf.float32) * t for k, t in enumerate(terms)]
                out["pred"] = tf.add_n(terms) / tf.maximum(Hf, 1.0)
            else:
                out["pred"] = tf.add_n(terms) / float(H_max)
        if cfg.get("tgt"):
            mean, std = escala_mp
            pred = promedio_ugm3(modelo, zs[HORIZONTE - 1], mean, std, training)
            out["tgt"] = l_tgt_icap(pred, lote["iz"], lote["izw"]) if cfg.get("icap") else \
                l_tgt(pred, lote["y"], lote["ym"], lote["w"])
    return {k: tf.cast(v, tf.float32) for k, v in out.items()}


def total(terms: dict, escalas: dict, lambdas: dict):
    """`Σ λ_k · L_k / escala_k` (`L_norma` no se normaliza: adimensional y parte en 0)."""
    import tensorflow as tf

    acc = []
    for k, v in terms.items():
        lam = float(lambdas.get(k, LAMBDAS.get(k, 1.0)))
        den = 1.0 if k in SIN_NORMALIZAR else max(float(escalas.get(k, 1.0)), 1e-12)
        acc.append(lam * v / den)
    return tf.add_n(acc)


# --- Pesos de la métrica (numpy) ----------------------------------------------------


def pesos_metrica(y: np.ndarray, alpha: float = ALPHA_DEFAULT) -> np.ndarray:
    """`w = 1 + α·max(0, y − umbral ICAP100)` por serie (`y [N, 6]`; NaN -> 1)."""
    pols = _pollutants()
    thr = np.array([regular_threshold(p) for p in pols])
    return 1.0 + alpha * np.maximum(0.0, np.nan_to_num(y, nan=-np.inf) - thr[None, :])


def icap_zona_real(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ICAP_zona real (máximo de las 6 series) y su peso `1 + max(0, ICAP − 100)`; 0 si falta alguna serie."""
    pols = _pollutants()
    icaps = np.stack([np.interp(np.nan_to_num(y[:, s], nan=0.0), [a[0] for a in ICAP_ANCHORS[p]],
                                [a[1] for a in ICAP_ANCHORS[p]], left=0.0, right=500.0)
                      for s, p in enumerate(pols)], axis=1)
    iz = icaps.max(axis=1)
    completo = np.isfinite(y).all(axis=1)
    return iz.astype("float32"), np.where(completo, 1.0 + np.maximum(0.0, iz - 100.0), 0.0).astype("float32")
