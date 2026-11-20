"""Tarea 3.1: pérdidas."""
import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from sint_koopman import contexto_y_posiciones, modelo_chico
from evaluate.evaluate import weighted_rmse, zone_icap
from koopman import entrenar as T
from koopman import perdidas as L
from operador import evaluacion as V


class _Lineal:
    """Modelo lineal exacto: `z` va en la última hora de la entrada; `dec(z) = z·W`."""

    def __init__(self, K, W):
        self.K, self.W = tf.constant(K), tf.constant(W)

    def encode(self, x, training=False):
        return x[:, -1, :self.K.shape[0]]

    def decode(self, z, training=False):
        return tf.reshape(tf.matmul(z, self.W), (-1, 24, 6))

    def paso(self, z, u):
        return tf.matmul(z, self.K, transpose_b=True)

    def rollout(self, z0, U, pasos=24):
        out, z = [], z0
        for k in range(pasos):
            z = self.paso(z, U[:, k])
            out.append(z)
        return out


def test_modelo_lineal_exacto_lin_y_pred_cero():
    rng = np.random.default_rng(0)
    d, N = 4, 16
    K = (np.diag([0.9, 0.8, 0.7, 0.95]) + 0.01 * rng.normal(size=(d, d))).astype("float32")
    W = rng.normal(size=(d, 144)).astype("float32")
    z = np.zeros((N, 26, d), dtype="float32")
    z[:, 0] = rng.normal(size=(N, d))
    for k in range(1, 26):
        z[:, k] = z[:, k - 1] @ K.T
    x0 = np.zeros((N, 24, 30), "float32"); x0[:, -1, :d] = z[:, 0]
    x1 = np.zeros((N, 24, 30), "float32"); x1[:, -1, :d] = z[:, 1]
    mp = np.stack([(z[:, k] @ W).reshape(N, 24, 6) for k in range(25)], axis=1)
    lote = {"x0": x0, "x1": x1, "v1": np.ones(N, "float32"), "mp": mp, "real": np.ones_like(mp),
            "U": np.zeros((N, 24, 4), "float32")}
    lote = {k: tf.constant(v) for k, v in lote.items()}
    m = _Lineal(K, W)
    z0 = m.encode(lote["x0"])
    assert float(L.l_lin(m, lote, z0)) < 1e-10
    assert float(L.l_pred(m, lote, z0, 24)[0]) < 1e-8


def test_escalas_iniciales_normalizadas(grilla_kp):
    grid, _ = grilla_kp
    ctx, pos, fin = contexto_y_posiciones(grid, 400)
    datos = T.Datos(grid, ctx.ng, pos, fin)
    m = modelo_chico()
    cfg = {"H": 6}
    esc_mp = tuple(tf.constant(a.astype("float32")) for a in (np.zeros(6), np.ones(6)))
    escalas = T._escalas(m, datos, cfg, esc_mp, L.LAMBDAS)
    idx = np.sort(np.random.default_rng(T.SEMILLA_ESCALA).choice(len(datos), size=len(datos), replace=False))
    t = L.terminos(m, {k: tf.constant(v) for k, v in datos.lote(idx).items()}, {"H": 24}, esc_mp, training=False)
    for k in ("rec", "lin", "pred"):
        assert abs(float(t[k]) / escalas[k] - 1.0) < 1e-3


def test_l_tgt_es_rmse_ponderado():
    rng = np.random.default_rng(1)
    y = rng.uniform(10, 200, size=(50, 6))
    pred = y + rng.normal(0, 20, size=y.shape)
    w = L.pesos_metrica(y)
    got = float(L.l_tgt(tf.constant(pred, tf.float32), tf.constant(y, tf.float32), tf.ones_like(tf.constant(y, tf.float32)),
                        tf.constant(w, tf.float32)))
    assert got == pytest.approx(weighted_rmse(y.ravel(), pred.ravel(), w.ravel()), rel=1e-5)


def test_icap_suave_tiende_al_icap_zona():
    rng = np.random.default_rng(2)
    pred = rng.uniform(0, 250, size=(20, 6))
    ts = pd.date_range("2026-05-01", periods=20, freq="h")
    frame = pd.concat([pd.DataFrame({"timestamp": ts, "estacion": s, "pollutant": p, "y_true": pred[:, c], "y_pred": pred[:, c]})
                       for c, (s, p) in enumerate(V.SERIES)], ignore_index=True)
    zona = zone_icap(frame).sort_values("timestamp")["icap_pred"].to_numpy()
    soft = np.asarray(L.icap_zona_suave(tf.constant(pred, tf.float32), tau=1e-3))
    np.testing.assert_allclose(soft, zona, atol=1e-2)
    iz, izw = L.icap_zona_real(pred)
    np.testing.assert_allclose(iz, zona, atol=1e-3)
    assert np.all(izw >= 1)
