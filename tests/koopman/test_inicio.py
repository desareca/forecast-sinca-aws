"""Tarea 4.1: arranque en caliente e init de K."""
import math

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from sint_koopman import contexto_y_posiciones, modelo_chico
from koopman import inicio as I
from koopman import modelo as M


def _ae_v2(hidden=8):
    from ae import data as aed
    from ae import models as aeml

    cfg = aeml.AEConfig(arch="gru", length=24, d=16, hidden=hidden, seed=0)
    return aeml.build_models(cfg, aed.N_MP, aed.N_AUX, aed.AUX_INPUT_DIM, aed.CAL_DIM)


def test_estado_oculto_inicial_igual_a_v2(grilla_kp):
    import keras

    grid, _ = grilla_kp
    ctx, pos, _ = contexto_y_posiciones(grid, 40)
    ae = _ae_v2()
    m = modelo_chico()
    I.encoder_desde_v2(m, ae.encoder)
    x = M.entradas(ctx.ng, pos)
    sub = ae.encoder.get_layer("encoder_mp")
    gru_v2 = next(l for l in sub.layers if isinstance(l, keras.layers.GRU))
    h_v2 = keras.Model(sub.input, gru_v2.output)(x[..., :M.N_MP])
    h_k = keras.Model(m.enc.input, m.enc.get_layer("enc_gru").output)(x)
    np.testing.assert_allclose(np.asarray(h_k), np.asarray(h_v2), atol=1e-6)
    w = m.enc.get_layer("enc_z").get_weights()[0]
    assert w.shape == (8, 10)
    I.decoder_lineal_ls(m, ctx.ng, pos)
    assert np.all(np.isfinite(m.kernel_decoder()))


def test_asignacion_de_autovalores_a_bloques():
    def c(h, per):
        return 0.5 ** (1 / h) * np.exp(1j * 2 * math.pi / per)

    lam = [c(300, 24.2), c(200, 480), c(100, 30), c(50, 12), 0.5 ** (1 / 400), 0.5 ** (1 / 80),
           0.9999995, 0.5 ** (1 / 20), 0.5 ** (1 / 10)]
    lam = np.array(lam + [np.conj(x) for x in lam if abs(np.imag(x)) > 0])
    out = I.asignar_autovalores(lam, 10)
    assert abs(out["diario"] - c(300, 24.2)) < 1e-12
    assert [abs(x) for x in out["libres"]] == sorted([abs(c(200, 480)), abs(c(100, 30))], reverse=True)
    assert len(out["reales"]) == 4 and abs(out["reales"][0] - 0.9999995) < 1e-12
    m = modelo_chico()
    I.init_k(m, lam)
    rho = np.asarray(m.op.rho())
    assert rho.max() <= M.RHO_MAX + 1e-7  # techo aplicado al real 0.9999995
    hl = math.log(0.5) / math.log(rho[0])
    assert hl == pytest.approx(300, rel=1e-3)
    k = m.op.matriz_np()
    assert 2 * math.pi / math.atan2(k[1, 0], k[0, 0]) == pytest.approx(24.0, abs=1e-3)  # diario fijo


@pytest.mark.parametrize("variante", ["", "kden"])
def test_reescalado_por_bloque_neutro(grilla_kp, variante):
    import tensorflow as tf

    from koopman import perdidas as L

    grid, _ = grilla_kp
    ctx, pos, _ = contexto_y_posiciones(grid, 60)
    # hidden ≥ d: con hidden < d el latente es de rango deficiente y el LS de mínima norma queda mal
    # condicionado en float32 (no ocurre en los jobs: hidden = 64, d = 10)
    m = M.KoopmanModelo(variante=variante, hidden=16, seed=0)
    rng = np.random.default_rng(0)
    m.op.B.assign(rng.normal(scale=0.1, size=m.op.B.shape).astype("float32"))
    I.decoder_lineal_ls(m, ctx.ng, pos)
    x = tf.constant(M.entradas(ctx.ng, pos))
    U = tf.constant(rng.normal(size=(len(pos), 24, M.N_U)).astype("float32"))

    def salidas():
        z = m.encode(x)
        return np.asarray(m.decode(z)), np.asarray(m.decode(m.rollout(z, U, 24)[-1]))

    r0, p0 = salidas()
    k0 = m.op.matriz_np().copy()
    s = I.reescalar_bloques(m)
    r1, p1 = salidas()
    assert not np.allclose(s, 1.0)
    np.testing.assert_allclose(r1, r0, rtol=1e-5, atol=1e-5 * np.abs(r0).max())
    np.testing.assert_allclose(p1, p0, rtol=1e-5, atol=1e-5 * np.abs(p0).max())
    assert float(L.l_norma(m)) == pytest.approx(0.0, abs=1e-6)
    if not variante:
        np.testing.assert_array_equal(m.op.matriz_np(), k0)


def test_reescalado_sin_efecto_con_decoder_gru():
    m = modelo_chico(decoder="gru")
    w0 = [w.copy() for w in m.enc.get_weights() + m.dec.get_weights()]
    assert np.all(I.reescalar_bloques(m) == 1.0)
    for a, b in zip(w0, m.enc.get_weights() + m.dec.get_weights()):
        np.testing.assert_array_equal(a, b)
