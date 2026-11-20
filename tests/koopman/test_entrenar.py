"""Tarea 3.2: bucle de entrenamiento."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from sint_koopman import contexto_y_posiciones, modelo_chico
from koopman import entrenar as T


def _datos(grid, n=300):
    ctx, pos, fin = contexto_y_posiciones(grid, n)
    cols = [c for c, _, _ in T.aed.mp_channels()]
    esc = (ctx.norm.mean[cols].to_numpy(), ctx.norm.std[cols].to_numpy())
    return T.Datos(grid, ctx.ng, pos, fin), esc


def test_curriculo_por_epoca():
    cfg = T.FASES["f2"]
    assert [T.h_de_epoca(cfg, e) for e in (0, 9, 10, 19, 20, 29)] == [1, 1, 6, 6, 24, 24]
    assert T.h_de_epoca(T.FASES["f3"], 3) == 24 and T.h_de_epoca(T.FASES["f1"], 3) == 0
    assert T.lr_de_paso(0, 10) == pytest.approx(T.LR / 20) and T.lr_de_paso(19, 10) == pytest.approx(T.LR)


def test_encoder_congelado_en_calentamiento(grilla_kp):
    datos, esc = _datos(grilla_kp[0])
    m = modelo_chico()
    enc0 = [w.copy() for w in m.enc.get_weights()]
    dec0 = [w.copy() for w in m.dec.get_weights()]
    out = T.entrenar(m, datos, "f1", esc, en_caliente=True, epocas=2, log=lambda *_: None)
    for a, b in zip(enc0, m.enc.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(dec0, m.dec.get_weights()))
    assert all(f["congelado"] for f in out["historia"])


def test_cierre_solo_k_y_b(grilla_kp):
    datos, esc = _datos(grilla_kp[0])
    m = modelo_chico()
    enc0, dec0 = [w.copy() for w in m.enc.get_weights()], [w.copy() for w in m.dec.get_weights()]
    a0 = np.asarray(m.op.a).copy()
    T.entrenar(m, datos, "cierre", esc, epocas=1, cfg_perdida={"H": 6}, log=lambda *_: None)
    for a, b in zip(enc0 + dec0, m.enc.get_weights() + m.dec.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a0, np.asarray(m.op.a))


@pytest.mark.parametrize("fase", ["f2", "f3"])
def test_dos_epocas_sin_nan(grilla_kp, fase):
    datos, esc = _datos(grilla_kp[0])
    m = modelo_chico(seed=1)
    out = T.entrenar(m, datos, fase, esc, epocas=2, log=lambda *_: None)
    assert len(out["historia"]) == 2
    for fila in out["historia"]:
        assert all(np.isfinite(v) for k, v in fila.items() if k.startswith("loss_"))
    assert np.all(np.isfinite(m.op.autovalores()))


def test_orden_canonico_conserva_pronostico(grilla_kp):
    from koopman import modelo as M

    grid = grilla_kp[0]
    ctx, pos, _ = contexto_y_posiciones(grid, 100)
    m = modelo_chico(seed=3)
    m.op.a.assign(np.array([0.1, 2.0, 0.5, 3.0, 0.2, 1.0, 0.05], "float32"))
    m.op.B.assign(np.random.default_rng(0).normal(size=(10, 4)).astype("float32"))
    f0, _ = M.pronostico(m, ctx.ng, ctx.norm, pos[:20])
    T.orden_canonico(m)
    rho = np.asarray(m.op.rho())
    assert np.all(np.diff(rho[1:3]) <= 0) and np.all(np.diff(rho[3:]) <= 0)
    f1, _ = M.pronostico(m, ctx.ng, ctx.norm, pos[:20])
    np.testing.assert_allclose(f0, f1, rtol=1e-4, atol=1e-4)


def test_medir_paso(grilla_kp):
    datos, esc = _datos(grilla_kp[0])
    r = T.medir_paso(modelo_chico(), datos, {"H": 24, "tgt": True}, esc, n=2, previos=1)
    assert r["s_paso"] > 0 and r["pasos_epoca"] == int(np.ceil(len(datos) / T.BATCH))
    assert r["s_epoca"] == pytest.approx(r["s_paso"] * r["pasos_epoca"])


def test_paso_compilado_equivale_a_eager(grilla_kp):
    from koopman import perdidas as L

    datos, esc = _datos(grilla_kp[0])
    m = modelo_chico()
    escala = tuple(tf.constant(np.asarray(a, dtype="float32")) for a in esc)
    lote = {k: tf.constant(v) for k, v in datos.lote(np.arange(64)).items()}
    t_e, l_e, g_e = T.perdida_y_gradientes(m, lote, {"tgt": True}, 6, escala, {}, L.LAMBDAS, training=False)
    cfg = {"tgt": True, "H_max": 24}
    f = tf.function(lambda lt, H: T.perdida_y_gradientes(m, lt, cfg, H, escala, {}, L.LAMBDAS, training=False))
    t_c, l_c, g_c = f(lote, tf.constant(6))
    assert set(t_e) == set(t_c)
    np.testing.assert_allclose(float(l_c), float(l_e), rtol=1e-5)
    for k in t_e:
        np.testing.assert_allclose(float(t_c[k]), float(t_e[k]), rtol=1e-5, atol=1e-7)
    for a, b in zip(g_c, g_e):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-4, atol=1e-6)


def test_un_trazado_con_curriculo_y_congelado(grilla_kp):
    datos, esc = _datos(grilla_kp[0])
    out = T.entrenar(modelo_chico(), datos, "f2", esc, epocas=3,
                     cfg_perdida={"curriculo": ((1, 1), (1, 6), (1, 24))}, log=lambda *_: None)
    assert [f["H"] for f in out["historia"]] == [1, 6, 24] and out["trazados"] == 1
    out = T.entrenar(modelo_chico(), datos, "smoke", esc, en_caliente=True, log=lambda *_: None)
    assert [f["congelado"] for f in out["historia"]] == [True, False] and out["trazados"] == 1
    assert "z_ini_b0" in out["historia"][0] and "z_fin_b0" in out["historia"][-1]


def test_lr_del_optimizador_llega_al_paso_compilado(grilla_kp):
    import keras

    from koopman import perdidas as L

    datos, esc = _datos(grilla_kp[0])
    m = modelo_chico()
    opt = keras.optimizers.Adam(learning_rate=T.LR, clipnorm=T.CLIPNORM)
    opt.build(m.entrenables)
    p = T.PasoCompilado(m, opt, {"H": 0}, esc, {}, L.LAMBDAS)
    act = tf.ones([len(m.entrenables)])
    w0 = [np.asarray(v).copy() for v in m.entrenables]
    opt.learning_rate = 0.0
    p(datos.lote(np.arange(32)), 0, act)
    assert all(np.array_equal(a, np.asarray(v)) for a, v in zip(w0, m.entrenables))
    opt.learning_rate = 1e-3
    p(datos.lote(np.arange(32)), 0, act)
    assert any(not np.array_equal(a, np.asarray(v)) for a, v in zip(w0, m.entrenables)) and p.trazados == 1
