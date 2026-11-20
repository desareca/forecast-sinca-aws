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
