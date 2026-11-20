"""Tareas 2.1 y 2.2: modelo, operador por bloques y pronóstico sin meteo futura."""
import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from sint_koopman import contexto_y_posiciones, modelo_chico
from koopman import modelo as M


def test_calendario_fuera_del_encoder(grilla_kp):
    grid, _ = grilla_kp
    ctx, pos, _ = contexto_y_posiciones(grid, 50)
    m = modelo_chico()
    x1 = M.entradas(ctx.ng, pos)
    ng2 = dataclasses.replace(ctx.ng, cal=ctx.ng.cal * 0 + 7.0)
    x2 = M.entradas(ng2, pos)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_allclose(np.asarray(m.encode(x1)), np.asarray(m.encode(x2)))
    assert x1.shape == (50, 24, M.N_IN)


@pytest.mark.parametrize("variante", ["", "perlib"])
def test_radio_acotado_y_periodo_diario(variante):
    op = M.OperadorBloques(10, variante)
    op.a.assign(np.full(7, -60.0, dtype="float32"))   # softplus(a) ≈ 0 -> techo
    op.b.assign(np.array([30.0, -30.0], dtype="float32"))
    if variante == "perlib":
        op.b_diario.assign(np.float32(2.0))
    lam = op.autovalores()
    assert np.max(np.abs(lam)) <= 2 ** (-1 / 1000) + 1e-6
    op.a.assign(np.full(7, 60.0, dtype="float32"))
    assert np.max(np.abs(op.autovalores())) < 1
    k = op.matriz_np()
    th = math.atan2(k[1, 0], k[0, 0])
    if variante == "":
        assert abs(2 * math.pi / th - 24.0) < 1e-3
    else:
        assert abs(2 * math.pi / th - 24.0) > 1


def test_decoder_lineal_suma_de_bloques():
    m = modelo_chico()
    z = np.random.default_rng(0).normal(size=(5, 10)).astype("float32")
    full = np.asarray(m.decode(z))
    bias = np.asarray(m.decode(np.zeros_like(z)))
    partes = []
    for b in M.bloques(10):
        zb = np.zeros_like(z)
        zb[:, b] = z[:, b]
        partes.append(np.asarray(m.decode(zb)) - bias)
    np.testing.assert_allclose(full, sum(partes) + bias, atol=1e-5)


def test_estructura_b4_y_variantes():
    assert len(M.bloques(8)) == 1 + 2 + 2
    assert M.OperadorBloques(10, "sinbu").B is None
    kd = M.OperadorBloques(10, "kden")
    assert kd.kden is not None and any(v is kd.kden for v in kd.trainable)


def test_pronostico_sin_meteo_futura(grilla_kp):
    grid, _ = grilla_kp
    ctx, pos, _ = contexto_y_posiciones(grid, None)
    m = modelo_chico()
    t = pos[1000:1003]
    f1, lp1 = M.pronostico(m, ctx.ng, ctx.norm, t)
    later = np.arange(len(grid.index)) > t.max()
    ng2 = dataclasses.replace(ctx.ng, aux_z=ctx.ng.aux_z.copy(), aux_avail=ctx.ng.aux_avail.copy(), mp_z=ctx.ng.mp_z.copy())
    ng2.aux_z.iloc[later, :] = 9.0
    ng2.aux_avail.iloc[later, :] = False
    ng2.mp_z.iloc[later, :] = 9.0
    f2, _ = M.pronostico(m, ng2, ctx.norm, t)
    np.testing.assert_allclose(f1, f2, rtol=1e-6)
    assert f1.shape == (3, 6) and np.isfinite(f1).all()
    u = M.u_futuro(ctx.ng, t)
    np.testing.assert_array_equal(u[:, 0], ctx.ng.cal.to_numpy(np.float32)[t + 1][:, M.U_IDX])
