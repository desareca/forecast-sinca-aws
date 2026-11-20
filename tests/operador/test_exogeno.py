"""Tarea 5.1: rollout exógeno sin meteo futura y DMDc."""
import dataclasses

import numpy as np
import pandas as pd

from features import partition as pt
from operador import estado as E
from operador import estimadores as S
from operador import rollout as R


def test_dmdc_recupera_a_y_b():
    rng = np.random.default_rng(0)
    a = np.diag([0.9, 0.7, -0.5]) + 0.05 * rng.normal(size=(3, 3))
    b = rng.normal(size=(3, 2))
    u = rng.normal(size=(500, 2))
    x = np.zeros((500, 3))
    for t in range(1, 500):
        x[t] = a @ x[t - 1] + b @ u[t - 1]
    op = S.dmdc([x[:250], x[250:]], [u[:250], u[250:]])
    np.testing.assert_allclose(op.A, a, atol=1e-8)
    np.testing.assert_allclose(op.meta["K_ma"], b, atol=1e-8)


def test_fuente_persistida():
    tau, src = R.fuente_persistida(np.array([100]), np.array([110]))
    assert tau[0, 0] == 87 and tau[0, -1] == 110
    np.testing.assert_array_equal(src[0, tau[0] <= 100], tau[0, tau[0] <= 100])
    np.testing.assert_array_equal(src[0, tau[0] > 100], tau[0, tau[0] > 100] - 24)


def test_ventana_persistida_usa_tau_menos_24_y_calendario_real(grilla_280):
    grid, _ = grilla_280
    ctx = E.contexto(grid, pt.get_partition("ae-dmd"))
    t = np.array([3000])
    xa = R.ventana_aux_persistida(ctx.ng, t, 5)
    aux = ctx.ng.aux_z.to_numpy(np.float32)
    cal = ctx.ng.cal.to_numpy(np.float32)
    # última hora de la ventana = t+5 > t: meteo de t+5−24, calendario de t+5
    np.testing.assert_array_equal(xa[0, -1, :12], aux[3005 - 24])
    np.testing.assert_array_equal(xa[0, -1, 24:], cal[3005])
    np.testing.assert_array_equal(xa[0, 0, :12], aux[3005 - 23])  # τ = t−18 ≤ t: observado


def test_alterar_meteo_posterior_no_cambia_pronostico(grilla_280):
    grid, _ = grilla_280
    ctx = E.contexto(grid, pt.get_partition("ae-dmd"))
    est = E.pca_state(ctx)
    segs = est.segments()
    op = S.dmdc([s[:, :est.d_mp] for s in segs], [s[:, est.d_mp:] for s in segs])
    t = est.positions[(est.split == "sel")][:1]

    def aux_fn_de(e):
        return lambda pos, k: e.extra["proj_aux"](R.ventana_aux_persistida(e.extra["ng"], pos, k).reshape(len(pos), -1).astype(float))

    f1, _ = R.exogeno(op, est, t, aux_fn_de(est))
    ng2 = dataclasses.replace(ctx.ng, aux_z=ctx.ng.aux_z.copy(), aux_avail=ctx.ng.aux_avail.copy())
    later = np.arange(len(grid.index)) > t[0]
    rng = np.random.default_rng(1)
    ng2.aux_z.iloc[later, :] = rng.normal(size=(int(later.sum()), ng2.aux_z.shape[1]))
    ng2.aux_avail.iloc[later, :] = False
    est2 = dataclasses.replace(est, extra={**est.extra, "ng": ng2})
    f2, _ = R.exogeno(op, est2, t, aux_fn_de(est2))
    np.testing.assert_allclose(f1, f2, atol=1e-12)
    # control: la meteo pasada sí cambia el pronóstico
    ng3 = dataclasses.replace(ctx.ng, aux_z=ctx.ng.aux_z.copy())
    ng3.aux_z.iloc[t[0] - 30:t[0] + 1, :] += 3.0
    est3 = dataclasses.replace(est, extra={**est.extra, "ng": ng3})
    f3, _ = R.exogeno(op, est3, t, aux_fn_de(est3))
    assert not np.allclose(f1, f3)


def test_entrada_hankel_persistida(grilla_280):
    grid, _ = grilla_280
    ctx = E.contexto(grid, pt.get_partition("ae-dmd"))
    t = np.array([2000, 2500])
    u0 = R.entrada_hankel(ctx.ng, t, 0)
    u3 = R.entrada_hankel(ctx.ng, t, 3)
    np.testing.assert_array_equal(u0[:, :12], ctx.ng.aux_z.to_numpy(float)[t])
    np.testing.assert_array_equal(u3[:, :12], ctx.ng.aux_z.to_numpy(float)[t + 3 - 24])
    np.testing.assert_array_equal(u3[:, 12:], ctx.ng.cal.to_numpy(float)[t + 3])
