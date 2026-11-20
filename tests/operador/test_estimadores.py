"""Tarea 1.3 (y 4.1 W = I): estimadores, Gavish–Donoho y espectro."""
import numpy as np
import pytest

from sint_operador import hankel_1d, sinusoide_amortiguada, sistema_lineal
from operador import estimadores as S
from operador import modos


def test_hankel_sinusoide_recupera_periodo_y_vida_media():
    h = hankel_1d(sinusoide_amortiguada(T=800), 24)
    op = S.exacto([h], "gd")
    esp = modos.espectro(op.eigvals())
    comp = esp[esp["tipo"] == "complejo"].iloc[0]
    assert comp["periodo_h"] == pytest.approx(24.0, rel=0.02)
    assert comp["vida_media_h"] == pytest.approx(36.0, rel=0.02)


def test_gavish_donoho_igual_pydmd():
    pydmd = pytest.importorskip("pydmd")
    rng = np.random.default_rng(1)
    _, z = sistema_lineal(d=6, T=400)
    x = np.concatenate([z, z @ rng.normal(size=(6, 14))], axis=1) + 1e-3 * rng.normal(size=(400, 20))
    z0, _, _ = S.pares([x])
    model = pydmd.DMD(svd_rank=0, exact=True)
    model.fit(x.T)
    assert S.rango_gd(z0) == len(model.eigs)


@pytest.mark.parametrize("rank", [None, 4])
def test_exacto_igual_pydmd(rank):
    pydmd = pytest.importorskip("pydmd")
    _, z = sistema_lineal(d=6, T=300, ruido=0.01)
    op = S.exacto([z], rank)
    model = pydmd.DMD(svd_rank=-1 if rank is None else rank, exact=True)
    model.fit(z.T)
    a, b = np.sort_complex(op.eigvals()), np.sort_complex(np.asarray(model.eigs))
    gap = max(np.abs(a[:, None] - b[None, :]).min(axis=1).max(), np.abs(b[:, None] - a[None, :]).min(axis=1).max())
    assert gap <= 1e-6


def test_exacto_recupera_sistema_y_segmentos():
    a, z = sistema_lineal(d=6, T=300)
    op = S.exacto([z[:150], z[150:]], None)
    np.testing.assert_allclose(op.K, a, atol=1e-8)
    z0, z1, _ = S.pares([z[:150], z[150:]])
    assert z0.shape[1] == 298  # sin el par que cruza el corte


def test_ponderado_con_pesos_unitarios_es_exacto():
    _, z = sistema_lineal(d=6, T=300, ruido=0.05)
    segs = [z[:120], z[120:]]
    w = [np.ones(len(s)) for s in segs]
    for rank in (None, 3):
        np.testing.assert_allclose(S.ponderado(segs, w, rank).K, S.exacto(segs, rank).K, atol=1e-10)


def test_espectro_par_conjugado_un_modo():
    a, _ = sistema_lineal(d=6)
    esp = modos.espectro(a)
    assert len(esp) == 4  # 2 pares + 2 reales
    assert (esp["tipo"] == "complejo").sum() == 2
