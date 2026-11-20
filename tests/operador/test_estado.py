"""Tareas 1.2 y 3.1: representaciones (forma, decodificación, ajuste solo en train)."""
import numpy as np
import pandas as pd

from ae import data as aed
from features import partition as pt
from operador import estado as E


def _ctx(grid):
    return E.contexto(grid, pt.get_partition("ae-dmd"), ("train",))


def test_hankel_decodifica_exacto(grilla_280):
    grid, _ = grilla_280
    ctx = _ctx(grid)
    for length, meteo in ((24, False), (24, True), (48, False)):
        e = E.hankel(ctx, length, meteo)
        assert e.d == length * 6 + (length * 12 if meteo else 0)
        assert e.d_mp == length * 6
        rows = np.arange(0, len(e.positions), 997)
        dec = e.decode(e.z[rows])
        assert dec.shape == (len(rows), 24, 6)
        for i, r in enumerate(rows):
            p = e.positions[r]
            truth = grid.mp.to_numpy(float)[p - 23:p + 1]
            np.testing.assert_allclose(dec[i], np.clip(truth, 0, None), rtol=1e-4, atol=1e-3)


def test_hankel_por_estacion_bloques():
    bloques = E.bloques_por_estacion(24)
    assert len(bloques) == 3
    allidx = np.sort(np.concatenate(bloques))
    np.testing.assert_array_equal(allidx, np.arange(144))
    assert all(len(b) == 48 for b in bloques)


def test_pca_rango_12_sin_error():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 12)) @ rng.normal(size=(12, 144)) + rng.normal(size=144)
    mean, comp = E.pca_fit(x, 12)
    np.testing.assert_allclose((x - mean) @ comp @ comp.T + mean, x, atol=1e-9)


def test_pca_state_forma(grilla_280):
    grid, _ = grilla_280
    e = E.pca_state(_ctx(grid))
    assert e.d == 16 and e.d_mp == 12
    assert e.decode(e.z[:5]).shape == (5, 24, 6)


def test_ajuste_solo_en_train(grilla_280):
    grid, data = grilla_280
    sinca = data["sinca"].copy()
    late = pd.to_datetime(sinca["timestamp"]) >= pd.Timestamp("2026-03-01", tz="America/Santiago")
    for col in ("pm25", "pm10", "pres", "wspd"):
        sinca.loc[late, col] = sinca.loc[late, col] * 3.0
    grid2 = aed.build_grid(sinca, data["open_meteo"], data["feriados"])
    c1, c2 = _ctx(grid), _ctx(grid2)
    pd.testing.assert_series_equal(c1.norm.mean, c2.norm.mean)
    pd.testing.assert_series_equal(c1.norm.std, c2.norm.std)
    p1, p2 = E.pca_state(c1), E.pca_state(c2)
    np.testing.assert_allclose(np.abs(p1.extra["c_mp"]), np.abs(p2.extra["c_mp"]), atol=1e-8)
    np.testing.assert_allclose(np.abs(p1.extra["c_aux"]), np.abs(p2.extra["c_aux"]), atol=1e-8)
    fit = p1.fit_mask()
    assert set(p1.split[fit]) == {"train"}
    end_train = aed.partition_spans(grid.index, pt.get_partition("ae-dmd"))["train"][1]
    assert p1.positions[fit].max() < end_train
