"""Tarea 4.1: ponderación de pares y smearing de Duan (solo train)."""
import numpy as np
import pandas as pd

from ae import data as aed
from features import partition as pt
from operador import estado as E
from operador import rollout as R


def test_duan_corrige_sesgo_de_la_media_lognormal():
    rng = np.random.default_rng(0)
    n = 20000
    lp = rng.normal(3.0, 0.5, size=(n, 24, 6))
    sigma = 0.6
    lr = lp + rng.normal(0, sigma, size=lp.shape)
    real = np.ones_like(lp, dtype=bool)
    s = R.duan_factores(lr[: n // 2], lp[: n // 2], real[: n // 2])
    np.testing.assert_allclose(s, np.exp(sigma**2 / 2), rtol=0.02)
    truth = np.expm1(lr[n // 2:]).mean()
    corr = R.aplicar_duan(lp[n // 2:], s).mean()
    naive = np.clip(np.expm1(lp[n // 2:]), 0, None).mean()
    assert abs(corr / truth - 1) < 0.02
    assert abs(naive / truth - 1) > 0.1


def test_duan_ignora_horas_no_reales():
    lp = np.zeros((10, 24, 6))
    lr = np.ones_like(lp)
    real = np.zeros_like(lp, dtype=bool)
    np.testing.assert_array_equal(R.duan_factores(lr, lp, real), 1.0)


class _Rec:
    def __init__(self, grid, features):
        self.grid = grid
        self.part = pt.get_partition("ae-dmd")
        self.features = features


def test_pesos_y_duan_solo_con_t_mas_24_en_train(grilla_280):
    from features.build import build_features
    from operador import run as RUN

    grid, data = grilla_280
    ctx = E.contexto(grid, pt.get_partition("ae-dmd"))
    est = E.hankel(ctx)
    feats = build_features(data["sinca"], data["open_meteo"], data["feriados"])
    feats = feats.assign(target=feats["target"] * 0 + 500.0)  # todo sobre el umbral: peso > 1
    rec = _Rec(grid, feats)
    w = RUN.pesos_por_fila(rec, est)
    pos = est.positions[est.fit_mask()]
    en = pt.row_mask(grid.index[pos], rec.part, "train")
    assert (~en).sum() >= 1
    np.testing.assert_array_equal(w[~en], 1.0)
    assert np.all(w[en] > 1.0)
    fd = RUN.filas_duan(rec, est)
    end_train = aed.partition_spans(grid.index, rec.part)["train"][1]
    assert fd.max() + 24 < end_train
    segs = RUN.pesos_segmentos(rec, est)
    assert sum(len(s) for s in segs) == len(pos)
