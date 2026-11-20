"""Tareas 1.4 y 2.1: rollout autónomo y oráculo."""
import numpy as np
import pandas as pd

from ae import data as aed
from sint_operador import sistema_lineal
from operador import estado as E
from operador import estimadores as S
from operador import rollout as R


def _norm_identidad():
    cols = [name for name, _, _ in aed.mp_channels()]
    return aed.Normalizer(mean=pd.Series(0.0, index=cols), std=pd.Series(1.0, index=cols))


def _estado(z):
    """Estado de prueba: `d = 6` = canales de MP; decoder = repetir el estado en las 24 h."""
    def decode_z(zz):
        return np.repeat(np.asarray(zz)[:, None, :], 24, axis=1)

    pos = np.arange(len(z))
    return E.Estado(nombre="lineal", z=z, positions=pos, split=np.array(["train"] * len(z), dtype=object), d_mp=6,
                    norm=_norm_identidad(), decode_z=decode_z)


def test_autonomo_iguala_verdad_en_sistema_lineal():
    a, z = sistema_lineal(d=6, T=400)
    z = z * 0.3
    e = _estado(z)
    op = S.exacto(e.segments(), None)
    t = np.arange(10, 300)
    forecast, dec = R.autonomo(op, e, t)
    assert dec.shape == (len(t), 24, 6)
    np.testing.assert_allclose(forecast, np.clip(np.expm1(z[t + 24]), 0, None), atol=1e-8)


def test_oraculo_con_decoder_identidad_es_target():
    _, z = sistema_lineal(d=6, T=200)
    e = _estado(z * 0.3)
    t = np.arange(0, 150)
    forecast, _ = R.oraculo(e, t)
    np.testing.assert_allclose(forecast, np.clip(np.expm1(0.3 * z[t + 24]), 0, None), atol=1e-12)


def test_directo24_en_sistema_exacto_es_k24():
    a, z = sistema_lineal(d=6, T=400)
    op = S.directo24([z], None)
    np.testing.assert_allclose(op.K, np.linalg.matrix_power(a, 24), atol=1e-8)
