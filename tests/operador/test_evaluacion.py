"""Tarea 1.5: filas, métricas, bootstrap emparejado."""
import numpy as np
import pandas as pd
import pytest

from operador import evaluacion as V


def _rows(days: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-04-01", periods=24 * days, freq="h", tz="America/Santiago")
    frames = []
    for c, (station, pollutant) in enumerate(V.SERIES):
        y = np.abs(40 + 30 * np.sin(np.arange(len(ts)) / 30) + rng.normal(0, 10, len(ts)))
        frames.append(pd.DataFrame({"timestamp": ts, "estacion": station, "pollutant": pollutant, "y_true": y,
                                    "persistencia": y + rng.normal(0, 8 if c % 2 else 4, len(ts)),
                                    "lightgbm": y + rng.normal(0, 4 if c % 2 else 8, len(ts)), "canal": c}))
    return pd.concat(frames, ignore_index=True)


def test_misma_semilla_mismos_intervalos():
    rows = _rows()
    a, b = V.Evaluador(rows, B=300, seed=0), V.Evaluador(rows, B=300, seed=0)
    yhat = rows["lightgbm"].to_numpy() * 1.05
    pd.testing.assert_frame_equal(a.tabla(a.puntaje(yhat)), b.tabla(b.puntaje(yhat)))


def test_variante_igual_referencia_diferencia_cero():
    rows = _rows()
    ev = V.Evaluador(rows, B=300)
    p = ev.puntaje(rows["persistencia"].to_numpy())
    d = V.diferencia(p, ev.ref("persistencia"))
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0
    assert not V.significativa(d)


def test_fila_faltante_error_con_serie_y_variante():
    rows = _rows(days=10)
    ev = V.Evaluador(rows, B=50)
    stamps = pd.DatetimeIndex(rows["timestamp"].unique())
    forecast = np.ones((len(stamps), 6))
    forecast[5, 2] = np.nan
    with pytest.raises(ValueError, match=r"Variante mi-variante.*Las Encinas\|pm25"):
        ev.alinear("mi-variante", stamps, forecast)
    with pytest.raises(ValueError, match="mi-variante"):
        ev.alinear("mi-variante", stamps[1:], np.ones((len(stamps) - 1, 6)))


def test_mejor_referencia_por_serie():
    rows = _rows()
    ev = V.Evaluador(rows, B=50)
    best = ev.mejor_ref()
    assert best == ["lightgbm" if c % 2 else "persistencia" for c in range(6)]
    t = ev.tabla(ev.puntaje(rows["lightgbm"].to_numpy()))
    lg = t[t["serie"] != "media"]
    np.testing.assert_allclose(lg["razon_mejor"].to_numpy()[1::2], 1.0)


def test_bootstrap_bloques_conteos():
    c = V.block_counts(90, B=100, dias=7, seed=0)
    assert c.shape == (100, 90)
    np.testing.assert_array_equal(c.sum(axis=1), 90)


def test_interseccion_h48():
    rows = _rows(days=5)
    sub = rows[rows["timestamp"] >= rows["timestamp"].min() + pd.Timedelta(hours=47)]
    inter = V.interseccion(rows, sub)
    assert len(inter) == len(sub)
