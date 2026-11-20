"""Tarea 8.1: eliminación hacia atrás de modos."""
import numpy as np
import pandas as pd

from operador import evaluacion as V
from operador import modos as M


def test_elimina_ruido_y_conserva_relevantes():
    rng = np.random.default_rng(0)
    # 2 modos relevantes (par diario lento + real lento) y 4 de ruido rápido en 6 dimensiones.
    lam = [0.995 * np.exp(1j * 2 * np.pi / 24), 0.995, 0.25, -0.2, 0.15 * np.exp(1j * 2.0)]
    blocks = []
    for l in lam:
        if abs(np.imag(l)) > 0:
            r, w = abs(l), np.angle(l)
            blocks.append(r * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]]))
        else:
            blocks.append(np.array([[np.real(l)]]))
    d = sum(b.shape[0] for b in blocks)
    base = np.zeros((d, d))
    i = 0
    for b in blocks:
        n = b.shape[0]
        base[i:i + n, i:i + n] = b
        i += n
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = q @ base @ q.T
    T = 24 * 60 + 48
    z = np.zeros((T, d))
    z[0] = rng.normal(size=d)
    for t in range(1, T):
        z[t] = a @ z[t - 1] + rng.normal(scale=0.3, size=d)
    assert d == 7
    z6 = z[:, :6]
    t_idx = np.arange(T - 24)
    ts = pd.date_range("2026-04-01", periods=len(t_idx), freq="h", tz="America/Santiago")
    rows = []
    for c, (station, pollutant) in enumerate(V.SERIES):
        y = 50 + 10 * z6[t_idx + 24, c]
        rows.append(pd.DataFrame({"timestamp": ts, "estacion": station, "pollutant": pollutant, "y_true": y,
                                  "persistencia": 50 + 10 * z6[t_idx, c], "lightgbm": y + 1.0, "canal": c}))
    ev = V.Evaluador(pd.concat(rows, ignore_index=True), B=200)
    lam_all = np.linalg.eigvals(a)
    reps = lam_all[M.representantes(lam_all)]

    def evaluar(conservar):
        a2 = M.reconstruir(a, conservar)
        pred = 50 + 10 * (z[t_idx] @ np.linalg.matrix_power(a2, 24).T)[:, :6]
        yhat = np.concatenate([pred[:, c] for c in range(6)])
        return ev.puntaje(yhat)

    kept, curva = M.eliminacion_atras(list(reps), evaluar)
    mods = np.abs(np.asarray(kept))
    assert len(kept) == 2 and np.all(mods > 0.9)
    assert curva["n_modos"].iloc[0] == 5 and not curva["aceptado"].iloc[-1]
