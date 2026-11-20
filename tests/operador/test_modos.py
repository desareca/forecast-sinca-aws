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
    assert "diff_vs_inicial" in curva


def _sistema(lam, d_obs=6, T=24 * 60 + 48, seed=0, ruido=0.3):
    rng = np.random.default_rng(seed)
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
    z = np.zeros((T, d))
    z[0] = rng.normal(size=d)
    for t in range(1, T):
        z[t] = a @ z[t - 1] + rng.normal(scale=ruido, size=d)
    return a, z


def _evaluador(a, z):
    t_idx = np.arange(len(z) - 24)
    ts = pd.date_range("2026-04-01", periods=len(t_idx), freq="h", tz="America/Santiago")
    rows = []
    for c, (station, pollutant) in enumerate(V.SERIES):
        y = 50 + 10 * z[t_idx + 24, c]
        rows.append(pd.DataFrame({"timestamp": ts, "estacion": station, "pollutant": pollutant, "y_true": y,
                                  "persistencia": 50 + 10 * z[t_idx, c], "lightgbm": y + 1.0, "canal": c}))
    ev = V.Evaluador(pd.concat(rows, ignore_index=True), B=200)

    def evaluar(conservar):
        a2 = M.reconstruir(a, conservar)
        pred = 50 + 10 * (z[t_idx] @ np.linalg.matrix_power(a2, 24).T)[:, :6]
        return ev.puntaje(np.concatenate([pred[:, c] for c in range(6)]))
    return evaluar


def test_regla_nueva_protege_diario_y_no_colapsa():
    # diario casi irrelevante a 24 h (|λ|^24 pequeño), un modo lento relevante y 3 de ruido rápido.
    lam = [0.85 * np.exp(1j * 2 * np.pi / 24), 0.995, 0.995 * np.exp(1j * 2 * np.pi / 300), 0.25, -0.2]
    a, z = _sistema(lam)
    lam_all = np.linalg.eigvals(a)
    reps = list(lam_all[M.representantes(lam_all)])
    diario = M.modo_diario(reps)
    assert diario is not None and abs(abs(diario) - 0.85) < 1e-6
    kept, curva = M.eliminacion_atras(reps, _evaluador(a, z), protegidos=[diario])
    kept = np.asarray(kept)
    assert np.any(np.abs(kept - diario) < 1e-8)  # el diario nunca se quita
    assert not np.any(np.abs(np.abs(kept) - 0.25) < 1e-6) and not np.any(np.abs(np.abs(kept) - 0.2) < 1e-6)
    assert len(kept) >= 3  # diario + 2 lentos relevantes: no colapsa a 1 modo


def test_compara_contra_inicial():
    from types import SimpleNamespace

    # Cada modo quitado empeora 0.3; el intervalo de (candidato − inicial) es 0.3·k ± 1.96·0.4:
    # k = 1, 2 contienen 0; k = 3 ya no -> se detiene y conserva k = 2 (3 de 5 modos).
    reps = [0.9, 0.8, 0.7, 0.6, 0.5]
    g = np.random.default_rng(0).normal(size=4000)

    def evaluar(conservar):
        k = len(reps) - len(conservar)
        boot = np.full(4000, 10.0) if k == 0 else 10 + 0.3 * k + 0.4 * g
        return SimpleNamespace(media=10 + 0.3 * k, media_boot=boot)

    kept, curva = M.eliminacion_atras(reps, evaluar)
    assert len(kept) == 3
    assert list(curva["aceptado"]) == [True, True, True, False]


def test_cierre_conserva_numero_de_modos_emparejados():
    from operador import estimadores as S

    lam = [0.98 * np.exp(1j * 2 * np.pi / 24), 0.995, 0.6 * np.exp(1j * 1.0), 0.3]
    a, z = _sistema(lam, T=3000, seed=1, ruido=0.1)
    op_sel = S.exacto([z[:1500]], None)
    op_cierre = S.exacto([z], None)
    lam_sel = op_sel.eigvals()
    reps = lam_sel[M.representantes(lam_sel)]
    keep = [reps[np.argmax(np.abs(reps))], M.modo_diario(reps)]
    op_fix, tabla = M.fijar_modos(op_cierre, keep)
    lam_fix = op_fix.eigvals()
    lam_fix = lam_fix[np.abs(lam_fix) > 1e-8]
    assert len(M.representantes(lam_fix)) == 2 and len(tabla) == 2
    assert tabla["aparece"].all()
    lam_c = op_cierre.eigvals()
    for x in lam_fix:
        assert np.min(np.abs(lam_c - x)) < 1e-8
