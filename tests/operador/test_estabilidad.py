"""Tarea 7.1: recorte, fbDMD y bootstrap por modo."""
import numpy as np

from sint_operador import sistema_lineal
from operador import estimadores as S
from operador import modos as M


def test_recorte_real_y_radio():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    w = 2 * np.pi / 24
    base = np.zeros((4, 4))
    base[:2, :2] = 1.02 * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    base[2, 2], base[3, 3] = 1.1, 0.5
    a = q @ base @ q.T
    assert np.max(np.abs(np.linalg.eigvals(a))) > 1
    op = S.recorte(S.Operador(A=a, U=None, rank=4, estimador="exacto"), 1e-3)
    assert np.isrealobj(op.A)
    lam = np.linalg.eigvals(op.A)
    assert np.max(np.abs(lam)) <= 1 - 1e-3 + 1e-9
    assert np.min(np.abs(lam - 0.5)) < 1e-9  # el modo estable no cambia
    stable = np.diag([0.5, 0.2])
    np.testing.assert_allclose(S.recorte_matriz(stable), stable)


def test_fbdmd_rango_r_igual_formula_completa():
    rng = np.random.default_rng(1)
    a_r = np.array([[0.95, -0.2, 0.0], [0.2, 0.95, 0.0], [0.0, 0.0, 0.7]])
    y = np.zeros((400, 3))
    y[0] = rng.normal(size=3)
    for t in range(1, 400):
        y[t] = a_r @ y[t - 1] + 0.05 * rng.normal(size=3)
    q, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    x = y @ q.T
    op = S.fbdmd([x], rank=3, eps=None)
    full = S.fb_formula(y[:-1].T, y[1:].T)
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(op.A)), np.sort_complex(np.linalg.eigvals(full)), atol=1e-8)


def _transitorios(a, n_seg, largo, ruido_proc, seed):
    """Trayectorias transitorias (condición inicial aleatoria por segmento) con ruido de proceso."""
    rng = np.random.default_rng(seed)
    d = a.shape[0]
    segs = []
    for _ in range(n_seg):
        z = np.zeros((largo, d))
        z[0] = rng.normal(size=d) * 3
        for t in range(1, largo):
            z[t] = a @ z[t - 1] + ruido_proc * rng.normal(size=d)
        segs.append(z)
    return segs


def test_fbdmd_mas_cerca_de_la_verdad_con_ruido():
    a, _ = sistema_lineal(d=6, T=10, seed=2)
    segs = _transitorios(a, 300, 40, 0.01, 2)
    rng = np.random.default_rng(3)
    noisy = [s + 0.3 * rng.normal(size=s.shape) for s in segs]  # ruido de medición
    true = np.linalg.eigvals(a)

    def gap(lam):
        return np.abs(np.asarray(lam)[:, None] - true[None, :]).min(axis=1).mean()

    ex = S.exacto(noisy, None).eigvals()
    fb = S.fbdmd(noisy, eps=None).eigvals()
    assert gap(fb) < gap(ex)


def test_bootstrap_sin_ruido_aparicion_total():
    a, _ = sistema_lineal(d=6, T=10, seed=4)
    segs = _transitorios(a, 60, 24, 0.0, 4)  # un "día" por segmento, sin ruido
    days = np.repeat(np.arange(60), 24)
    ref = np.linalg.eigvals(a)

    def fit(w):
        return S.ponderado(segs, list(np.split(w, 60)), None).eigvals()

    out = M.bootstrap_modos(ref, fit, days, B=20, dias=7)
    assert len(out) == 4
    np.testing.assert_allclose(out["aparicion"], 1.0)


def test_modo_real_solo_vida_media():
    ref = np.array([0.9 + 0j])
    assert M.aparece(ref, np.array([0.905 + 0j]))[0]
    assert not M.aparece(ref, np.array([0.5 + 0j]))[0]
    comp = np.array([0.9 * np.exp(1j * 2 * np.pi / 24)])
    assert M.aparece(comp, np.array([0.9 * np.exp(1j * 2 * np.pi / 26)]))[0]
    assert not M.aparece(comp, np.array([0.9 * np.exp(1j * 2 * np.pi / 40)]))[0]
