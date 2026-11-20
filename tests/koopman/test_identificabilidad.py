"""Tareas 5.2 y 5.3: aportes, dominantes, compuerta, jacobiano y eliminación de bloques."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from sint_koopman import modelo_chico
from koopman import identificabilidad as I
from operador import modos


def test_dominantes_reproducibles():
    m = modelo_chico()
    z = np.random.default_rng(0).normal(size=(200, 10))
    w = np.ones((200, 6))
    a1 = I.aportes(I.contribuciones(m, z), w)
    a2 = I.aportes(I.contribuciones(m, z), w)
    np.testing.assert_array_equal(a1, a2)
    assert I.dominantes(a1) == I.dominantes(a2)
    assert I.dominantes(np.array([5.0, 3.0, 1.0, 1.0])) == [0, 1]


def test_jacobiano_exacto_con_decoder_lineal():
    m = modelo_chico()
    z = np.random.default_rng(1).normal(size=(5, 10))
    jac = I.jacobiano_promedio(m, z)
    w = m.kernel_decoder().reshape(10, 24, 6).mean(axis=1)
    np.testing.assert_allclose(jac, np.broadcast_to(w.T, jac.shape), atol=1e-5)


def _boot(lams, aparicion=1.0):
    lams = np.asarray(lams, complex)
    esp = modos.espectro(lams)
    esp["aparicion"] = aparicion
    for col in ("periodo_h", "vida_media_h"):
        esp[f"{col}_lo"] = esp[col] * 0.9
        esp[f"{col}_hi"] = esp[col] * 1.1
    return esp


def _lam(hl, per=None):
    r = 0.5 ** (1 / hl)
    return r * np.exp(1j * 2 * np.pi / per) if per else complex(r)


def test_compuerta_mismos_modos_aprueba_y_ausente_no():
    base = [_lam(300, 24), _lam(200), _lam(80, 400)]
    sem = {s: {"dominantes": base[1:], "modos": base, "boot": _boot(base)} for s in (0, 1, 2)}
    ok, tabla = I.compuerta(sem)
    assert ok and len(tabla) == 6
    sem[2] = {"dominantes": [base[1]], "modos": [base[0], base[1], _lam(5)], "boot": _boot([base[0], base[1], _lam(5)])}
    ok2, tabla2 = I.compuerta(sem)
    assert not ok2
    assert not tabla2[(tabla2.semilla == 0) & (tabla2.lam_im > 0)]["pareja"].iloc[0]


def test_compuerta_bootstrap_bajo_no_aprueba():
    base = [_lam(300, 24), _lam(200)]
    sem = {s: {"dominantes": [base[1]], "modos": base, "boot": _boot(base, 0.5 if s == 1 else 1.0)} for s in (0, 1, 2)}
    assert not I.compuerta(sem)[0]


def test_eliminacion_bloques_ruido_fuera_diario_dentro():
    m = modelo_chico()
    # bloques: 0 diario, 1-2 libres, 3-6 reales; el 6 es "ruido" (quitarlo no cambia nada), el diario tampoco.
    relevantes = {1, 2, 3, 4, 5}
    g = np.random.default_rng(0).normal(size=4000)

    def puntaje(conservar):
        faltan = len(relevantes - set(conservar))
        boot = np.full(4000, 10.0) if faltan == 0 else 10 + 2.0 * faltan + 0.4 * g
        return SimpleNamespace(media=10 + 2.0 * faltan, media_boot=boot)

    kept, curva = I.eliminacion_bloques(m, puntaje)
    assert 0 in kept and 6 not in kept
    assert relevantes <= set(kept)
    assert not curva["aceptado"].iloc[-1]
