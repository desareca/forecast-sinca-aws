"""Contraste numpy vs `pydmd` del pase DMD (ae-dmd-prep, tarea 5.4 a; D8).

Los casos que necesitan `pydmd` se saltan si no está instalado (en el entorno local no
lo está; se verifica en la imagen `tf-cpu-v2`, tarea 11.2). Los demás corren siempre.
"""
import importlib.metadata
import sys

import numpy as np
import pytest

from ae import dmd as dmdm


def _stable_linear_system(d: int = 6, steps: int = 240, seed: int = 0) -> np.ndarray:
    """Trayectoria de `z_{t+1} = A z_t` con autovalores de módulo 0.9 (sin ruido)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    angles = np.linspace(0.2, 1.1, d // 2)
    blocks = np.zeros((d, d))
    for i, theta in enumerate(angles):
        rot = 0.9 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        blocks[2 * i:2 * i + 2, 2 * i:2 * i + 2] = rot
    a = q @ blocks @ q.T
    z = np.zeros((steps, d))
    z[0] = rng.standard_normal(d)
    for t in range(steps - 1):
        z[t + 1] = a @ z[t]
    return z


def test_longest_segment_elige_el_mas_largo():
    short = np.zeros((5, 2))
    long = np.ones((9, 2))
    mid = np.full((7, 2), 2.0)
    chosen = dmdm.longest_segment([short, long, mid])
    assert chosen.shape == (9, 2)
    assert np.all(chosen == 1.0)


def test_longest_segment_sin_segmentos_lanza():
    with pytest.raises(ValueError):
        dmdm.longest_segment([])


def test_pydmd_crosscheck_sin_pydmd_lanza_importerror(monkeypatch):
    """Sin `pydmd` el contraste falla con ImportError (el llamador decide)."""
    monkeypatch.setitem(sys.modules, "pydmd", None)
    with pytest.raises(ImportError):
        dmdm.pydmd_crosscheck(_stable_linear_system())


def test_pase_sin_pydmd_registra_no_disponible(monkeypatch):
    """`_pydmd_check` sin `pydmd`: sin métricas y tag `no-disponible` (sin cortar el job)."""
    pytest.importorskip("tensorflow")
    from ae import run as aerun

    monkeypatch.setitem(sys.modules, "pydmd", None)
    metrics, version = aerun._pydmd_check([_stable_linear_system()], trunc_rank=3)
    assert metrics == {}
    assert version == "no-disponible"


def test_sistema_lineal_estable_concuerda_con_pydmd_completo_y_truncado():
    pytest.importorskip("pydmd")
    segment = _stable_linear_system(d=6)
    full = dmdm.pydmd_crosscheck(segment, rank=None, tol=1e-6)
    assert full["ok"] is True
    assert full["eig_maxdiff"] <= 1e-6
    assert full["pred1_maxdiff"] <= 1e-6
    trunc = dmdm.pydmd_crosscheck(segment, rank=3, tol=1e-6)
    assert trunc["ok"] is True
    assert trunc["eig_maxdiff"] <= 1e-6
    assert np.isnan(trunc["pred1_maxdiff"])
    assert trunc["version"] == importlib.metadata.version("pydmd")
    assert full["version"] == trunc["version"]
