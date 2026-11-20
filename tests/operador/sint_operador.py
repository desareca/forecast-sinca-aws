"""Datos sintéticos del change `dmd-operador-global` (nombre único: no choca con `tests/ae/conftest`)."""
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

_spec = importlib.util.spec_from_file_location("ae_conftest_sintetico", ROOT / "tests" / "ae" / "conftest.py")
_ae_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ae_conftest)
synthetic_validated = _ae_conftest.synthetic_validated

ARTIFACT_LOCATION = "file:///opt/ml/mlflow/mlruns"
EXPERIMENT = "sinca-baseline"


def sistema_lineal(d: int = 6, T: int = 600, seed: int = 0, ruido: float = 0.0):
    """Trayectoria `z_{t+1} = A z_t` con `A` estable (autovalores conocidos). Devuelve `(A, Z [T, d])`."""
    rng = np.random.default_rng(seed)
    lam = [0.97 * np.exp(1j * 2 * np.pi / 24), 0.9 * np.exp(1j * 2 * np.pi / 60), 0.95, 0.8]
    blocks = []
    for l in lam:
        if np.iscomplex(l):
            r, w = abs(l), np.angle(l)
            blocks.append(r * np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]]))
        else:
            blocks.append(np.array([[l.real]]))
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
        z[t] = a @ z[t - 1] + ruido * rng.normal(size=d)
    return a, z


def sinusoide_amortiguada(T: int = 600, periodo: float = 24.0, vida_media: float = 36.0, ruido: float = 1e-6, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    r = 0.5 ** (1.0 / vida_media)
    return r**t * np.cos(2 * np.pi * t / periodo) + ruido * rng.normal(size=T)


def hankel_1d(x: np.ndarray, h: int = 24) -> np.ndarray:
    return np.stack([x[i:i + h] for i in range(len(x) - h + 1)])
