"""DMD global estándar sobre el latente congelado (change `ae-reconstruccion`, grupo 4).

Sin búsqueda de parámetros (design.md D7): DMD exacto por mínimos cuadrados
sobre pares `(z_t, z_{t+1})`, rango completo y una variante truncada. Solo numpy.

Los datos llegan como **segmentos** de horas consecutivas (lista de arrays
`[T_i, d]`): las muestras pueden tener huecos, y los pares solo se forman
dentro de cada segmento.
"""
from __future__ import annotations

import numpy as np

HORIZON_FORECAST = 24  # promedio de las últimas 24 h decodificadas = mean(MP(t+1…t+24h))


def split_segments(z: np.ndarray, positions: np.ndarray) -> list[np.ndarray]:
    """Parte `z` (filas alineadas con `positions`) en bloques de posiciones consecutivas."""
    z = np.asarray(z, dtype=float)
    positions = np.asarray(positions, dtype=int)
    if z.shape[0] == 0:
        return []
    breaks = np.where(np.diff(positions) != 1)[0] + 1
    return [chunk for chunk in np.split(z, breaks)]


def _pairs(segments) -> tuple[np.ndarray, np.ndarray]:
    z0, z1 = [], []
    for seg in segments:
        seg = np.asarray(seg, dtype=float)
        if seg.shape[0] >= 2:
            z0.append(seg[:-1])
            z1.append(seg[1:])
    if not z0:
        raise ValueError("Se necesitan al menos dos instantes consecutivos de latente.")
    return np.concatenate(z0).T, np.concatenate(z1).T  # d × N


def fit_dmd(segments, rank: int | None = None) -> dict:
    """Ajusta `K` tal que `z_{t+1} ≈ K z_t`.

    `segments`: lista de `[T_i, d]` (o un array `[T, d]`). `rank=None` es el rango
    completo (`Z1 · pinv(Z0)`); con `rank` se usa la proyección truncada de DMD.
    """
    if isinstance(segments, np.ndarray):
        segments = [segments]
    z0, z1 = _pairs(segments)
    d = z0.shape[0]
    if rank is None or rank >= min(z0.shape):
        k = z1 @ np.linalg.pinv(z0)
        used_rank = d
    else:
        u, s, vt = np.linalg.svd(z0, full_matrices=False)
        u_r, s_r, v_r = u[:, :rank], s[:rank], vt[:rank].T
        a_tilde = u_r.T @ z1 @ v_r @ np.diag(1.0 / s_r)
        k = u_r @ a_tilde @ u_r.T
        used_rank = rank
    eigvals = np.linalg.eigvals(k)
    return {"K": k, "rank": int(used_rank), "eigvals": eigvals}


def predict_latent(k: np.ndarray, z0: np.ndarray, steps: int) -> np.ndarray:
    """`z_{t+h} = K^h z_t` para cada fila de `z0` (`[N, d]`)."""
    kh = np.linalg.matrix_power(k, steps)
    return np.asarray(z0, dtype=float) @ kh.T


def linearizability(k: np.ndarray, segments, horizons: tuple[int, ...] = (1, HORIZON_FORECAST)) -> dict[str, float]:
    """RMSE de predecir `z` a `h` pasos con `K^h` (clave `dmd_latent_rmse_<h>`)."""
    if isinstance(segments, np.ndarray):
        segments = [segments]
    out: dict[str, float] = {}
    for h in horizons:
        errors = []
        for seg in segments:
            seg = np.asarray(seg, dtype=float)
            if seg.shape[0] > h:
                errors.append((predict_latent(k, seg[:-h], h) - seg[h:]) ** 2)
        out[f"dmd_latent_rmse_{h}"] = float(np.sqrt(np.concatenate(errors).mean())) if errors else float("nan")
    return out


def forecast_from_decoded(decoded_ugm3: np.ndarray, horizon: int = HORIZON_FORECAST) -> np.ndarray:
    """Pronóstico informativo: promedio de las últimas `horizon` horas de cada ventana decodificada.

    `decoded_ugm3`: `[N, L, 6]` en µg/m³, con la última hora en `t+24`; sus últimas 24
    horas cubren `t+1 … t+24` (design.md D7). Devuelve `[N, 6]`.
    """
    decoded = np.asarray(decoded_ugm3, dtype=float)
    if decoded.shape[1] < horizon:
        raise ValueError("La ventana decodificada debe tener L >= 24.")
    return decoded[:, -horizon:, :].mean(axis=1)


def dmd_operator_arrays(fit: dict) -> dict[str, np.ndarray]:
    """Contenido de `dmd_operator.npz`: `K`, rango y autovalores (parte real e imaginaria)."""
    eig = np.asarray(fit["eigvals"])
    return {
        "K": np.asarray(fit["K"]),
        "rank": np.asarray(fit["rank"]),
        "eigvals_real": eig.real,
        "eigvals_imag": eig.imag,
    }
