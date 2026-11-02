"""Métricas de reconstrucción, referencias, diagnósticos del latente y compuerta.

Todo en numpy (sin TensorFlow). Las métricas de MP se calculan en µg/m³, solo
sobre valores reales (design.md D5).
"""
from __future__ import annotations

import numpy as np

# Tramos de concentración (µg/m³) del diagnóstico de error.
CONCENTRATION_BINS: tuple[tuple[str, float, float], ...] = (
    ("lt50", -np.inf, 50.0),
    ("50_100", 50.0, 100.0),
    ("gt100", 100.0, np.inf),
)
# Bandas espectrales por periodo (horas).
SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("ge168h", 168.0, np.inf),
    ("24_168h", 24.0, 168.0),
    ("lt24h", 0.0, 24.0),
)


def _clean(y_true, y_recon, mask=None):
    t = np.asarray(y_true, dtype=float).ravel()
    r = np.asarray(y_recon, dtype=float).ravel()
    keep = np.isfinite(t) & np.isfinite(r)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    return t[keep], r[keep]


def recon_metrics(y_true, y_recon, mask=None) -> dict[str, float]:
    """Métricas de aplanamiento y error de una serie (µg/m³, solo valores reales)."""
    t, r = _clean(y_true, y_recon, mask)
    if t.size < 2:
        return {}
    var_t = float(np.var(t))
    out: dict[str, float] = {
        "rmse": float(np.sqrt(np.mean((t - r) ** 2))),
        "bias": float(np.mean(r - t)),
        "var_ratio": float(np.std(r) / np.std(t)) if np.std(t) > 0 else float("nan"),
        # Pendiente de la regresión de lo reconstruido sobre lo real (1 = sin sesgo de escala).
        "slope": float(np.cov(t, r, ddof=0)[0, 1] / var_t) if var_t > 0 else float("nan"),
        "max_capture": float(r.max() / t.max()) if t.max() > 0 else float("nan"),
        "p95_ratio": float(np.quantile(r, 0.95) / np.quantile(t, 0.95)) if np.quantile(t, 0.95) > 0 else float("nan"),
        "p99_ratio": float(np.quantile(r, 0.99) / np.quantile(t, 0.99)) if np.quantile(t, 0.99) > 0 else float("nan"),
    }
    for label, lo, hi in CONCENTRATION_BINS:
        sel = (t >= lo) & (t < hi)
        out[f"n_{label}"] = float(sel.sum())
        if sel.any():
            out[f"rmse_{label}"] = float(np.sqrt(np.mean((t[sel] - r[sel]) ** 2)))
            out[f"bias_{label}"] = float(np.mean(r[sel] - t[sel]))
    return out


def spectral_ratio(y_true, y_recon, dt_hours: float = 1.0) -> dict[str, float]:
    """Potencia reconstruida / real por banda de periodo (atenuación de frecuencias altas).

    Requiere series contiguas sin `NaN` (el llamador resuelve huecos).
    """
    t = np.asarray(y_true, dtype=float).ravel()
    r = np.asarray(y_recon, dtype=float).ravel()
    if t.size < 8 or not (np.isfinite(t).all() and np.isfinite(r).all()):
        return {}
    power_t = np.abs(np.fft.rfft(t - t.mean())) ** 2
    power_r = np.abs(np.fft.rfft(r - r.mean())) ** 2
    freqs = np.fft.rfftfreq(t.size, d=dt_hours)
    out: dict[str, float] = {}
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / np.where(freqs > 0, freqs, 1.0), np.inf)
    for label, lo, hi in SPECTRAL_BANDS:
        sel = (periods >= lo) & (periods < hi) & (freqs > 0)
        denom = power_t[sel].sum()
        out[f"spec_ratio_{label}"] = float(power_r[sel].sum() / denom) if denom > 0 else float("nan")
    return out


# --- Referencias -------------------------------------------------------------


def pca_fit(x_train: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA con `d` componentes sobre filas `x_train` (N × F). Devuelve `(media, componentes)`."""
    mean = x_train.mean(axis=0)
    _, _, vt = np.linalg.svd(x_train - mean, full_matrices=False)
    d = min(d, vt.shape[0])
    return mean, vt[:d].T  # F × d


def pca_reconstruct(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components @ components.T + mean


# --- Latente -----------------------------------------------------------------


def latent_diagnostics(z: np.ndarray, d_aux: int, dead_std: float = 1e-4) -> dict[str, float]:
    """Dimensión efectiva (razón de participación del PCA de `z`) y dimensiones muertas."""
    z = np.asarray(z, dtype=float)
    stds = z.std(axis=0)
    cov = np.cov(z, rowvar=False, ddof=0)
    eig = np.clip(np.linalg.eigvalsh(np.atleast_2d(cov)), 0.0, None)
    total = eig.sum()
    effective = float(total**2 / np.sum(eig**2)) if total > 0 else 0.0
    d_mp = z.shape[1] - d_aux
    return {
        "eff_dim_pr": effective,
        "dead_dims": float(np.sum(stds < dead_std)),
        "dead_dims_mp": float(np.sum(stds[:d_mp] < dead_std)),
        "dead_dims_aux": float(np.sum(stds[d_mp:] < dead_std)),
        "std_z_mp_mean": float(stds[:d_mp].mean()) if d_mp > 0 else float("nan"),
        "std_z_aux_mean": float(stds[d_mp:].mean()) if d_aux > 0 else float("nan"),
    }


# --- Compuerta ---------------------------------------------------------------


def gate_decision(
    ae_metrics: dict[str, dict[str, float]],
    pca_metrics: dict[str, dict[str, float]],
    mean_metrics: dict[str, dict[str, float]] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Compuerta de reconstrucción sobre MP, en dos capas (design.md D5).

    - **Capa relativa** (siempre activa, sin parámetros): el AE supera, en RMSE de
      MP y en **todas** las series, al predictor del promedio (`mean_metrics`) y a
      PCA(d) (`pca_metrics`). Se registra siempre como `gate_relative`.
    - **Capa numérica** (solo si se pasan `thresholds`, fijados por el Arquitecto
      en el punto de control 10.3): `min_var_ratio`, `slope_tol` (|pendiente−1| ≤ tol)
      y `max_rmse_vs_pca` (RMSE_AE ≤ factor × RMSE_PCA). Sin umbrales `numeric` es None.

    `gate_pass` = capa relativa **y**, si hay umbrales, capa numérica.
    """
    series = sorted(ae_metrics)
    beats_pca = {s: ae_metrics[s]["rmse"] < pca_metrics[s]["rmse"] for s in series if s in pca_metrics}
    beats_mean = {}
    if mean_metrics is not None:
        beats_mean = {s: ae_metrics[s]["rmse"] < mean_metrics[s]["rmse"] for s in series if s in mean_metrics}
    beats_pca_all = bool(beats_pca) and all(beats_pca.values())
    beats_mean_all = bool(beats_mean) and all(beats_mean.values())
    relative = beats_pca_all and beats_mean_all

    numeric: bool | None = None
    checks: dict[str, bool] = {}
    if thresholds is not None:
        for s in series:
            m = ae_metrics[s]
            if "min_var_ratio" in thresholds:
                checks[f"{s}_var"] = m["var_ratio"] >= thresholds["min_var_ratio"]
            if "slope_tol" in thresholds:
                checks[f"{s}_slope"] = abs(m["slope"] - 1.0) <= thresholds["slope_tol"]
            if "max_rmse_vs_pca" in thresholds and s in pca_metrics:
                checks[f"{s}_pca"] = m["rmse"] <= thresholds["max_rmse_vs_pca"] * pca_metrics[s]["rmse"]
        numeric = bool(checks) and all(checks.values())

    gate_pass = relative and (numeric is None or numeric)
    return {
        "gate_pass": bool(gate_pass),
        "relative": bool(relative),
        "numeric": numeric,
        "beats_pca": bool(beats_pca_all),
        "beats_mean": bool(beats_mean_all),
        "beats_pca_by_series": {s: bool(v) for s, v in beats_pca.items()},
        "beats_mean_by_series": {s: bool(v) for s, v in beats_mean.items()},
        "thresholds": thresholds,
        "checks": {k: bool(v) for k, v in checks.items()},
    }
