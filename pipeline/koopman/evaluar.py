"""Evaluación del Koopman con el arnés de `operador` (`deep-koopman`, D10).

Filas fijas de 7.1 reconstruidas desde los runs de referencia (36 DMD v2 ∩ LightGBM v2 ∩ persistencia v2),
bootstrap de bloques de 7 días (B = 2000, emparejado), referencias LightGBM, persistencia y
GRU+DMD+Duan de 7.1 (media de semillas), Duan con y sin, diagnósticos y veredicto D12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import load as analysis_load
from operador import evaluacion as evm
from operador import modos
from operador import rollout as ro

from .modelo import KoopmanModelo, pronostico

RUN_71 = "sinca-zona-dmdop-gru-e16-aedmd-v2-s{seed}"
SEEDS_71 = (0, 1, 2)


def _preds(state_dir: str, run: dict, split: str = "sel") -> pd.DataFrame:
    path = analysis_load.artifact_path(state_dir, run["run_id"], "predictions.parquet")
    if not path:
        raise FileNotFoundError(f"Run {run.get('run_name')} sin predictions.parquet")
    frame = pd.read_parquet(path)
    return frame[frame["split"] == split][["timestamp", "estacion", "pollutant", "y_true", "y_pred"]]


def filas_referencia(state_dir: str, runs: list[dict] | None = None) -> pd.DataFrame:
    """`eval_rows` de 7.1 desde los runs v2 del estado (mismas filas que `operador`, D3)."""
    from operador.run import clasificar_v2

    runs = runs if runs is not None else analysis_load.list_runs(state_dir)
    v2 = clasificar_v2(runs)
    dmd = {f"dmd-{a}-s{s}": pd.concat([_preds(state_dir, r) for r in rs], ignore_index=True)
           for (a, s), rs in sorted(v2["dmd"].items())}
    lgbm = pd.concat([_preds(state_dir, r) for r in v2["lgbm"]], ignore_index=True)
    pers = _preds(state_dir, v2["pers"])
    return evm.construir_eval_rows(dmd, lgbm, pers)


def run_ids_referencia(runs: list[dict]) -> list[str]:
    """Runs a descargar para las filas fijas y la referencia de 7.1 (descarga selectiva, D1)."""
    from operador.run import clasificar_v2, ids_descarga

    ids = ids_descarga(clasificar_v2(runs))
    nombres = {RUN_71.format(seed=s) for s in SEEDS_71}
    ids += [r["run_id"] for r in runs if r.get("run_name") in nombres and not r.get("tags", {}).get("descartado")]
    return ids


def puntaje_71(state_dir: str, runs: list[dict], ev: evm.Evaluador) -> evm.Puntaje | None:
    """Puntaje de GRU+DMD+Duan de 7.1 (media de semillas) sobre las filas del evaluador."""
    items = []
    for s in SEEDS_71:
        run = next((r for r in runs if r.get("run_name") == RUN_71.format(seed=s)
                    and not r.get("tags", {}).get("descartado")), None)
        if run is None:
            continue
        f = _preds(state_dir, run)
        y = ev.rows[evm.KEYS].merge(f[evm.KEYS + ["y_pred"]], on=evm.KEYS, how="left")["y_pred"].to_numpy(float)
        items.append(ev.puntaje(y))
    return evm.Puntaje.promedio(items) if items else None


def posiciones_filas(grid, rows: pd.DataFrame) -> np.ndarray:
    stamps = pd.DatetimeIndex(pd.to_datetime(rows["timestamp"]).unique())
    idx = grid.index
    if idx.tz is not None:
        stamps = stamps.tz_localize(idx.tz) if stamps.tz is None else stamps.tz_convert(idx.tz)
    pos = idx.get_indexer(stamps)
    if (pos < 0).any():
        raise ValueError("Timestamps de las filas fuera de la grilla")
    return np.sort(pos)


def factores_duan(modelo: KoopmanModelo, ctx, positions_train) -> np.ndarray:
    """`S_{s,k}` de Duan con rollouts de train cuyo `t+24` está en train (D7 de 7.1)."""
    _, lp = pronostico(modelo, ctx.ng, ctx.norm, positions_train)
    log_real, real = ro.ventanas_reales_log(ctx.grid, positions_train)
    return ro.duan_factores(log_real, lp, real)


def evaluar_modelo(modelo: KoopmanModelo, ctx, ev: evm.Evaluador, nombre: str, duan_S: np.ndarray | None = None,
                   ref71: evm.Puntaje | None = None) -> dict:
    """Pronóstico sobre las filas fijas, puntaje, tabla D12, diferencia vs 7.1 y diagnósticos."""
    pos = posiciones_filas(ctx.grid, ev.rows)
    forecast, lp = pronostico(modelo, ctx.ng, ctx.norm, pos)
    dec = np.clip(np.expm1(lp), 0.0, None)
    if duan_S is not None:
        dec = np.clip(np.exp(lp) * duan_S[None] - 1.0, 0.0, None)  # = ro.aplicar_duan por hora
        forecast = dec.mean(axis=1)
    yhat = ev.alinear(nombre, ctx.grid.index[pos], forecast)
    p = ev.puntaje(yhat)
    tabla = ev.tabla(p)
    out = {"nombre": nombre, "yhat": yhat, "puntaje": p, "tabla": tabla, "positions": pos,
           "frames": {"predictions.parquet": ev.predicciones(yhat),
                      "horizon.parquet": evm.horizonte(dec, pos, ctx.grid),
                      "episodes.parquet": evm.episodios(ev, yhat), "tramos.parquet": evm.tramos(ev, yhat),
                      "bootstrap.parquet": tabla,
                      "spectrum.parquet": modos.espectro(modelo.op.autovalores())}}
    if ref71 is not None:
        out["diff_71"] = evm.diferencia(p, ref71)
    return out


def pesos_por_timestamp(ev: evm.Evaluador, positions, grid) -> np.ndarray:
    """`w [N_pos, 6]` de la métrica por timestamp de las filas (0 si la serie no está en esa fila)."""
    stamps = grid.index[np.asarray(positions, int)]
    ts = pd.DatetimeIndex(pd.to_datetime(ev.rows["timestamp"]))
    if ts.tz is None and stamps.tz is not None:
        ts = ts.tz_localize(stamps.tz)
    ti = stamps.get_indexer(ts)
    w = np.zeros((len(stamps), 6))
    ok = ti >= 0
    w[ti[ok], ev.canal[ok]] = ev.w[ok]
    return w
