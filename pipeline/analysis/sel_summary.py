"""Resumen de la validación de selección `sel` (change `ae-dmd-prep`, D10).

Compara, sobre **filas comunes**, las predicciones de `sel` de: persistencia, LightGBM
reajustado (`lightgbm-aedmd`) y los pases DMD por semilla. Además tabla los runs AE
(compuerta, linealidad y contraste con `pydmd`).

Descarga selectiva (D6): baja solo `mlflow.sqlite` y, después, los artefactos de los runs
de la partición pedida. Sin runs de la partición, avisa y sale con 0.

Uso (desde la raíz del repo, con el venv):

    python pipeline/analysis/sel_summary.py --partition ae-dmd --out output/sel_summary_ae_dmd.md

Sin `--state-dir` usa un directorio temporal (descarga desde S3 con el profile de la sesión).
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys

import numpy as np
import pandas as pd

# `pipeline/` es la raíz de importación (`analysis`, `evaluate`, ...) también al ejecutar este archivo.
_PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from analysis import load as analysis_load  # noqa: E402
from persist.s3 import get_client  # noqa: E402
from evaluate.evaluate import ALPHA_DEFAULT, rmse, sample_weights, weighted_rmse  # noqa: E402

log = logging.getLogger(__name__)

KEYS = ["estacion", "pollutant", "timestamp"]
PRED_COLUMNS = ["timestamp", "estacion", "pollutant", "y_true", "y_pred"]
AE_METRICS = ("gate_pass", "sel_recon_rmse_mean", "dmd_sel_latent_rmse_1", "dmd_sel_latent_rmse_24", "dmd_pydmd_ok")
PERSISTENCE_LABEL = "persistencia"
LIGHTGBM_LABEL = "lightgbm"


# --- Filas comunes y métricas ------------------------------------------------


def common_rows(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Intersección por `(estacion, pollutant, timestamp)` entre todas las etiquetas."""
    if not frames:
        return {}
    # Filas comunes = filas con valor finito en TODAS las etiquetas (si no, n difiere por etiqueta).
    finite = {
        label: frame[np.isfinite(frame["y_true"].to_numpy(dtype=float)) & np.isfinite(frame["y_pred"].to_numpy(dtype=float))]
        for label, frame in frames.items()
    }
    cleaned = {label: frame.drop_duplicates(KEYS, keep="last") for label, frame in finite.items()}
    common = None
    for frame in cleaned.values():
        index = pd.MultiIndex.from_frame(frame[KEYS])
        common = index if common is None else common.intersection(index)
    out: dict[str, pd.DataFrame] = {}
    for label, frame in cleaned.items():
        index = pd.MultiIndex.from_frame(frame[KEYS])
        out[label] = frame[index.isin(common)].sort_values(KEYS).reset_index(drop=True)
    return out


def _series_rows(frame: pd.DataFrame, alpha: float) -> list[dict]:
    rows: list[dict] = []
    for (station, pollutant), group in frame.groupby(["estacion", "pollutant"]):
        y = group["y_true"].to_numpy(dtype=float)
        yhat = group["y_pred"].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(yhat)
        y, yhat = y[ok], yhat[ok]
        if y.size == 0:
            continue
        rows.append({
            "estacion": station,
            "pollutant": pollutant,
            "rmse": rmse(y, yhat),
            "weighted_rmse": weighted_rmse(y, yhat, sample_weights(y, pollutant, alpha)),
            "n": int(y.size),
        })
    return rows


def summary_tables(frames: dict[str, pd.DataFrame], alpha: float = ALPHA_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla por serie y tabla de la media simple de las series, por etiqueta (sobre filas comunes)."""
    common = common_rows(frames)
    per_series: list[dict] = []
    means: list[dict] = []
    for label, frame in common.items():
        rows = _series_rows(frame, alpha)
        per_series.extend({"modelo": label, **row} for row in rows)
        if rows:
            means.append({
                "modelo": label,
                "rmse": float(np.mean([r["rmse"] for r in rows])),
                "weighted_rmse": float(np.mean([r["weighted_rmse"] for r in rows])),
                "n": min(r["n"] for r in rows),
                "series": len(rows),
            })
    return pd.DataFrame(per_series), pd.DataFrame(means)


def select_version(runs: list[dict], version: int | None = None) -> tuple[list[dict], int | None]:
    """Filtra por el param `version` (13.3). Sin `version` usa la mayor presente (la corrida vigente)."""
    def _version(run: dict) -> int | None:
        raw = run.get("params", {}).get("version")
        try:
            return int(float(raw)) if raw is not None else None
        except ValueError:
            return None

    present = [v for v in (_version(r) for r in runs) if v is not None]
    chosen = version if version is not None else (max(present) if present else None)
    selected = [r for r in runs if chosen is not None and _version(r) == chosen]
    return selected, chosen


def ae_table(runs: list[dict]) -> pd.DataFrame:
    """Una fila por run AE con la compuerta, la linealidad y el contraste con `pydmd`."""
    rows = []
    for run in runs:
        metrics = run.get("metrics", {})
        row = {"run": run.get("run_name")}
        row.update({name: metrics.get(name, float("nan")) for name in AE_METRICS})
        rows.append(row)
    return pd.DataFrame(rows, columns=["run", *AE_METRICS])


# --- Carga desde MLflow ------------------------------------------------------


def label_for(run: dict) -> str | None:
    """Etiqueta de comparación de un run de la partición (None si no entra en el resumen)."""
    params = run.get("params", {})
    tags = run.get("tags", {})
    if params.get("tipo") == "referencia" and not str(tags.get("modelo", "")).startswith("persistence"):
        return None
    modelo = str(tags.get("modelo", ""))
    if modelo.startswith("persistence"):
        return PERSISTENCE_LABEL
    if modelo.startswith("lightgbm"):
        return LIGHTGBM_LABEL
    if modelo.startswith("ae-") and modelo.endswith("-dmd"):
        arch = modelo[len("ae-"):-len("-dmd")]
        return f"dmd-{arch}-s{tags.get('seed', '?')}"
    return None


def load_sel_predictions(state_dir: str, runs: list[dict]) -> dict[str, pd.DataFrame]:
    """`predictions.parquet` (split = sel) agrupado por etiqueta."""
    parts: dict[str, list[pd.DataFrame]] = {}
    for run in runs:
        label = label_for(run)
        if label is None:
            continue
        predictions = analysis_load.load_artifacts(state_dir, run["run_id"])["predictions.parquet"]
        if predictions is None or predictions.empty or "split" not in predictions.columns:
            log.warning("El run %s no tiene predicciones: se omite de %s.", run.get("run_name"), label)
            continue
        selected = predictions[predictions["split"] == "sel"]
        if not selected.empty:
            parts.setdefault(label, []).append(selected[PRED_COLUMNS])
    return {label: pd.concat(frames, ignore_index=True) for label, frames in parts.items()}


# --- Markdown ----------------------------------------------------------------


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(sin filas)_"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("" if not np.isfinite(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(partition: str, per_series: pd.DataFrame, means: pd.DataFrame, ae: pd.DataFrame,
                    version: int | None = None) -> str:
    n_common = int(means["n"].min()) if not means.empty else 0
    title_version = f", versión {version}" if version is not None else ""
    return "\n".join([
        f"# Resumen de `sel` - partición `{partition}`{title_version}",
        "",
        f"Filas comunes a todas las etiquetas (valor finito en todas): n mínimo por serie = {n_common}; "
        "en la tabla por serie, n es el de cada serie tras la intersección. Métricas sobre `sel`, con embargo de 24 h.",
        "",
        "## Media de las 6 series",
        "",
        _md_table(means),
        "",
        "## Por serie",
        "",
        _md_table(per_series),
        "",
        "## Runs AE",
        "",
        _md_table(ae),
        "",
    ])


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Resumen de la validación de selección `sel` (ae-dmd-prep).")
    parser.add_argument("--partition", default="ae-dmd", help="Valor del tag `particion` de los runs a resumir.")
    parser.add_argument("--state-dir", default=None, help="Directorio local del estado MLflow (defecto: temporal, desde S3).")
    parser.add_argument("--experiment", default=analysis_load.EXPERIMENT_NAME)
    parser.add_argument("--out", default=None,
                        help="Markdown de salida (defecto output/sel_summary_<partición>_v<n>.md).")
    parser.add_argument("--version", type=int, default=None,
                        help="Filtra el param `version` de los runs (defecto: la mayor presente en la partición).")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--profile", default=None,
                        help="Profile AWS de la cuenta personal (obligatorio: el valor de AWS_PROFILE en .env).")
    args = parser.parse_args(argv)
    if not args.profile:
        parser.error("--profile es obligatorio (valor de AWS_PROFILE en .env); no se usa el profile default.")
    client = get_client(args.profile)

    state = analysis_load.ensure_state(data_dir=args.state_dir, run_ids=[], client=client)  # solo mlflow.sqlite
    runs = [run for run in analysis_load.list_runs(state, experiment_name=args.experiment)
            if run.get("tags", {}).get("particion") == args.partition]
    runs, version = select_version(runs, args.version)
    if not runs:
        print(f"Sin runs con particion={args.partition}"
              f"{'' if version is None else f' y version={version}'} en el experimento {args.experiment}.")
        return 0

    analysis_load.ensure_state(data_dir=state, run_ids=[run["run_id"] for run in runs], client=client)  # artefactos
    frames = load_sel_predictions(state, runs)
    per_series, means = summary_tables(frames, args.alpha)
    ae = ae_table([run for run in runs if analysis_load.is_ae_run(run)])
    markdown = render_markdown(args.partition, per_series, means, ae, version)

    token = args.partition.replace("-", "_")
    out = pathlib.Path(args.out or f"output/sel_summary_{token}_v{version}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
