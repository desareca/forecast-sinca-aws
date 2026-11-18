"""Runs fuente de LightGBM para el reajuste de `ae-dmd-prep` (tarea 11.3). Solo lectura.

Lista, por serie, los runs `lightgbm` `version = 1` activos del `mlflow.sqlite` local (sqlite en
modo `ro`), con su `run_id`, `start_time` y sus hiperparámetros. Imprime los comandos
`aws s3 ls` para confirmar cuál tiene `predictions.parquet` (el operador lo corre) y arma el
string `slug:pollutant=run_id,...` con el run más reciente de cada serie (regla por defecto,
Pregunta 3). El operador confirma el string antes de usarlo en `run_ae_dmd_prep.ps1`.

No llama a AWS. Uso desde la raíz del repo:

    python scripts/mlflow/lgbm_sources.py --state-dir .\\mlflow-maint
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sqlite3
import sys

_PIPELINE = pathlib.Path(__file__).resolve().parents[2] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from config import STATIONS  # noqa: E402

DB_FILENAME = "mlflow.sqlite"
S3_PREFIX = "_mlflow"
BUCKET_PLACEHOLDER = "sinca-mlflow"
EXPERIMENT_DEFAULT = "sinca-baseline"
PROFILE = "<AWS_PROFILE>"
HP_SHOWN = ("learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree",
            "reg_alpha", "reg_lambda")


def slug(station: str) -> str:
    return STATIONS.get(station, {}).get("slug", station.lower().replace(" ", "_"))


def lgbm_v1_runs(state_dir: str, experiment: str) -> list[dict]:
    path = pathlib.Path(state_dir, DB_FILENAME).resolve()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        exp = con.execute("select experiment_id from experiments where name = ?", (experiment,)).fetchone()
        if not exp:
            return []
        rows = con.execute(
            "select run_uuid, name, experiment_id, start_time from runs "
            "where experiment_id = ? and lifecycle_stage = 'active'",
            (exp[0],),
        ).fetchall()
        out: list[dict] = []
        for run_id, name, experiment_id, start_time in rows:
            params = dict(con.execute("select key, value from params where run_uuid = ?", (run_id,)).fetchall())
            tags = dict(con.execute("select key, value from tags where run_uuid = ?", (run_id,)).fetchall())
            if tags.get("modelo") != "lightgbm" or params.get("version") != "1":
                continue
            if not params.get("estacion") or not params.get("pollutant"):
                continue
            out.append({
                "run_id": run_id,
                "name": name,
                "experiment_id": str(experiment_id),
                "estacion": params["estacion"],
                "pollutant": params["pollutant"],
                "start": dt.datetime.fromtimestamp(start_time / 1000).isoformat(timespec="minutes") if start_time else "?",
                "start_ms": start_time or 0,
                "hp": {k: params.get(k, "?") for k in HP_SHOWN},
            })
        return sorted(out, key=lambda r: (r["estacion"], r["pollutant"], -r["start_ms"]))
    finally:
        con.close()


def render(runs: list[dict]) -> str:
    lines = ["serie | run_id | start | " + " | ".join(HP_SHOWN)]
    for r in runs:
        hp = " | ".join(str(r["hp"][k]) for k in HP_SHOWN)
        lines.append(f"{r['estacion']}/{r['pollutant']} | {r['run_id']} | {r['start']} | {hp}")
    return "\n".join(lines)


def suggested_string(runs: list[dict]) -> tuple[str, list[str]]:
    """Run más reciente por serie y avisos si la cuenta por serie no es 1."""
    newest: dict[tuple[str, str], dict] = {}
    counts: dict[tuple[str, str], int] = {}
    for r in runs:  # ya ordenados por start descendente dentro de la serie
        key = (r["estacion"], r["pollutant"])
        counts[key] = counts.get(key, 0) + 1
        newest.setdefault(key, r)
    warnings = [f"{k[0]}/{k[1]}: {n} runs v1 activos (revisar cuál usar)" for k, n in counts.items() if n != 1]
    parts = [f"{slug(est)}:{pol}={r['run_id']}" for (est, pol), r in sorted(newest.items())]
    return ",".join(parts), warnings


def artifact_prefixes(run: dict) -> list[str]:
    """Prefijos S3 de los artefactos del run, en orden: primero `mlruns/<run>/`, luego `mlruns/<exp>/<run>/`."""
    return [
        f"{S3_PREFIX}/mlruns/{run['run_id']}/",
        f"{S3_PREFIX}/mlruns/{run['experiment_id']}/{run['run_id']}/",
    ]


def resolve_predictions_prefix(run: dict, existing_keys: set[str]) -> str | None:
    """Primer prefijo de artefactos con `predictions.parquet` entre las claves dadas (None si no hay)."""
    for prefix in artifact_prefixes(run):
        if f"{prefix}artifacts/predictions.parquet" in existing_keys:
            return prefix
    return None


def s3_check_commands(runs: list[dict], profile: str = PROFILE) -> list[str]:
    """Un `aws s3 ls` por layout. Si el primero no devuelve nada, usar el segundo."""
    cmds = []
    for r in runs:
        for prefix in artifact_prefixes(r):
            key = f"{prefix}artifacts/predictions.parquet"
            cmds.append(f"aws s3 ls s3://{BUCKET_PLACEHOLDER}/{key} --profile {profile}")
    return cmds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runs fuente LightGBM v1 por serie (ae-dmd-prep, 11.3). Solo lectura.")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--experiment", default=EXPERIMENT_DEFAULT)
    parser.add_argument("--profile", default=None,
                        help="Profile AWS de la cuenta personal (obligatorio: valor de AWS_PROFILE en .env).")
    args = parser.parse_args(argv)
    if not args.profile:
        parser.error("--profile es obligatorio (valor de AWS_PROFILE en .env); no se usa el profile default.")

    runs = lgbm_v1_runs(args.state_dir, args.experiment)
    if not runs:
        print("Sin runs lightgbm v1 activos en el experimento.")
        return 1
    print(render(runs))
    print("\nComprobar predictions.parquet (el operador corre estos comandos):")
    print("\n".join(s3_check_commands(runs, args.profile)))
    string, warnings = suggested_string(runs)
    print("\n--source-run-ids sugerido (run más reciente por serie):")
    print(string)
    for warning in warnings:
        print(f"AVISO: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
