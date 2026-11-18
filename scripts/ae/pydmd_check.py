"""Contraste numpy vs `pydmd` del DMD (change `ae-dmd-prep`, D8; plan B).

Dos modos:

- `--synthetic`: sistema lineal estable conocido (sin pandas ni MLflow). Es la prueba de
  humo de la imagen `tf-cpu-v2` y del venv de `requirements-dmd.txt`.
- `--state-dir DIR --run-ids ID[,ID]`: descarga selectiva de los runs AE (solo sus artefactos),
  lee `latent.parquet` con `split = train`, parte en segmentos de horas consecutivas, toma el
  más largo y contrasta rango completo y truncado `d // 2`.

Imprime una tabla y sale con 1 si algún contraste no es `ok`. Requiere `pydmd` instalado.

    python scripts/ae/pydmd_check.py --synthetic
    python scripts/ae/pydmd_check.py --state-dir .\\mlflow-maint --run-ids <id1>,<id2>
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_PIPELINE = pathlib.Path(__file__).resolve().parents[2] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from ae import dmd as dmdm  # noqa: E402  (solo numpy)

HEADER = ("caso", "rango", "eig_maxdiff", "pred1_maxdiff", "ok", "pydmd")


def stable_linear_system(d: int = 6, steps: int = 240, seed: int = 0) -> np.ndarray:
    """Trayectoria de `z_{t+1} = A z_t` con autovalores de módulo 0.9 (sin ruido)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    blocks = np.zeros((d, d))
    for i, theta in enumerate(np.linspace(0.2, 1.1, d // 2)):
        rot = 0.9 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        blocks[2 * i:2 * i + 2, 2 * i:2 * i + 2] = rot
    a = q @ blocks @ q.T
    z = np.zeros((steps, d))
    z[0] = rng.standard_normal(d)
    for t in range(steps - 1):
        z[t + 1] = a @ z[t]
    return z


def _row(case: str, rank_label: str, result: dict) -> dict:
    return {
        "caso": case,
        "rango": rank_label,
        "eig_maxdiff": f"{result['eig_maxdiff']:.2e}",
        "pred1_maxdiff": "-" if not np.isfinite(result["pred1_maxdiff"]) else f"{result['pred1_maxdiff']:.2e}",
        "ok": "ok" if result["ok"] else "FALLA",
        "pydmd": result["version"],
    }


def check_segment(case: str, segment: np.ndarray, d: int) -> list[dict]:
    full = dmdm.pydmd_crosscheck(segment, None)
    trunc = dmdm.pydmd_crosscheck(segment, max(2, d // 2))
    return [_row(case, "completo", full), _row(case, f"truncado {max(2, d // 2)}", trunc)]


def _table(rows: list[dict]) -> str:
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in HEADER}
    line = " | ".join(h.ljust(widths[h]) for h in HEADER)
    sep = "-+-".join("-" * widths[h] for h in HEADER)
    body = [" | ".join(str(r[h]).ljust(widths[h]) for h in HEADER) for r in rows]
    return "\n".join([line, sep, *body])


def synthetic_rows() -> list[dict]:
    segment = stable_linear_system(d=6)
    return check_segment("sintetico-d6", segment, d=6)


def run_rows(state_dir: str, run_ids: list[str], profile: str) -> list[dict]:
    import pandas as pd

    from analysis import load as analysis_load
    from persist.s3 import get_client

    state = analysis_load.ensure_state(data_dir=state_dir, run_ids=run_ids, client=get_client(profile))
    rows: list[dict] = []
    for run_id in run_ids:
        path = analysis_load.artifact_path(state, run_id, "latent.parquet")
        if path is None:
            raise SystemExit(f"El run {run_id} no tiene latent.parquet en el estado local.")
        latent = pd.read_parquet(path)
        train = latent[latent["split"] == "train"].sort_values("timestamp").reset_index(drop=True)
        cols = [c for c in train.columns if c not in ("timestamp", "split")]
        stamps = pd.to_datetime(train["timestamp"])
        gaps = stamps.diff().dt.total_seconds().fillna(3600.0).ne(3600.0).cumsum()
        segments = [g[cols].to_numpy(dtype=float) for _, g in train.groupby(gaps)]
        segment = dmdm.longest_segment(segments)
        rows += check_segment(run_id[:8], segment, d=len(cols))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contraste numpy vs pydmd del DMD (ae-dmd-prep, D8).")
    parser.add_argument("--synthetic", action="store_true", help="Sistema lineal conocido, sin pandas ni MLflow.")
    parser.add_argument("--state-dir", default=None, help="Estado MLflow local (se completa con descarga selectiva).")
    parser.add_argument("--run-ids", default=None, help="IDs de runs AE separados por coma.")
    parser.add_argument("--profile", default=None,
                        help="Profile AWS de la cuenta personal (obligatorio con --state-dir: valor de AWS_PROFILE en .env).")
    args = parser.parse_args(argv)
    if args.state_dir and not args.profile:
        parser.error("--profile es obligatorio con --state-dir (valor de AWS_PROFILE en .env); no se usa el profile default.")

    try:
        import pydmd  # noqa: F401
    except ImportError:
        print("pydmd no está instalado: usar la imagen tf-cpu-v2 o requirements-dmd.txt.")
        return 1

    if args.synthetic:
        rows = synthetic_rows()
    elif args.state_dir and args.run_ids:
        rows = run_rows(args.state_dir, [r for r in args.run_ids.split(",") if r], args.profile)
    else:
        parser.error("usar --synthetic o --state-dir con --run-ids")
        return 2

    print(_table(rows))
    return 0 if all(row["ok"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
