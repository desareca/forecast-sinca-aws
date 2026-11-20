"""Publica en S3 los runs nuevos del operador global (`dmd-operador-global`, D1). Script simple.

- `plan`: lista los runs locales con `stage=dmd-operador-global` (sin tag `descartado`) aún no publicados (según
  `<state-dir>/.download.json`) y falla si alguno no tiene artefactos.
- `apply --yes`: aborta si el `LastModified` remoto de `mlflow.sqlite` difiere del guardado en
  `.download.json`; respalda el sqlite remoto en `_mlflow/backups/mlflow-<AAAAMMDD-HHMM>.sqlite`;
  sube el sqlite y solo `mlruns/<run_id>/` de los runs nuevos; reescribe `.download.json` con el
  nuevo `LastModified` (`head_object`). Sin borrados. Sin `--yes` no escribe.

Uso (raíz del repo): `python scripts/mlflow/publish_state.py plan --state-dir .\\mlflow-op --profile <AWS_PROFILE>`
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

_PIPELINE = pathlib.Path(__file__).resolve().parents[2] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from train import mlflow_log  # noqa: E402

BUCKET = mlflow_log.MLFLOW_BUCKET_DEFAULT
PREFIX = mlflow_log.MLFLOW_PREFIX_DEFAULT
DB_FILENAME = mlflow_log.DB_FILENAME
STAGE = "dmd-operador-global"
META = ".download.json"


def leer_meta(state_dir: str) -> dict:
    path = pathlib.Path(state_dir, META)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def escribir_meta(state_dir: str, meta: dict) -> None:
    pathlib.Path(state_dir, META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def plan(state_dir: str, stage: str = STAGE) -> list[tuple[str, str, list[pathlib.Path]]]:
    try:
        return mlflow_log.runs_nuevos(state_dir, stage, leer_meta(state_dir).get("published_runs", []))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def apply(state_dir: str, client, bucket: str = BUCKET, prefix: str = PREFIX, now: dt.datetime | None = None,
          stage: str = STAGE) -> list[str]:
    """Publica con `train.mlflow_log.publish_new_runs` y actualiza `.download.json`."""
    meta = leer_meta(state_dir)
    plan(state_dir, stage)
    try:
        res = mlflow_log.publish_new_runs(state_dir, stage, client, meta.get("last_modified"), bucket, prefix,
                                          published=meta.get("published_runs", []), now=now)
    except mlflow_log.PublishConflict as exc:
        raise SystemExit(f"{exc} ({META})") from exc
    if not res["runs"]:
        return []
    meta["last_modified"] = res["last_modified"]
    meta["published_runs"] = sorted(set(meta.get("published_runs", [])) | set(res["runs"]))
    meta.setdefault("backups", []).append(res["backup"])
    escribir_meta(state_dir, meta)
    return res["uploaded"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publica runs nuevos del operador global en S3.")
    parser.add_argument("accion", choices=("plan", "apply"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--profile", default=None, help="Profile AWS de la cuenta personal (obligatorio).")
    parser.add_argument("--yes", action="store_true", help="Confirma la escritura en S3 (solo `apply`).")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--stage", default=STAGE, help="Tag `stage` de los runs a publicar (defecto dmd-operador-global).")
    args = parser.parse_args(argv)
    if not args.profile:
        parser.error("--profile es obligatorio (valor de AWS_PROFILE en .env).")
    nuevos = plan(args.state_dir, args.stage)
    print(f"Runs nuevos ({len(nuevos)}):")
    for run_id, name, files in nuevos:
        print(f"  {name}  {run_id}  ({len(files)} archivos)")
    if args.accion == "plan":
        return 0
    if not args.yes:
        print("Sin --yes no se escribe nada.")
        return 1
    import boto3

    client = boto3.Session(profile_name=args.profile).client("s3")
    uploaded = apply(args.state_dir, client, args.bucket, stage=args.stage)
    if not uploaded:
        print("Sin runs nuevos: no se escribe nada.")
        return 0
    print(f"Subidos {len(uploaded)} objetos (respaldo: {uploaded[0]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
