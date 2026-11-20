"""Publica en S3 los runs nuevos del operador global (`dmd-operador-global`, D1). Script simple.

- `plan`: lista los runs locales con `stage=dmd-operador-global` aún no publicados (según
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
import sqlite3
import sys

BUCKET = "sinca-mlflow"
PREFIX = "_mlflow"
DB_FILENAME = "mlflow.sqlite"
STAGE = "dmd-operador-global"
META = ".download.json"


def leer_meta(state_dir: str) -> dict:
    path = pathlib.Path(state_dir, META)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def escribir_meta(state_dir: str, meta: dict) -> None:
    pathlib.Path(state_dir, META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _iso(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def runs_stage(state_dir: str) -> list[tuple[str, str]]:
    path = pathlib.Path(state_dir, DB_FILENAME).resolve()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select r.run_uuid, r.name from runs r join tags t on t.run_uuid = r.run_uuid "
            "where t.key = 'stage' and t.value = ? and r.lifecycle_stage = 'active' order by r.name", (STAGE,)
        ).fetchall()
    finally:
        con.close()
    return [(r[0], r[1]) for r in rows]


def archivos_run(state_dir: str, run_id: str) -> list[pathlib.Path]:
    base = pathlib.Path(state_dir, "mlruns", run_id)
    return sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []


def plan(state_dir: str) -> list[tuple[str, str, list[pathlib.Path]]]:
    published = set(leer_meta(state_dir).get("published_runs", []))
    out = []
    missing = []
    for run_id, name in runs_stage(state_dir):
        if run_id in published:
            continue
        files = archivos_run(state_dir, run_id)
        if not files:
            missing.append(name)
        out.append((run_id, name, files))
    if missing:
        raise SystemExit(f"Runs sin artefactos: {', '.join(missing)}")
    return out


def apply(state_dir: str, client, bucket: str = BUCKET, prefix: str = PREFIX, now: dt.datetime | None = None) -> list[str]:
    meta = leer_meta(state_dir)
    nuevos = plan(state_dir)
    if not nuevos:
        return []
    db_key = f"{prefix}/{DB_FILENAME}"
    remote = _iso(client.head_object(Bucket=bucket, Key=db_key)["LastModified"])
    if remote != meta.get("last_modified"):
        raise SystemExit(f"LastModified remoto ({remote}) difiere de {META} ({meta.get('last_modified')}): "
                         "otro proceso escribió el sqlite. No se sube nada.")
    stamp = (now or dt.datetime.now()).strftime("%Y%m%d-%H%M")
    backup = f"{prefix}/backups/mlflow-{stamp}.sqlite"
    client.copy_object(Bucket=bucket, Key=backup, CopySource={"Bucket": bucket, "Key": db_key})
    uploaded = [backup]
    client.upload_file(str(pathlib.Path(state_dir, DB_FILENAME)), bucket, db_key)
    uploaded.append(db_key)
    for run_id, _, files in nuevos:
        for path in files:
            rel = path.relative_to(pathlib.Path(state_dir)).as_posix()
            key = f"{prefix}/{rel}"
            client.upload_file(str(path), bucket, key)
            uploaded.append(key)
    meta["last_modified"] = _iso(client.head_object(Bucket=bucket, Key=db_key)["LastModified"])
    meta["published_runs"] = sorted(set(meta.get("published_runs", [])) | {r for r, _, _ in nuevos})
    meta.setdefault("backups", []).append(backup)
    escribir_meta(state_dir, meta)
    return uploaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publica runs nuevos del operador global en S3.")
    parser.add_argument("accion", choices=("plan", "apply"))
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--profile", default=None, help="Profile AWS de la cuenta personal (obligatorio).")
    parser.add_argument("--yes", action="store_true", help="Confirma la escritura en S3 (solo `apply`).")
    parser.add_argument("--bucket", default=BUCKET)
    args = parser.parse_args(argv)
    if not args.profile:
        parser.error("--profile es obligatorio (valor de AWS_PROFILE en .env).")
    nuevos = plan(args.state_dir)
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
    uploaded = apply(args.state_dir, client, args.bucket)
    if not uploaded:
        print("Sin runs nuevos: no se escribe nada.")
        return 0
    print(f"Subidos {len(uploaded)} objetos (respaldo: {uploaded[0]}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
