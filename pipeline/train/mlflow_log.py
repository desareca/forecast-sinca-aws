"""Logging directo a MLflow (modo 1, sin servidor) y sincronización con S3.

El Training Job **no** levanta un servidor MLflow: configura el `tracking_uri` a
un archivo SQLite local, loguea métricas/parámetros/artefactos, y al terminar
sube `mlflow.sqlite` + `mlruns/` a `s3://<bucket>/<prefijo>/`.

Convención de nombres de run: `sinca-{estacion}-{modelo}-v{n}`.

Consistencia (SQLite no admite escrituras concurrentes): antes de loguear se
**descarga** el estado de S3 (para no perder historia) y al terminar se
**sube**. La escritura final se serializa (una corrida a la vez), tal como pide
el blueprint de tracking.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sqlite3

from botocore.exceptions import ClientError

from config import STATIONS
from persist.s3 import get_client

log = logging.getLogger(__name__)

MLFLOW_BUCKET_DEFAULT = "sinca-mlflow"
MLFLOW_PREFIX_DEFAULT = "_mlflow"
DB_FILENAME = "mlflow.sqlite"
ARTIFACTS_DIRNAME = "mlruns"
EXPERIMENT_NAME = "sinca-baseline"
MODEL_ARTIFACTS_PREFIX = "model-artifacts"

_MISSING_CODES = {"404", "NoSuchKey", "NoSuchBucket"}


def station_slug(station: str) -> str:
    return STATIONS.get(station, {}).get("slug", station.lower().replace(" ", "_"))


def run_name(station: str, model: str, version: int | None = None) -> str:
    """Nombre de run con la convención `sinca-{estacion}-{modelo}-v{n}`."""
    name = f"sinca-{station_slug(station)}-{model}"
    return f"{name}-v{version}" if version is not None else name


def _s3_uri(tracking_uri: str | None, db_path: str) -> str:
    if tracking_uri:
        return tracking_uri
    posix = pathlib.Path(db_path).as_posix()
    # `sqlite:///<path>` -> con path absoluto POSIX queda `sqlite:////opt/...`
    # (equivalente a `/opt/...`), y en Windows `sqlite:///C:/...`.
    return f"sqlite:///{posix}"


def configure(
    local_dir: str,
    db_filename: str = DB_FILENAME,
    artifacts_dirname: str = ARTIFACTS_DIRNAME,
    experiment: str = EXPERIMENT_NAME,
    tracking_uri: str | None = None,
):
    """Configura MLflow con backend SQLite + artefactos locales. Devuelve `(uri, db_path, artifacts_dir)`."""
    import mlflow

    os.makedirs(local_dir, exist_ok=True)
    db_path = os.path.join(local_dir, db_filename)
    artifacts_dir = os.path.join(local_dir, artifacts_dirname)
    os.makedirs(artifacts_dir, exist_ok=True)

    uri = _s3_uri(tracking_uri, db_path)
    mlflow.set_tracking_uri(uri)

    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_location = pathlib.Path(artifacts_dir).absolute().as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifact_location)
    mlflow.set_experiment(experiment)
    log.info("MLflow: tracking_uri=%s experiment=%s", uri, experiment)
    return uri, db_path, artifacts_dir


def _get_client(client=None):
    return client or get_client()


def parse_mlflow_runs(text: str | None) -> list[str] | None:
    """Interpreta `--mlflow-runs`: `None`/`""`/`all` -> todo (`None`); `none` -> `[]`; IDs separados por coma."""
    if text is None:
        return None
    value = text.strip()
    if value in ("", "all"):
        return None
    if value == "none":
        return []
    return [item for item in value.split(",") if item]


def _list_objects(client, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """Claves y tamaños bajo un prefijo (sin marcadores de directorio)."""
    out: list[tuple[str, int]] = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("/"):
                out.append((obj["Key"], int(obj.get("Size", -1))))
    return out


def _experiment_id(local_dir: str, run_id: str) -> str | None:
    """`experiment_id` del run en el `mlflow.sqlite` local (None si no está)."""
    db = os.path.join(local_dir, DB_FILENAME)
    if not os.path.exists(db):
        return None
    con = sqlite3.connect(db)
    try:
        row = con.execute("select experiment_id from runs where run_uuid = ?", (run_id,)).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        con.close()
    return str(row[0]) if row else None


def _download_selected_runs(client, bucket: str, prefix: str, local_dir: str, run_ids: list[str]) -> list[str]:
    """Baja solo los artefactos de los runs pedidos (layout con y sin experimento)."""
    downloaded: list[str] = []
    for run_id in run_ids:
        objects: list[tuple[str, int]] = []
        experiment_id = _experiment_id(local_dir, run_id)
        if experiment_id is not None:
            objects = _list_objects(client, bucket, f"{prefix}/{ARTIFACTS_DIRNAME}/{experiment_id}/{run_id}/")
        if not objects:
            objects = _list_objects(client, bucket, f"{prefix}/{ARTIFACTS_DIRNAME}/{run_id}/")
        if not objects:
            log.warning("Run %s sin objetos en s3://%s/%s; no se baja nada.", run_id, bucket, prefix)
            continue
        for key, size in objects:
            relative = key[len(prefix) + 1:]
            destination = os.path.join(local_dir, *relative.split("/"))
            if size >= 0 and os.path.exists(destination) and os.path.getsize(destination) == size:
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            client.download_file(bucket, key, destination)
            downloaded.append(key)
    return downloaded


def download_state(bucket: str, prefix: str, local_dir: str, client=None, run_ids: list[str] | None = None) -> list[str]:
    """Baja `mlflow.sqlite` y `mlruns/` de S3 al directorio local (si existen).

    `run_ids=None` baja todo (comportamiento original). Una lista (vacía o no) baja
    el sqlite y, por cada run, solo sus artefactos (`ae-dmd-prep`, D6); `[]` = solo sqlite.
    """
    client = _get_client(client)
    os.makedirs(local_dir, exist_ok=True)
    downloaded: list[str] = []

    db_key = f"{prefix}/{DB_FILENAME}"
    db_path = os.path.join(local_dir, DB_FILENAME)
    try:
        client.download_file(bucket, db_key, db_path)
        downloaded.append(db_key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in _MISSING_CODES:
            raise
        log.info("Estado previo ausente en s3://%s/%s (arranca vacío).", bucket, db_key)

    if run_ids is not None:
        downloaded.extend(_download_selected_runs(client, bucket, prefix, local_dir, run_ids))
        log.info("Estado MLflow descargado (selectivo): %d objetos.", len(downloaded))
        return downloaded

    paginator = client.get_paginator("list_objects_v2")
    objects_prefix = f"{prefix}/{ARTIFACTS_DIRNAME}/"
    for page in paginator.paginate(Bucket=bucket, Prefix=objects_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            relative = key[len(f"{prefix}/"):]
            destination = os.path.join(local_dir, *relative.split("/"))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            client.download_file(bucket, key, destination)
            downloaded.append(key)
    log.info("Estado MLflow descargado: %d objetos.", len(downloaded))
    return downloaded


def upload_state(bucket: str, prefix: str, local_dir: str, client=None) -> list[str]:
    """Sube `mlflow.sqlite` + `mlruns/` del directorio local a S3.

    Escritura final serializada (una corrida a la vez).
    """
    client = _get_client(client)
    uploaded: list[str] = []

    db_path = os.path.join(local_dir, DB_FILENAME)
    if os.path.exists(db_path):
        key = f"{prefix}/{DB_FILENAME}"
        client.upload_file(db_path, bucket, key)
        uploaded.append(key)

    artifacts_dir = os.path.join(local_dir, ARTIFACTS_DIRNAME)
    if os.path.isdir(artifacts_dir):
        for root, _, files in os.walk(artifacts_dir):
            for filename in files:
                full = os.path.join(root, filename)
                relative = os.path.relpath(full, local_dir).replace(os.sep, "/")
                key = f"{prefix}/{relative}"
                client.upload_file(full, bucket, key)
                uploaded.append(key)
    log.info("Estado MLflow subido a s3://%s/%s: %d objetos.", bucket, prefix, len(uploaded))
    return uploaded


def upload_model_artifact(local_path: str, bucket: str, run: str, filename: str | None = None, client=None) -> str:
    """Sube un artefacto de modelo a `s3://<bucket>/model-artifacts/<run>/<filename>`."""
    client = _get_client(client)
    name = filename or os.path.basename(local_path)
    key = f"{MODEL_ARTIFACTS_PREFIX}/{run}/{name}"
    client.upload_file(local_path, bucket, key)
    log.info("Artefacto de modelo subido: s3://%s/%s", bucket, key)
    return key
