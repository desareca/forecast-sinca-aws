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


def download_state(bucket: str, prefix: str, local_dir: str, client=None) -> list[str]:
    """Baja `mlflow.sqlite` y `mlruns/` de S3 al directorio local (si existen)."""
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
