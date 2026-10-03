#!/bin/bash
set -euo pipefail

# Servidor MLflow on-demand con backend SQLite persistido en S3.
#
# Ciclo de vida:
#   1. Al arrancar, descarga `mlflow.sqlite` de S3 (si existe).
#   2. Levanta `mlflow server` con backend SQLite local y artefactos directo a S3.
#   3. Al recibir SIGTERM/SIGINT (Fargate al hacer `stop-task`), sube el
#      `.sqlite` de vuelta a S3 antes de salir.

MLFLOW_BUCKET="${MLFLOW_BUCKET:-sinca-mlflow}"
MLFLOW_PREFIX="${MLFLOW_PREFIX:-_mlflow}"
DB_FILE="${MLFLOW_DB_FILE:-/mlflow.sqlite}"
DB_KEY="${MLFLOW_PREFIX}/mlflow.sqlite"
ARTIFACTS_DEST="s3://${MLFLOW_BUCKET}/${MLFLOW_PREFIX}/mlruns"

echo "[entrypoint] bucket=s3://${MLFLOW_BUCKET} prefijo=${MLFLOW_PREFIX} db=${DB_FILE}"

# 1. Descargar el backend SQLite de S3 (si existe).
python - "$MLFLOW_BUCKET" "$DB_KEY" "$DB_FILE" <<'PY'
import sys

import boto3
from botocore.exceptions import ClientError

bucket, key, db_file = sys.argv[1], sys.argv[2], sys.argv[3]
s3 = boto3.client("s3")
try:
    s3.download_file(bucket, key, db_file)
    print(f"[entrypoint] backend descargado: s3://{bucket}/{key}")
except ClientError as exc:
    code = exc.response.get("Error", {}).get("Code", "")
    if code in ("404", "NoSuchKey"):
        print(f"[entrypoint] no existe s3://{bucket}/{key}; arrancando backend nuevo")
    else:
        raise
PY

# 2. Subir el backend a S3 (se usa al apagar).
upload_backend() {
  if [ ! -f "$DB_FILE" ]; then
    echo "[entrypoint] no hay $DB_FILE para subir"
    return 0
  fi
  echo "[entrypoint] subiendo backend a s3://${MLFLOW_BUCKET}/${DB_KEY}"
  python - "$MLFLOW_BUCKET" "$DB_KEY" "$DB_FILE" <<'PY'
import sys

import boto3

bucket, key, db_file = sys.argv[1], sys.argv[2], sys.argv[3]
boto3.client("s3").upload_file(db_file, bucket, key)
print(f"[entrypoint] backend subido: s3://{bucket}/{key}")
PY
}

# 3. Arrancar el servidor MLflow en background para poder capturar señales.
#    Se usa un backend SQLite con ruta absoluta (`sqlite:////mlflow.sqlite`).
#    `--allowed-hosts` con `*` desactiva la validación de Host header (MLflow 3.x
#    rechaza por defecto los Host no-locales con HTTP 403); el acceso ya está
#    restringido por el security group al IP del operador.
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "sqlite:///${DB_FILE}" \
  --artifacts-destination "${ARTIFACTS_DEST}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-*}" &
MLFLOW_PID=$!

shutdown() {
  echo "[entrypoint] señal de apagado recibida; deteniendo mlflow (pid ${MLFLOW_PID})"
  kill -TERM "$MLFLOW_PID" 2>/dev/null || true
  wait "$MLFLOW_PID" 2>/dev/null || true
  upload_backend
  exit 0
}
trap shutdown SIGTERM SIGINT

# Si mlflow termina solo (crash o fin normal), subir igual y propagar el estado.
set +e
wait "$MLFLOW_PID"
STATUS=$?
set -e
upload_backend
exit "$STATUS"
