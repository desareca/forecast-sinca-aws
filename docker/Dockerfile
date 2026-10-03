FROM ghcr.io/mlflow/mlflow

# boto3 para el ciclo download/upload del backend SQLite contra S3.
# La imagen oficial de MLflow NO lo trae por defecto.
RUN python -m pip install --no-cache-dir boto3

# Entrypoint propio: descarga `mlflow.sqlite` de S3 al arrancar, levanta el
# servidor MLflow, y sube el `.sqlite` de vuelta al recibir SIGTERM/SIGINT.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
