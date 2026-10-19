# Imagen de entrenamiento del baseline (Fase 3).
#
# Lleva SOLO el runtime (patrón `compute`): el código del modelo viaja por
# `source_dir`/`entry_point` del SDK de SageMaker, no va horneado acá. Rebuild
# solo cuando cambian las dependencias.
#
# Build desde la raíz del repo:
#   docker build -f docker/train.Dockerfile -t sinca-train .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_DEFAULT_REGION=us-east-1

# LightGBM y scikit-learn enlazan contra OpenMP (libgomp), y `sagemaker-training`
# solo se publica como sdist (compila una extensión C mínima) -> necesita gcc.
# Nada de esto viene en `python:3.11-slim`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime del entrenamiento (+ `sagemaker-training`: provee el entrypoint
# `train` del script mode, que descarga el `source_dir` y ejecuta el entry_point).
RUN pip install --no-cache-dir \
    "lightgbm>=4.0,<5.0" \
    "pandas>=2.0,<3.0" \
    "numpy>=1.26,<3.0" \
    "scikit-learn>=1.4,<2.0" \
    "mlflow>=3.0,<4.0" \
    "boto3>=1.34,<2.0" \
    "pyarrow>=15.0,<22.0" \
    "sagemaker-training>=4.7,<5.0"

ENTRYPOINT ["train"]
