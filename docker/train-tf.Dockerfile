# Imagen de entrenamiento TensorFlow de los autoencoders (change `ae-reconstruccion`).
#
# Lleva SOLO el runtime (patrón `compute`, igual que `train.Dockerfile`): el código
# viaja por `source_dir`/`entry_point` desde `pipeline/` (p. ej. `ae/run.py`).
#
# Tag explícito `tf-cpu-v1` (ECR `sinca-train`). NO se usa `:latest`, y la imagen
# LightGBM sigue en su tag `lgbm-v1`.
#
# Versión de TensorFlow FIJADA: debe coincidir con `requirements-notebooks-tf.txt`
# (el nivel 2 de los notebooks carga los `.keras` de esta imagen).
#
# Build desde la raíz del repo:
#   docker build -f docker/train-tf.Dockerfile -t sinca-train:tf-cpu-v1 .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AWS_DEFAULT_REGION=us-east-1

# scikit-learn (usado por evaluate) y sagemaker-training (sdist) necesitan toolchain
# mínimo; libgomp lo requiere scikit-learn en tiempo de ejecución.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "tensorflow-cpu==2.20.0" \
    "numpy>=1.26,<3.0" \
    "pandas>=2.0,<3.0" \
    "scikit-learn>=1.4,<2.0" \
    "mlflow>=3.0,<4.0" \
    "boto3>=1.34,<2.0" \
    "pyarrow>=15.0,<22.0" \
    "sagemaker-training>=4.7,<5.0"

ENTRYPOINT ["train"]
