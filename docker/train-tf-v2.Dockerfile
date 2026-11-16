# Imagen de entrenamiento TensorFlow v2 (change `ae-dmd-prep`, D8): `tf-cpu-v1` + `pydmd` a prueba.
#
# Construye SOBRE la imagen `tf-cpu-v1` ya publicada en ECR (BASE_IMAGE, por tag o digest):
# la única capa nueva es `pydmd`. Así, todo lo demás queda idéntico a v1 por construcción.
# `train-tf.Dockerfile` (tf-cpu-v1) no se modifica ni se reconstruye.
# `pydmd` no entra al entrenamiento TF: el contraste corre sobre latentes como arrays.
#
# Tag explícito `tf-cpu-v2` (ECR `sinca-train`). NO se usa `:latest`.
#
# Aceptación (docker/README.md): el freeze de v2 debe ser el de v1 más líneas agregadas,
# sin cambiar ninguna versión de v1 (incluido mlflow); `pip check` igual que v1. Si pydmd
# fuerza cambiar algún paquete de v1: plan B (tf-cpu-v1 + venv de requirements-dmd.txt).
#
# Build desde la raíz del repo (BASE_IMAGE es obligatorio):
#   docker build -f docker/train-tf-v2.Dockerfile --build-arg BASE_IMAGE=<reg>/sinca-train:tf-cpu-v1 -t sinca-train:tf-cpu-v2 .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Capa v2: contraste con pydmd (D8). Versión fijada.
RUN pip install --no-cache-dir "pydmd==2025.8.1"

ENTRYPOINT ["train"]
