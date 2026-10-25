# Imagen del dashboard de análisis (Fase 3.5).
#
# Lleva SOLO el runtime (patrón `compute`/`tracking`): el código se monta como
# volumen para hot-reload, sin rebuild. Build desde la raíz del repo:
#
#   docker build -f docker/analysis.Dockerfile -t sinca-analysis .
#
# Correr contra el fixture (sin AWS):
#
#   docker run --rm -p 8501:8501 \
#     -v "$PWD:/app" sinca-analysis \
#     pipeline/analysis/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
#
# Correr contra S3 real: montar `~/.aws` y pasar AWS_PROFILE/AWS_DEFAULT_REGION.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-analysis.txt /app/requirements-analysis.txt
RUN pip install --no-cache-dir -r /app/requirements-analysis.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
