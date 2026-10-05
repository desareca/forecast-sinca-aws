# Imagen del scraper del pipeline de datos (Fase 2).
# Build desde la raíz del repo:
#   docker build -f docker/scraper.Dockerfile -t sinca-scraper .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencias del pipeline (runtime pesado: pandas + pandera + pyarrow).
RUN pip install --no-cache-dir \
    "pandas>=2.0,<3.0" \
    "pandera>=0.19,<0.24" \
    "numpy>=1.26,<3.0" \
    "requests>=2.31,<3.0" \
    "beautifulsoup4>=4.12,<5.0" \
    "boto3>=1.34,<2.0" \
    "pyarrow>=15.0,<22.0"

# Código del pipeline (entrypoint.py + subpaquetes scraper/icap/validate/persist).
COPY pipeline/ /app/

ENTRYPOINT ["python", "entrypoint.py"]
