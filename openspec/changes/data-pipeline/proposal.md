## Why

Fase 2 del plan. La infraestructura base (buckets S3, SageMaker Studio, MLflow self-hosted, CI/CD) ya está operativa. Falta el **pipeline de datos** que ingiere las 3 estaciones SINCA + Open-Meteo + feriados, calcula el ICAP, valida con pandera y persiste en Parquet. Sin esto no hay datos para entrenar (Fase 3+), y el proyecto no puede avanzar hacia el modelado.

## What Changes

- **Pipeline de datos en `pipeline/`**: scraper SINCA parametrizado por estación (Padre Las Casas II ID 263, Ñielol, Las Encinas), Open-Meteo (altura de capa límite), feriados, cálculo de ICAP (fórmula oficial D.S. 12/2011), validación con pandera, y persistencia en la convención `raw/` → `validated/` → `quarantine/` (Parquet).
- **Task Fargate on-demand** para el scraper (imagen ECR + task definition), reutilizando el patrón ECR+Fargate ya montado.
- **Rol IAM del scraper** (S3 write a `sinca-data/*`, mínimo privilegio).
- **Schedule EventBridge diario** (~01:00 hora local) que dispara la task Fargate.
- **Backfill inicial de 5 años** (2021-01-01 → hoy) + modo incremental diario.

No incluye (queda para Fases 3+): entrenamiento de modelos, SageMaker Pipeline, ni inferencia. El `pipeline-cd.yml` sigue como skeleton (es para la SageMaker Pipeline de entrenamiento, no para la ingesta).

## Capabilities

### New Capabilities

- `data-pipeline`: ingesta de datos (scraper SINCA/Open-Meteo/feriados), cálculo de ICAP, validación de calidad con pandera, persistencia en S3, y schedule de ejecución diaria.

### Modified Capabilities

- `iam`: se agrega el rol de ejecución de la task Fargate del scraper, con acceso S3 acotado a `sinca-data/*` y mínimo privilegio.

## Impact

- **Infraestructura AWS**: recursos nuevos en la cuenta personal (`AWS_ACCOUNT_ID`): repo ECR para la imagen del scraper, task definition Fargate, rol IAM del scraper, y regla de EventBridge Schedule. Región `us-east-1`. Reutiliza el default VPC.
- **Repo**: nueva carpeta `pipeline/` (código del pipeline), `docker/` ampliado (imagen del scraper), `infra/` ampliado (Fargate + IAM + schedule).
- **Dependencia previa**: buckets `sinca-data`/`sinca-mlflow`, remote state, y el patrón ECR+Fargate ya operativos (Fase 1).
- **Costo mensual estimado**: Fargate on-demand (una corrida diaria corta, ~$0.01-0.05/día), EventBridge (despreciable), S3 (despreciable a este volumen). Total por debajo de USD 5/mes; nada queda corriendo 24/7.
