## ADDED Requirements

### Requirement: Bucket de datos con convención de prefijos

El sistema SHALL disponer de un bucket S3 `sinca-data` con la convención de prefijos `raw/`, `validated/` y `quarantine/` para el ciclo de ingesta/validación de datos en formato Parquet.

#### Scenario: Layout de prefijos
- **WHEN** se lista el bucket `sinca-data`
- **THEN** existen (o se pueden crear) los prefijos `raw/`, `validated/` y `quarantine/`

### Requirement: Bucket de tracking MLflow

El sistema SHALL disponer de un bucket S3 `sinca-mlflow` para el tracking de experimentos, con el prefijo dedicado `_mlflow/` para el backend SQLite y los artefactos.

#### Scenario: Prefijo de tracking
- **WHEN** un cómputo con permisos de tracking escribe en `sinca-mlflow/_mlflow/`
- **THEN** persisten `mlflow.sqlite` y `mlruns/` bajo ese prefijo exclusivo

### Requirement: Separación de buckets para mínimo privilegio

Los buckets de datos y de tracking SHALL estar separados para permitir acotar los permisos IAM por bucket, sin solapamiento entre datos y tracking.

#### Scenario: Alcance IAM por bucket
- **WHEN** un rol con acceso solo a `sinca-data` intenta acceder a `sinca-mlflow`
- **THEN** el acceso es denegado (los buckets son distintos y los permisos no se solapan)
