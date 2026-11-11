# tracking-experimentos Specification

## Purpose
TBD - created by archiving change setup-mlflow-ci-cd. Update Purpose after archive.
## Requirements
### Requirement: Servidor MLflow self-hosted on-demand

El sistema SHALL disponer de un servidor MLflow self-hosted que se levanta como task Fargate on-demand (no un service 24/7), con backend SQLite persistido en S3, para inspeccionar experimentos de forma interactiva.

#### Scenario: Levantar y apagar el servidor
- **WHEN** se ejecuta el script de operación con el subcomando de arranque
- **THEN** una task Fargate arranca el servidor MLflow y expone su UI, y al detenerla no queda ningún cómputo corriendo

#### Scenario: Persistencia del backend entre sesiones
- **WHEN** el servidor se apaga tras una sesión de inspección
- **THEN** el archivo `mlflow.sqlite` se sube de vuelta a `s3://sinca-mlflow/_mlflow/` y la historia de experimentos se conserva para la próxima sesión

### Requirement: Imagen MLflow custom versionada en ECR

El sistema SHALL disponer de una imagen MLflow custom, construida a partir de la imagen oficial, que incluya un entrypoint que descarga el backend SQLite de S3 al arrancar y lo sube de vuelta al apagar, y SHALL estar versionada en un repositorio ECR del proyecto.

#### Scenario: Ciclo download/upload del backend
- **WHEN** el contenedor arranca
- **THEN** descarga `mlflow.sqlite` de `s3://sinca-mlflow/_mlflow/` (si existe) antes de levantar el servidor, y al recibir SIGTERM lo sube de vuelta antes de terminar

#### Scenario: Imagen reproducible desde ECR
- **WHEN** se referencia la imagen desde la task definition
- **THEN** la imagen existe en el repositorio ECR del proyecto y es descargable por Fargate

### Requirement: Artefactos de tracking en S3

El sistema SHALL almacenar los artefactos de tracking (modelos, plots, métricas) directamente en S3 bajo `s3://sinca-mlflow/_mlflow/mlruns`, sin requerir sincronización de vuelta al apagar el servidor.

#### Scenario: Artefactos directo a S3
- **WHEN** una corrida o el servidor escribe un artefacto de tracking
- **THEN** el artefacto persiste en `s3://sinca-mlflow/_mlflow/mlruns` sin depender del ciclo download/upload del backend SQLite

### Requirement: Logging directo sin servidor

El sistema SHALL soportar el modo de logging directo (sin servidor): el cómputo de entrenamiento importa `mlflow`, configura el `tracking_uri` a un archivo SQLite local, loguea métricas/parámetros/artefactos, y al terminar sube `mlflow.sqlite` y `mlruns/` a `sinca-mlflow/_mlflow/`, sin levantar un servidor.

#### Scenario: Logging directo desde un Training Job
- **WHEN** un Training Job entrena un modelo y loguea a MLflow
- **THEN** las métricas y artefactos persisten en `sinca-mlflow/_mlflow/` sin requerir un servidor MLflow corriendo

#### Scenario: Una sola fuente de escritura
- **WHEN** se ejecuta una corrida de entrenamiento
- **THEN** la escritura final a `sinca-mlflow/_mlflow/` es serializada (una corrida a la vez), para no corromper el backend SQLite

### Requirement: Convención de runs para modelos conjuntos de reconstrucción

El sistema SHALL registrar los runs de autoencoders como runs de **modelo conjunto** que cubren las 6 series, con nombre `sinca-zona-ae_{arq}-v{n}`, tags `modelo=ae-{arq}`, `tipo=entrenamiento` y `stage=ae-reconstruccion`, y sin parámetros `estacion`/`pollutant`. Los runs de referencia SHALL llevar `tipo=referencia` y los del pase de DMD SHALL registrarse por serie con `modelo=ae-{arq}-dmd`, parámetros `estacion`/`pollutant`/`version` y un tag `ae_run_id` que apunta al run del AE.

#### Scenario: Run de AE sin serie
- **WHEN** se entrena un autoencoder conjunto
- **THEN** el run se registra con nombre `sinca-zona-ae_{arq}-v{n}` y los tags definidos, sin parámetros `estacion`/`pollutant`

#### Scenario: Runs del pase de DMD por serie
- **WHEN** se evalúa el pase de DMD de un AE
- **THEN** se registra un run por serie con el mismo contrato de artefactos que LightGBM y el tag `ae_run_id` del AE de origen

#### Scenario: Una sola fuente de escritura
- **WHEN** se ejecutan varias corridas de AE
- **THEN** se ejecutan de una en una y cada escritura final a `sinca-mlflow/_mlflow/` se serializa
