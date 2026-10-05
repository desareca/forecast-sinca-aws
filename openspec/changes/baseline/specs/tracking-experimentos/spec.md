## ADDED Requirements

### Requirement: Logging directo sin servidor

El sistema SHALL soportar el modo de logging directo (sin servidor): el cómputo de entrenamiento importa `mlflow`, configura el `tracking_uri` a un archivo SQLite local, loguea métricas/parámetros/artefactos, y al terminar sube `mlflow.sqlite` y `mlruns/` a `sinca-mlflow/_mlflow/`, sin levantar un servidor.

#### Scenario: Logging directo desde un Training Job
- **WHEN** un Training Job entrena un modelo y loguea a MLflow
- **THEN** las métricas y artefactos persisten en `sinca-mlflow/_mlflow/` sin requerir un servidor MLflow corriendo

#### Scenario: Una sola fuente de escritura
- **WHEN** se ejecuta una corrida de entrenamiento
- **THEN** la escritura final a `sinca-mlflow/_mlflow/` es serializada (una corrida a la vez), para no corromper el backend SQLite
