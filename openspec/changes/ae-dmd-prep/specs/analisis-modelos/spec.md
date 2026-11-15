## ADDED Requirements

### Requirement: Resumen de la validación de selección

El sistema SHALL producir, para una partición con nombre, un resumen de la validación de selección que compare persistencia, LightGBM y cada pase de DMD sobre las filas comunes a todos ellos por serie, con RMSE y RMSE ponderado por serie, su media sobre las 6 series y el número de filas comunes, y una tabla de los runs de AE con su compuerta, su reconstrucción en `sel`, su linealizabilidad y el resultado del contraste con `pydmd`.

#### Scenario: Filas comunes
- **WHEN** un modelo no tiene predicción para un `timestamp` de una serie
- **THEN** ese `timestamp` se excluye de esa serie para todos los modelos del resumen

#### Scenario: Resumen sin runs de la partición
- **WHEN** no hay runs con la partición pedida
- **THEN** el resumen informa que no hay runs y termina sin error

## MODIFIED Requirements

### Requirement: Lectura del estado de MLflow sin servidor

El sistema SHALL leer el estado de MLflow (`mlflow.sqlite` y `mlruns/`) desde `sinca-mlflow/_mlflow/`, incluyendo los artefactos de diagnóstico (`predictions.parquet`, `feature_importance.parquet`, `zone_icap.parquet`), sin requerir un servidor MLflow corriendo. El sistema SHALL permitir descargar solo `mlflow.sqlite` más los artefactos de los runs pedidos. El listado de runs NO SHALL incluir por defecto los runs borrados (`lifecycle_stage = deleted`).

#### Scenario: Carga y listado de runs
- **WHEN** se inicia el dashboard o se refrescan los datos
- **THEN** se descargan el `mlflow.sqlite` y los artefactos necesarios desde S3 y se listan los runs disponibles del experimento

#### Scenario: Descarga selectiva
- **WHEN** se pide el estado para una lista de runs
- **THEN** se descargan `mlflow.sqlite` y solo los artefactos de esos runs

#### Scenario: Run borrado
- **WHEN** un run fue borrado con `delete_run`
- **THEN** no aparece en el listado de runs
