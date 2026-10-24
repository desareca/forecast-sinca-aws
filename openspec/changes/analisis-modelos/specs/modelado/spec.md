## ADDED Requirements

### Requirement: Persistencia de diagnósticos por run

El sistema SHALL persistir, como artefactos de tracking MLflow de cada run de entrenamiento LightGBM, los diagnósticos del modelo: las predicciones (`predictions.parquet` con `timestamp`, `estacion`, `pollutant`, `y_true`, `y_pred`, `split`), la importancia de features (`feature_importance.parquet`), y el `ICAP_zona` real y predicho del test held-out (`zone_icap.parquet`).

#### Scenario: Diagnósticos persistidos
- **WHEN** un Training Job entrena y evalúa una serie
- **THEN** el run loguea, además de métricas y parámetros, los artefactos `predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet` en `sinca-mlflow/_mlflow/mlruns/`
