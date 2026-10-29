# analisis-modelos Specification

## Purpose

Análisis y visualización on-demand de los resultados de entrenamiento del baseline: dashboard interactivo que lee el estado de MLflow desde S3 (sin servidor) y expone series real vs predicho, `ICAP_zona`, feature importance y comparación contra el baseline de persistencia.

## Requirements

### Requirement: Dashboard de análisis de modelos

El sistema SHALL disponer de un dashboard Streamlit interactivo que, por run de entrenamiento, muestre: resumen de RMSE por serie (LightGBM vs persistencia), series real vs predicho con bandas ICAP (alerta 100 / preemergencia 200 / emergencia 300), `ICAP_zona` real vs predicho, y feature importance por serie.

#### Scenario: Visualizar resultados de un run
- **WHEN** el operador selecciona un run y una serie en el dashboard
- **THEN** se muestran las figuras correspondientes generadas desde los diagnósticos persistidos del run

### Requirement: Lectura del estado de MLflow sin servidor

El sistema SHALL leer el estado de MLflow (`mlflow.sqlite` y `mlruns/`) desde `sinca-mlflow/_mlflow/`, incluyendo los artefactos de diagnóstico (`predictions.parquet`, `feature_importance.parquet`, `zone_icap.parquet`), sin requerir un servidor MLflow corriendo.

#### Scenario: Carga y listado de runs
- **WHEN** se inicia el dashboard o se refrescan los datos
- **THEN** se descargan el `mlflow.sqlite` y los artefactos necesarios desde S3 y se listan los runs disponibles del experimento

### Requirement: Degradación ante runs sin diagnósticos

El sistema SHALL mostrar las métricas disponibles y un aviso cuando un run no tenga diagnósticos persistidos (p. ej. `predictions.parquet` ausente), sin fallar.

#### Scenario: Run sin predicciones persistidas
- **WHEN** un run no tiene `predictions.parquet`
- **THEN** el dashboard muestra las métricas del run y un aviso de que no hay diagnósticos persistidos para ese run

### Requirement: Ejecución on-demand sin infraestructura nueva

El sistema SHALL poder ejecutarse on-demand (local en contenedor Docker o en el SageMaker Studio Space vía port-forward), sin un servicio 24/7 ni recursos AWS nuevos.

#### Scenario: Levantar y apagar
- **WHEN** el operador ejecuta el dashboard
- **THEN** el dashboard queda disponible y, al terminar la ejecución, no queda ningún cómputo corriendo
