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

### Requirement: Visualización de reconstrucción y linealizabilidad

El sistema SHALL mostrar en el dashboard, para los runs de autoencoder, las métricas y los diagnósticos de reconstrucción (real vs reconstruido por serie, razón de varianza, pendiente, espectro, error por nivel de concentración, estado de la compuerta) y, cuando existan, las métricas de linealizabilidad `dmd_latent_rmse_1` y `dmd_latent_rmse_24`, leídos desde los artefactos persistidos del run.

#### Scenario: Run de AE seleccionado
- **WHEN** el operador selecciona un run de autoencoder
- **THEN** el dashboard muestra los diagnósticos de reconstrucción y el estado de la compuerta generados desde `reconstruction.parquet` y las métricas del run

#### Scenario: Comparación del pase de DMD
- **WHEN** el operador selecciona los runs `ae-{arq}-dmd` de una serie
- **THEN** el dashboard los compara contra persistencia y LightGBM con las mismas vistas existentes

### Requirement: Notebooks de exploración sobre artefactos persistidos

El sistema SHALL disponer de plantillas de notebook en `notebooks/` para explorar los datos de entrada del autoencoder, la reconstrucción y el latente/DMD, que lean los artefactos persistidos de los runs (`reconstruction.parquet`, `latent.parquet`, `dmd_operator.npz`) y los datos validados, ejecutables en JupyterLab del Space con el rol del Space y sin infraestructura nueva. El sistema SHALL permitir explorar sin TensorFlow los diagnósticos y el análisis del DMD sobre el latente persistido, y SHALL permitir, con TensorFlow instalado en el Space en la misma versión que la imagen de training, cargar `encoder.keras` y `decoder_mp.keras` para explorar lo que pasa por el decoder.

#### Scenario: Exploración en el Space sin TensorFlow
- **WHEN** el operador abre un notebook de exploración en JupyterLab del Space y ejecuta las celdas de nivel 1
- **THEN** el notebook carga los artefactos persistidos desde `sinca-mlflow/_mlflow/` y `sinca-data/validated/` y genera sus figuras sin credenciales explícitas

#### Scenario: Exploración que pasa por el decoder
- **WHEN** el operador ejecuta las celdas de nivel 2 (reajustar el DMD y decodificar a µg/m³, o recorrer una dimensión del latente)
- **THEN** el notebook carga `encoder.keras` y `decoder_mp.keras` del run, decodifica y muestra el efecto sobre las 6 series

#### Scenario: Notebooks versionados con outputs
- **WHEN** se versiona un notebook de exploración tras ejecutarlo con datos reales
- **THEN** se guarda con los outputs de esa corrida y sin credenciales, IDs de cuenta ni ARNs

### Requirement: Tolerancia a runs sin métricas de pronóstico

El sistema SHALL listar y mostrar los runs que no tienen parámetros `estacion`/`pollutant` ni `feature_importance.parquet` ni métricas de pronóstico, sin fallar y sin mezclarlos con las vistas por serie del baseline.

#### Scenario: Run de AE sin pronóstico
- **WHEN** el dashboard carga un run de AE que no tiene métricas de pronóstico
- **THEN** muestra sus métricas de reconstrucción y omite las vistas de pronóstico sin error

#### Scenario: Run sin feature importance
- **WHEN** un run no tiene `feature_importance.parquet`
- **THEN** el dashboard omite ese panel y muestra un aviso
