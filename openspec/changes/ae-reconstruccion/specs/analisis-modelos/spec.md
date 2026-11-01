## ADDED Requirements

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

#### Scenario: Plantillas sin salidas
- **WHEN** se versiona una plantilla de notebook
- **THEN** se guarda sin outputs ni contadores de ejecución

### Requirement: Tolerancia a runs sin métricas de pronóstico

El sistema SHALL listar y mostrar los runs que no tienen parámetros `estacion`/`pollutant` ni `feature_importance.parquet` ni métricas de pronóstico, sin fallar y sin mezclarlos con las vistas por serie del baseline.

#### Scenario: Run de AE sin pronóstico
- **WHEN** el dashboard carga un run de AE que no tiene métricas de pronóstico
- **THEN** muestra sus métricas de reconstrucción y omite las vistas de pronóstico sin error

#### Scenario: Run sin feature importance
- **WHEN** un run no tiene `feature_importance.parquet`
- **THEN** el dashboard omite ese panel y muestra un aviso
