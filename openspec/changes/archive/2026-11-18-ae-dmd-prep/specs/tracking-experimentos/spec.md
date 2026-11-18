## ADDED Requirements

### Requirement: Sincronización parcial del estado en los jobs

El sistema SHALL permitir que un job descargue, antes de loguear, todo el estado de MLflow, solo `mlflow.sqlite`, o `mlflow.sqlite` más los artefactos de una lista de runs. Por defecto SHALL descargar todo. Un job que use un artefacto de otro run SHALL descargar al menos los artefactos de ese run. La subida final SHALL publicar los archivos locales sin borrar objetos de S3.

#### Scenario: Job que solo agrega runs
- **WHEN** un job se lanza con sincronización `none`
- **THEN** descarga solo `mlflow.sqlite`, registra sus runs y al terminar los artefactos previos siguen en S3

#### Scenario: Pase de DMD standalone
- **WHEN** se lanza el modo `dmd` con un `ae_run_id`
- **THEN** se descargan los artefactos de ese run aunque la sincronización sea `none`

### Requirement: Mantenimiento del estado de MLflow

El sistema SHALL ofrecer una herramienta de mantenimiento con un modo de plan de solo lectura, que liste los runs `lightgbm` con `version = 99` y los nombres duplicados de runs de AE y de referencia con su nombre nuevo, y un modo de aplicación que trabaje sobre una copia local del sqlite. La herramienta NO SHALL escribir en S3. La aplicación en S3 SHALL quedar en manos del operador, con respaldo previo del sqlite.

#### Scenario: Plan sin cambios pendientes
- **WHEN** el plan no encuentra runs v99 ni duplicados
- **THEN** informa que no hay acciones y termina sin modificar ningún archivo

#### Scenario: Nombres únicos tras el plan
- **WHEN** se aplica un plan de renombre a la copia local
- **THEN** no quedan dos runs activos de AE o de referencia con el mismo nombre

## MODIFIED Requirements

### Requirement: Convención de runs para modelos conjuntos de reconstrucción

El sistema SHALL registrar los runs de autoencoders como runs de **modelo conjunto** que cubren las 6 series, sin parámetros `estacion`/`pollutant`, con tags `modelo=ae-{arq}`, `tipo=entrenamiento` y `seed`, y con nombre que incluye la semilla: `sinca-zona-ae_{arq}-v{n}-s{seed}` en la partición `holdout` (tag `stage=ae-reconstruccion`) y `sinca-zona-ae_{arq}-{part}-v{n}-s{seed}` en una partición con nombre, donde `{part}` es el nombre de la partición sin guiones (tags `stage=ae-dmd-prep` y `particion`). Los runs de referencia SHALL llevar `tipo=referencia` y nombre `ref-mean-L{L}-v{n}` o `ref-pca-d{d}-L{L}-v{n}`. Los del pase de DMD SHALL registrarse por serie con `modelo=ae-{arq}-dmd`, parámetros `estacion`/`pollutant`/`version`/`seed`, un tag `ae_run_id` que apunta al run del AE y nombre `sinca-{estacion}-ae-{arq}-dmd-v{n}-s{seed}` en `holdout` o `sinca-{estacion}-{pollutant}-ae-{arq}-dmd-{part}-v{n}-s{seed}` en una partición con nombre. Las referencias de pronóstico de una partición con nombre SHALL usar un `modelo` propio (`lightgbm-{part}`, `persistence-{part}`) para no agruparse con el baseline, y los runs LightGBM por serie SHALL llamarse `sinca-{estacion}-{pollutant}-lightgbm-{part}-v{n}`.

#### Scenario: Run de AE sin serie
- **WHEN** se entrena un autoencoder conjunto
- **THEN** el run se registra con el nombre que incluye la semilla y los tags definidos, sin parámetros `estacion`/`pollutant`

#### Scenario: Semillas distinguibles por nombre
- **WHEN** se entrenan 3 semillas de la misma configuración
- **THEN** los 3 runs tienen nombres distintos

#### Scenario: Series distinguibles por nombre
- **WHEN** se registran los runs por serie de una partición con nombre
- **THEN** los runs de `pm10` y `pm25` de una misma estación tienen nombres distintos

#### Scenario: Runs del pase de DMD por serie
- **WHEN** se evalúa el pase de DMD de un AE
- **THEN** se registra un run por serie con el mismo contrato de artefactos que LightGBM y el tag `ae_run_id` del AE de origen

#### Scenario: Una sola fuente de escritura
- **WHEN** se ejecutan varias corridas de AE
- **THEN** se ejecutan de una en una y cada escritura final a `sinca-mlflow/_mlflow/` se serializa
