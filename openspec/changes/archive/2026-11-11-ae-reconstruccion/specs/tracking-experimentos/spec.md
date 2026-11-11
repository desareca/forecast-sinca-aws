## ADDED Requirements

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
