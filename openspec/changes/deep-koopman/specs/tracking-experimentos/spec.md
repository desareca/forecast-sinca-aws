## ADDED Requirements

### Requirement: Convención de runs del Koopman entrenado

Los runs del Koopman SHALL ser conjuntos (6 series, sin params `estacion`/`pollutant`), con nombre `sinca-zona-koop-{enc}-{fase}{var}-{part}-v{n}-s{seed}`, tags `modelo=koop-{enc}`, `stage=deep-koopman`, `particion`, `fase`, `variante`, `seed` y `parent_run_id` (run del que arranca). Los de test del modelo final SHALL ser por serie con el contrato de LightGBM.

#### Scenario: Fases y variantes distinguibles
- **WHEN** se entrena el mismo encoder y semilla en F1 principal, F1 con K denso y F2
- **THEN** los 3 runs tienen nombres distintos y su `parent_run_id` encadena F2 con F1

### Requirement: Publicación desde Training Jobs

Un job que escribe en `sinca-mlflow/_mlflow/` SHALL registrar el `LastModified` del sqlite al descargarlo y, antes de subir, SHALL abortar la subida si cambió; si no cambió, SHALL respaldar el sqlite remoto en `_mlflow/backups/` y subir solo el sqlite y los artefactos de sus runs nuevos. Si aborta, el run SHALL quedar en la salida del job para publicarlo después.

#### Scenario: Escritura concurrente
- **WHEN** otro proceso modificó el sqlite remoto durante el job
- **THEN** el job no sube nada a `_mlflow/`, termina con error y su run queda en la salida del job
