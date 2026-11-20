## ADDED Requirements

### Requirement: Convención de runs del operador global

Los runs del operador SHALL ser conjuntos (6 series, sin params `estacion`/`pollutant`), con nombre `sinca-zona-dmdop-{rep}-{exp}-{part}-v{n}-s{seed}` (`{seed}` = semilla del AE, o la del bootstrap si la representación es determinista), tags `modelo=dmdop-{rep}`, `stage=dmd-operador-global`, `particion`, `etapa`, `seed` y, si aplica, `ae_run_id`, y con sus artefactos en el directorio de artefactos del estado local. Los de test del ganador SHALL ser por serie con el contrato de LightGBM.

#### Scenario: Semillas y etapas distinguibles
- **WHEN** se evalúa la misma representación con 3 semillas en dos etapas
- **THEN** los 6 runs tienen nombres distintos y cada uno lleva su `etapa` y su `seed`

#### Scenario: Oráculo marcado
- **WHEN** se registra un run del oráculo
- **THEN** lleva el param `tipo=diagnostico` y no aparece en las vistas por serie del dashboard

#### Scenario: Artefactos resolubles
- **WHEN** se registra un run sobre un estado descargado cuyo experimento tiene `artifact_location` de otra máquina
- **THEN** sus artefactos quedan bajo `mlruns/<run_id>/artifacts/` del estado local y se publican con el run
