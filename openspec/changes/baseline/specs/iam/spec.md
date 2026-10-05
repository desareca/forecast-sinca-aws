## MODIFIED Requirements

### Requirement: Rol de training (skeleton)

El sistema SHALL definir un rol IAM para entrenamiento (`sinca-training-role`) con acceso de solo lectura a los datos validados, escritura de artefactos de modelo y de tracking, sin más permisos.

#### Scenario: Lectura de datos validados
- **WHEN** el rol de training lee datos de entrada
- **THEN** puede leer desde `sinca-data/validated/*` únicamente

#### Scenario: Escritura de artefactos de modelo
- **WHEN** el rol de training escribe artefactos de modelo
- **THEN** solo puede escribir en `sinca-mlflow/model-artifacts/*`

#### Scenario: Escritura de tracking
- **WHEN** el rol de training loguea métricas y artefactos de tracking
- **THEN** puede leer/escribir en `sinca-mlflow/_mlflow/*`
