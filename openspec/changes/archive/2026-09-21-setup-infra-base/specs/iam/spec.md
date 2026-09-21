## ADDED Requirements

### Requirement: Rol del entorno de desarrollo

El sistema SHALL definir un rol IAM para el entorno de desarrollo (`sinca-dev-role`) con permisos de mínimo privilegio acotados a: lectura/escritura sobre los prefijos S3 exactos del proyecto, logs en su log group propio, y SSM para acceso remoto.

#### Scenario: Permiso acotado a prefijos S3
- **WHEN** el rol de desarrollo accede a S3
- **THEN** solo puede leer/escribir en `sinca-data/*` y `sinca-mlflow/_mlflow/*`, no en otros buckets ni prefijos

#### Scenario: Sin permisos IAM ni políticas amplias
- **WHEN** se inspecciona la política del rol de desarrollo
- **THEN** no contiene permisos IAM ni managed policies `*FullAccess` / `AdministratorAccess`

### Requirement: Rol de training (skeleton)

El sistema SHALL definir un rol IAM para entrenamiento (`sinca-training-role`) con acceso de solo lectura a los datos validados y escritura de artefactos a un prefijo acotado de tracking, sin más permisos.

#### Scenario: Lectura de datos validados
- **WHEN** el rol de training lee datos de entrada
- **THEN** puede leer desde `sinca-data/validated/*` únicamente

#### Scenario: Escritura de artefactos acotada
- **WHEN** el rol de training escribe artefactos de modelo
- **THEN** solo puede escribir en `sinca-mlflow/model-artifacts/*`

### Requirement: Verificación de identidad de cuenta

Todo comando que toque AWS SHALL ejecutarse contra el profile `AWS_PROFILE`, verificando que la identidad corresponda a la cuenta personal `AWS_ACCOUNT_ID` y nunca a un profile de otra cuenta.

#### Scenario: Confirmación de cuenta
- **WHEN** se ejecuta `aws sts get-caller-identity --profile $AWS_PROFILE`
- **THEN** el `Account` devuelto es `AWS_ACCOUNT_ID`
