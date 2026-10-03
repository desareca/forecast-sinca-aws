# iam Specification

## Purpose
TBD - created by archiving change setup-infra-base. Update Purpose after archive.
## Requirements
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

### Requirement: Rol OIDC de infraestructura

El sistema SHALL definir un rol IAM `sinca-github-infra-role` asumible por GitHub Actions vía OIDC, con permisos de mínimo privilegio para ejecutar `terraform plan`/`apply` sobre los recursos que Terraform gestiona, y trust policy restringida al repo `desareca/forecast-sinca-aws`.

#### Scenario: Asunción desde el repo correcto
- **WHEN** un workflow del repo `desareca/forecast-sinca-aws` asume `sinca-github-infra-role`
- **THEN** la asunción es permitida y el rol puede operar sobre los recursos gestionados por Terraform

#### Scenario: Sin políticas amplias
- **WHEN** se inspecciona la política de `sinca-github-infra-role`
- **THEN** no contiene `AdministratorAccess` ni managed policies `*FullAccess`

### Requirement: Rol OIDC de pipeline

El sistema SHALL definir un rol IAM `sinca-github-pipeline-role` asumible por GitHub Actions vía OIDC, con permisos para actualizar la definición del pipeline y escribir artefactos, y trust policy restringida a `refs/heads/main` del repo `desareca/forecast-sinca-aws`.

#### Scenario: Asunción solo desde main
- **WHEN** un workflow intenta asumir `sinca-github-pipeline-role` desde una rama distinta a `main`
- **THEN** la asunción es denegada

#### Scenario: Permisos acotados a artefactos y pipeline
- **WHEN** `sinca-github-pipeline-role` opera
- **THEN** solo puede actualizar la definición del pipeline y escribir artefactos en los prefijos acotados, sin permisos de infraestructura de red ni IAM

### Requirement: Rol de ejecución de la task Fargate de MLflow

El sistema SHALL definir un rol IAM para la task Fargate del servidor MLflow, con acceso S3 acotado al prefijo `sinca-mlflow/_mlflow/*` (GetObject/PutObject/ListBucket) y logs en su log group propio, sin más permisos.

#### Scenario: Acceso acotado al prefijo de tracking
- **WHEN** la task Fargate accede a S3
- **THEN** solo puede leer/escribir en `sinca-mlflow/_mlflow/*`, no en otros buckets ni prefijos

#### Scenario: Sin permisos IAM
- **WHEN** se inspecciona la política del rol de la task Fargate
- **THEN** no contiene permisos IAM ni managed policies amplias

