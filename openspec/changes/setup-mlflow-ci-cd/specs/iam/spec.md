## ADDED Requirements

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
