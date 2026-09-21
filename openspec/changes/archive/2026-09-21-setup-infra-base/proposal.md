## Why

El proyecto `forecast-sinca-aws` (predicción ICAP zona saturada Temuco/PLC, horizonte 24h) no tiene todavía ninguna infraestructura de base: no hay entorno de desarrollo remoto, ni buckets de datos, ni roles IAM definidos como código. Este change establece la base Terraform mínima (Fase 1 del plan, "Setup") sobre la que se montarán el pipeline de datos, los experimentos y la productivización. Es el equivalente infra de "crear el repo y el primer ambiente" antes de escribir lógica de negocio.

## What Changes

- Provisiona el **remote state de Terraform** (bucket S3 + tabla DynamoDB de locking) como backend, en la cuenta personal (`AWS_PROFILE`).
- Crea los **buckets S3** `sinca-data` (datos: `raw/`/`validated/`/`quarantine/`) y `sinca-mlflow` (tracking MLflow, prefijo `_mlflow/`).
- Provisiona el **entorno de desarrollo remoto**: SageMaker Studio domain + Space (`ml.t3.large`), home sobre EFS, lifecycle config (`~/.on_start` reinstala Node + Open Code CLI + clona el repo), acceso remoto SSH sobre SSM.
- Define los **roles IAM de base**: rol del entorno de desarrollo y rol de training (skeleton), ambos con mínimo privilegio y ARNs acotados por prefijo.
- Actualiza `README.md` para reflejar el estado del proyecto (deja de ser el README de la plantilla).

No incluye (queda para `setup-mlflow-ci-cd`): servidor MLflow, OIDC + workflows de GitHub Actions, ni lógica de negocio.

## Capabilities

### New Capabilities

- `dev-environment`: entorno de desarrollo remoto persistente (SageMaker Studio Space sobre EFS), reproducible vía Terraform, con acceso SSH sobre SSM para VS Code local.
- `storage`: buckets S3 del proyecto (`sinca-data`, `sinca-mlflow`) y la convención de prefijos (`raw`/`validated`/`quarantine`, `_mlflow/`).
- `iam`: roles IAM de base (dev environment y training) con política de mínimo privilegio acotada a los prefijos S3 exactos.

### Modified Capabilities

<!-- Ninguno: no hay specs previas. -->

## Impact

- **Infraestructura AWS**: recursos nuevos en la cuenta personal (`AWS_ACCOUNT_ID`) (buckets S3, dominio SageMaker Studio + Space + EFS, roles IAM). Región `us-east-1`.
- **Repo**: nueva carpeta `infra/` con el código Terraform; `README.md` actualizado.
- **Dependencia previa**: requiere el bootstrap del remote state (bucket `<account-id>-tfstate` + DynamoDB `terraform-locks`) creado manualmente antes del primer `terraform apply`.
- **Costo mensual estimado**: Space `ml.t3.large` ~$0.052/h (solo cuando está prendido); EFS ~$0.30/GB-mes; S3 despreciable a este volumen. Total típico por debajo de USD 5/mes si el Space se apaga cuando no se usa.
