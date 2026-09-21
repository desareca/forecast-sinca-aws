## 0. Pre-work (operador)

- [x] 0.1 Conectar git: `git init` + `git remote add origin https://github.com/desareca/forecast-sinca-aws` (el repo ya existe en GitHub)
- [x] 0.2 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.3 Bootstrap remote state: crear bucket `<account-id>-tfstate` y tabla DynamoDB `terraform-locks` en `us-east-1` (bucket con versioning, tabla con hash key `LockID` tipo String)

## 1. Escritura de Terraform base

- [x] 1.1 Crear `infra/backend.tf` con `terraform { backend "s3" }` (bucket `<account-id>-tfstate`, key `forecast-sinca-aws/infra.tfstate`, DynamoDB `terraform-locks`, región `us-east-1`)
- [x] 1.2 Crear `infra/providers.tf` con provider `aws` (profile `$AWS_PROFILE`, región `us-east-1`) y `required_providers`
- [x] 1.3 Crear `infra/variables.tf` con variables de nombres de bucket (`sinca-data`, `sinca-mlflow`), nombre de dominio/space e instancia (`ml.t3.large`)
- [x] 1.4 Crear `infra/buckets.tf` con los buckets `sinca-data` y `sinca-mlflow` (versioning solo donde la política lo exija; sin bloquear acceso público por defecto)
- [x] 1.5 Crear `infra/sagemaker-domain.tf` con dominio Studio (`app_network_access_type = "PublicInternetOnly"`), user profile y Space `ml.t3.large` con acceso remoto (SSH sobre SSM) habilitado
- [x] 1.6 Crear `infra/lifecycle.tf` con lifecycle config (`~/.on_start`): instalar Node.js + Open Code CLI + AWS CLI + Terraform, clonar `desareca/forecast-sinca-aws`, setear credenciales git
- [x] 1.7 Crear `infra/iam-dev.tf` con rol `sinca-dev-role` (S3 `sinca-data/*` + `sinca-mlflow/_mlflow/*`, `cloudwatch:logs` log group propio, `ssm`; sin permisos IAM)
- [x] 1.8 Crear `infra/iam-training.tf` con rol `sinca-training-role` skeleton (S3 read `sinca-data/validated/*`, write `sinca-mlflow/model-artifacts/*`, `cloudwatch:logs`)
- [x] 1.9 Crear `infra/outputs.tf` con ARNs de buckets, dominio y roles

## 2. Documentación

- [x] 2.1 Reemplazar `README.md` por README específico del proyecto (qué datos entran, qué hace, estructura del repo, estado actual)
- [x] 2.2 Llenar `project-decisions.md` (contexto, blueprints, storage, IAM, cambios cerrados) reflejando lo aprobado en este change

## 3. Ejecución real (operador corre y pega output)

- [x] 3.1 `terraform init` (descarga providers y activa backend remoto)
- [x] 3.2 `terraform plan` y revisar el plan con el operador antes de aplicar
- [x] 3.3 `terraform apply` (crear buckets, dominio Studio, Space, roles)

## 4. Verificación funcional

- [x] 4.1 Listar el Space creado (confirmar que existe y quedó en estado activo/apagable) y el dominio Studio
- [x] 4.2 Write/read de un objeto dummy en `sinca-data/raw/` con el profile `$AWS_PROFILE` (confirmar permisos y bucket operativo)
- [x] 4.3 Confirmar `sinca-mlflow/_mlflow/` accesible con el rol de desarrollo (o negación esperada para el rol de training)

## Notas de implementación

- **Tarea 1.5 — VPC obligatoria**: `terraform validate` demostró que `aws_sagemaker_domain` **exige `vpc_id` + `subnet_ids`** (no es opcional, a diferencia de lo asumido en `design.md` decisión 1 "sin VPC"). Se resolvió reutilizando el **default VPC** ya existente en la cuenta (`vpc-039ab17427fdd7242`, 6 subnets) manteniendo `app_network_access_type = "PublicInternetOnly"`. Se preserva la intención de costo (sin NAT gateway, sin recursos de red nuevos), pero el agente `plan` debería revisar/actualizar `design.md` decisión 1 para reflejar que un VPC es requerido por SageMaker Studio.
- **Tarea 1.3 / 1.7 — acceso remoto SSH sobre SSM**: el remote access (SSM) de Studio no se expresa como recurso Terraform directo; se configura sobre el Space al momento de conectarse (VS Code + AWS Toolkit). El rol `sinca-dev-role` no incluye permisos `ssm` explícitos porque el SSH-over-SSM de Studio lo gestiona el backend de SageMaker; queda pendiente confirmar en `terraform apply`/verificación (4.1) si se requiere un permiso adicional.
- **Backend deprecation**: Terraform 1.15 marca `dynamodb_table` como deprecado a favor de `use_lockfile`. Se mantuvo `dynamodb_table` por coherencia con la tabla `terraform-locks` ya creada y el patrón de `modules/cicd/blueprint.md`; el plan evalúa migrarlo a `use_lockfile` en un change posterior.
- **Tarea 1.5 / 3.3 — `space_settings.app_type` obligatorio**: el primer `terraform apply` creó todo menos `aws_sagemaker_space` con el error `AppType [null] is not supported for private spaces. Use [CodeEditor, JupyterLab]`. El `space_settings` de un Space privado exige declarar `app_type` explícito (no se infiere de `jupyter_lab_app_settings`). Se agregó `app_type = "JupyterLab"`. Requiere un segundo `terraform apply` para completar.
- **Tarea 4.3 — alcance real de la verificación IAM**: se confirmó que los buckets y el prefijo `_mlflow/` son escribibles/legibles desde el profile `$AWS_PROFILE` (admin). La verificación fina del límite entre `sinca-dev-role` y `sinca-training-role` (prefijos acotados) no se probó porque esos roles solo son asumibles por `sagemaker.amazonaws.com` (no por un usuario humano vía STS). Queda diferida a cuando un app de Studio / Training Job asuma cada rol (Fases 2+).
- **Artefactos de prueba en S3**: quedaron `sinca-data/raw/verificacion.txt` y `sinca-mlflow/_mlflow/probe.txt` como verificación. Inofensivos; se pueden limpiar (no los borré porque `aws s3 rm` está en deny list).
