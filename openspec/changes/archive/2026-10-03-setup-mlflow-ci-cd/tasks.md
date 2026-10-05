## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar que ECR y ECS/Fargate están disponibles en `us-east-1` (no requieren habilitación previa, solo verificar)

## 1. Escritura de la imagen MLflow

- [x] 1.1 Crear `docker/Dockerfile` con `FROM ghcr.io/mlflow/mlflow`, `COPY entrypoint.sh`, y `ENTRYPOINT ["/entrypoint.sh"]`
- [x] 1.2 Crear `docker/entrypoint.sh` que: descarga `mlflow.sqlite` de `s3://sinca-mlflow/_mlflow/` (si existe), arranca `mlflow server` con `--backend-store-uri sqlite:///mlflow.sqlite` y `--artifacts-destination s3://sinca-mlflow/_mlflow/mlruns`, y con `trap` de SIGTERM/SIGINT sube el `.sqlite` de vuelta a S3 antes de salir

## 2. Escritura de Terraform

- [x] 2.1 Crear `infra/ecr.tf` con el repositorio ECR de la imagen MLflow (nombre `sinca-mlflow`)
- [x] 2.2 Crear `infra/iam-mlflow-task.tf` con el rol de ejecución de la task Fargate (S3 `sinca-mlflow/_mlflow/*` GetObject/PutObject/ListBucket + `logs` en log group propio)
- [x] 2.3 Crear `infra/fargate-mlflow.tf` con la task definition Fargate (imagen ECR, reutiliza default VPC, sin service 24/7)
- [x] 2.4 Crear `infra/oidc.tf` con `aws_iam_openid_connect_provider` (`token.actions.githubusercontent.com`, audiencia `sts.amazonaws.com`)
- [x] 2.5 Crear `infra/iam-oidc.tf` con `sinca-github-infra-role` (trust `sub` = `repo:desareca/forecast-sinca-aws:*`, permisos plan/apply acotados) y `sinca-github-pipeline-role` (trust `sub` = `repo:desareca/forecast-sinca-aws:ref:refs/heads/main`, permisos de pipeline/artefactos)
- [x] 2.6 Actualizar `infra/outputs.tf` con los ARNs de ECR, roles OIDC y task definition

## 3. Escritura de workflows de GitHub Actions

- [x] 3.1 Crear `.github/workflows/ci.yml` (en PR: `terraform fmt -check` + `terraform validate` sobre `infra/`, sin asumir roles)
- [x] 3.2 Crear `.github/workflows/infra-cd.yml` (en PR: `terraform plan`; al mergear a main: `terraform apply`, gateado por `if: github.ref == 'refs/heads/main'`, asume `sinca-github-infra-role` vía OIDC)
- [x] 3.3 Crear `.github/workflows/pipeline-cd.yml` (skeleton: al mergear cambios en `pipeline/` a main, actualizará la definición del pipeline; asume `sinca-github-pipeline-role`; no-op hasta que exista `pipeline/`)

## 4. Escritura del script de operación

- [x] 4.1 Crear `infra/scripts/mlflow-server.sh` con subcomandos `up` (descarga estado, `aws ecs run-task`) y `down` (`aws ecs stop-task`), usando el profile `$AWS_PROFILE`

## 5. Documentación

- [x] 5.1 Actualizar `README.md` (estado: Fase 1b completada; cómo levantar/apagar el servidor MLflow; workflows disponibles)
- [x] 5.2 Actualizar `project-decisions.md` §9 (aclarar restricción `sub`: rol de infra → cualquier ref con apply gateado por `if:`; rol de pipeline → solo main)

## 6. Ejecución real (operador corre y pega output)

- [x] 6.1 Setear `mlflow_allowed_cidr` a la IP del operador en `infra/terraform.tfvars` (no dejar `0.0.0.0/0`: la UI de MLflow no tiene auth por defecto)
- [x] 6.2 `terraform init` (si hace falta) y `terraform plan` — revisar el plan con el operador antes de aplicar
- [x] 6.3 `terraform apply` (crea ECR, task Fargate, OIDC provider, roles)
- [x] 6.4 Build de la imagen: `docker build -t sinca-mlflow docker/` y push a ECR (login + tag + push) — después del apply, cuando el repo ECR ya existe

## 7. Verificación funcional

- [x] 7.1 Levantar el servidor MLflow con `infra/scripts/mlflow-server.sh up` y confirmar que la UI responde (HTTP 200 en el puerto 5000)
- [x] 7.2 Apagar con `infra/scripts/mlflow-server.sh down` y confirmar que `mlflow.sqlite` quedó en `s3://sinca-mlflow/_mlflow/` (listar el objeto con `aws s3 ls`)
- [x] 7.3 Confirmar que el Identity Provider OIDC y los 2 roles existen (`aws iam list-open-id-connect-providers` y `aws iam get-role` para cada rol)

## Notas de implementación

- **2.2 — dos roles en vez de uno**: Fargate exige un *rol de ejecución*
  (`sinca-mlflow-execution-role`: pull de ECR + logs) separado del *rol de la
  task* que usa el contenedor (`sinca-mlflow-task-role`: S3 `_mlflow/*` + logs).
  La tarea mencionaba un solo "rol de ejecución"; se crearon los dos porque el
  rol de ejecución no se expone al contenedor y el task role no puede hacer pull
  de ECR. Ambos en `infra/iam-mlflow-task.tf`.
- **2.3 — recursos no listados pero necesarios**: se agregaron al `.tf` el
  cluster ECS (`sinca-mlflow`), el log group (`/ecs/sinca-mlflow`) y un security
  group que habilita el puerto 5000 (sin él la UI no es alcanzable y
  `run-task` no tiene red). CPU/memoria Fargate = `512`/`1024` (no decidido en
  `project-decisions.md`; expuestos como variables).
- **1.2 — backend SQLite absoluto**: se usó `sqlite:////mlflow.sqlite` (4
  slashes = ruta absoluta `/mlflow.sqlite`) en vez de `sqlite:///mlflow.sqlite`
  (relativa al cwd), para que el archivo descargado/subido sea siempre el mismo
  independientemente del `WORKDIR` de la imagen.
- **providers.tf — profile condicional**: se cambió `profile = var.aws_profile`
  a `profile = var.aws_profile != "" ? var.aws_profile : null`. Sin esto el
  workflow OIDC no funciona: con `profile` seteado el provider ignora las
  credenciales por variables de entorno que inyecta `configure-aws-credentials`.
  En CI se pasa `-var="aws_profile="`; localmente el `terraform.tfvars` sigue
  exigiendo el named profile.
- **infra-cd.yml — config del backend en CI**: `backend.hcl` está gitignored, así
  que el workflow reconstruye el `-backend-config` con variables del repo
  (`vars.TFSTATE_BUCKET`, `vars.AWS_INFRA_ROLE_ARN`, `vars.AWS_PIPELINE_ROLE_ARN`).
  Requiere configurarlas en el repo (documentado en `README.md`).
- **2.5 — política del rol de infra amplia pero explícita**: para `plan`/`apply`
  de todos los recursos gestionados se listan acciones explícitas por servicio
  (sin `Action: "*"` y sin managed policies `*FullAccess`/`AdministratorAccess`).
  Varios servicios (IAM, SageMaker, ECS) usan `Resource: "*"` porque sus acciones
  no admiten restricción por recurso. `iam:PassRole` queda con condición
  `iam:PassedToService` acotada a ECS/SageMaker.
- **security group — CIDR por defecto `0.0.0.0/0`**: el default deja la UI
  alcanzable desde cualquier IP mientras la task está viva (MLflow sin auth por
  defecto). Conviene setear `mlflow_allowed_cidr` a la IP del operador en
  `terraform.tfvars`.
- **6.1/6.2 — archivos locales faltaban**: `infra/terraform.tfvars` y
  `infra/backend.hcl` (ambos gitignored) no existían en la máquina. Se recrearon
  desde los `.example` con los valores reales (profile `AWS_PROFILE`, bucket de
  state `<account-id>-tfstate`, tabla `terraform-locks`). En la primera pasada el
  `terraform.tfvars` recreado omitía `user_profile_name`, y el plan proponía
  **reemplazar** el user profile SageMaker (`<user-profile>` → `sinca-dev-user`,
  default) y cambiar el owner del Space. Se restauró
  `user_profile_name = "<user-profile>"` (valor real leído del state con
  `terraform state show`); el plan quedó en 14 add / 1 change / 0 destroy.
- **6.2 — drift preexistente en el dominio SageMaker**: el plan incluye un
  update in-place de `aws_sagemaker_domain.studio` que remueve
  `studio_web_portal_settings {}` (bloque vacío presente en el state y no
  declarado en el config; default del provider aws 5.100). No es destructivo,
  es ajeno a este change, y no se tocó `sagemaker-domain.tf`.
- **6.4 / 7.1 — `boto3` faltaba en la imagen oficial**: la primera corrida del
  contenedor murió con `ModuleNotFoundError: No module named 'boto3'` (la
  imagen `ghcr.io/mlflow/mlflow` no lo trae). Se agregó
  `RUN python -m pip install --no-cache-dir boto3` al `Dockerfile` y se
  reconstruyó/pusheó la imagen.
- **7.1 — MLflow 3.x rechaza Host no-local (HTTP 403)**: el *security
  middleware* nuevo devolvía `403 Forbidden` con
  `Rejected request with invalid Host header`. Se agregó
  `--allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-*}"` al `mlflow server` del
  entrypoint (con `*`, MLflow no agrega el middleware de validación de Host;
  el acceso ya está restringido por el security group al IP del operador).
- **7.1 — OOM con 1024 MiB**: la task fue matada por memoria
  (`exit 137`, `OutOfMemoryError`). Se subió la task Fargate a
  `cpu=1024` / `memory=2048` (1 vCPU / 2 GiB) en `variables.tf`; con eso la UI
  respondió HTTP 200 de forma estable.
- **Entorno (no repo) — CA bundle de WSL**: el `~/.aws/config` de WSL tenía
  `ca_bundle` apuntando a un path de Windows (`C:\...\.claude-ca-bundle\...`),
  inválido dentro de WSL (fallaba el SSL de `aws`). Se creó un bundle dedicado
  para esta cuenta (`~/.aws/<ca-bundle>.pem`, copia del store del
  sistema) y se apuntó solo el profile `AWS_PROFILE` a él. No afecta al repo.
