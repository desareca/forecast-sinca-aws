## Context

Proyecto `forecast-sinca-aws` (predicción ICAP zona saturada Temuco/Padre Las Casas, horizonte 24h, 6 series target MP2.5/MP10 de 3 estaciones). Cuenta AWS personal (Account ID `AWS_ACCOUNT_ID`, profile `AWS_PROFILE`, región `us-east-1`). Repo GitHub `desareca/forecast-sinca-aws`.

No existe infraestructura previa. Este change (Fase 1 "Setup", subchange `setup-infra-base`) crea la base: remote state, buckets, entorno de desarrollo remoto y roles IAM. Un segundo change (`setup-mlflow-ci-cd`) agrega MLflow self-hosted + CI/CD.

Decisiones ya cerradas en `project-decisions.md` (aún por llenar en el Paso 1 del workflow, reflejadas acá): tracking MLflow self-hosted, Model Registry MLflow (reabrir en Fase 6), ramas por feature branch, región `us-east-1`.

## Goals / Non-Goals

**Goals:**
- Tener un entorno de desarrollo remoto persistente (Space) reproducible vía Terraform, que pueda apagarse sin perder código/contexto.
- Dejar creados los buckets S3 con los prefijos donde caerán los datos y el tracking.
- Dejar definidos los roles IAM de base con mínimo privilegio (dev env + training skeleton).
- Remote state de Terraform operativo desde el día uno (regla fija del blueprint `cicd`).

**Non-Goals:**
- Servidor MLflow (Fargate self-hosted) → change `setup-mlflow-ci-cd`.
- OIDC + workflows de GitHub Actions → change `setup-mlflow-ci-cd`.
- Lógica de negocio (scraper SINCA, features, modelos) → Fases 2+.
- Pipeline de datos o schedule → Fases 2+.
- Secrets Manager: no aplica en esta fase (SINCA/Open-Meteo/feriados son HTTP públicos).

## Decisions

### 1. Red de SageMaker Studio sin VPC propia (`PublicInternetOnly`)

El domain de SageMaker Studio se crea con `app_network_access_type = "PublicInternetOnly"` y **sin `vpc_id`**. Evita el riesgo clásico de `quick setup` en consola que levanta una VPC + NAT Gateway facturando 24/7 (Principio 3 de `costos`).

- Alternativa considerada: domain con VPC propia para control de red → descartada, introduce un NAT Gateway (~$0.045/h) sin beneficio para este proyecto personal.
- `quick setup` de consola → descartado; todo se declara en Terraform (visible y versionable).

### 2. Home del Space sobre EFS + lifecycle config

El cómputo del Space es efímero; lo persistente vive en EFS (`/home/sagemaker-user`). El `~/.on_start` (lifecycle config) reinstala lo efímero al prender: Node.js + Open Code CLI, `aws` CLI + `terraform`, clona `desareca/forecast-sinca-aws`, setea credenciales git. Todo lo reusable vive en EFS; todo lo efímero se reinstala en `~/.on_start` (regla del blueprint `dev-environment`).

### 3. Tamaño de instancia `ml.t3.large`

Mínima útil (8 GB RAM): suficiente para editar, correr `openspec` y pruebas ligeras. El entrenamiento pesado va a SageMaker Training Job (no al Space), según `compute`/`dev-environment`.

### 4. Dos buckets separados para IAM limpio

`bucket1 = sinca-data` (datos) y `bucket2 = sinca-mlflow` (tracking), en vez de un único bucket con prefijos. Motivo: acotar los permisos IAM por bucket (un rol que solo toca datos no tiene alcance sobre `_mlflow/` y viceversa), alineado con mínimo privilegio.

- Convención dentro de `sinca-data`: `raw/` → `validated/` → `quarantine/` (Parquet), según `data-quality`.
- `sinca-mlflow`: prefijo `_mlflow/` (`mlflow.sqlite` + `mlruns/`), según `tracking-experimentos`.

### 5. IAM mínimo privilegio por rol

- **Rol dev environment** (`sinca-dev-role`): S3 `sinca-data/*` y `sinca-mlflow/_mlflow/*` (read/write acotado a esos prefijos), `cloudwatch:logs` en el log group propio, `ssm` para SSH remoto. Sin permisos IAM/full.
- **Rol training** (`sinca-training-role`, skeleton): S3 `sinca-data/validated/*` read + write de artefactos a `sinca-mlflow/model-artifacts/*`, `cloudwatch:logs`. Se amplía en fases posteriores, siempre acotado por prefijo.

Nunca policies `*FullAccess` ni `AdministratorAccess` (regla fija de `AGENTS.md`).

### 6. Remote state de Terraform (backend S3 + DynamoDB)

`terraform { backend "s3" }` con `bucket = <account-id>-tfstate`, `key = forecast-sinca-aws/infra.tfstate`, `dynamodb_table = terraform-locks`, región `us-east-1`. Como el bucket no puede crearse desde el propio backend (chicken-egg), se bootstrapa manualmente (tarea 0 con el operador).

## Risks / Trade-offs

- **Chicken-egg del backend S3** → mitigación: bootstrap manual del bucket + tabla antes del primer `terraform init` (tarea 0, corre el operador).
- **El Space queda prendido y factura** → mitigación: hábito de apagar el Space (cycle de `dev-environment`); costo bajo si se apaga; se documenta en `README`.
- **EFS persiste y acumula** → mitigación: EFS es barato ($0.30/GB-mes) y el volumen de este proyecto es chico; no amerita lifecycle policy por ahora.
- **Profile equivocado (otra cuenta)** → mitigación: `sts get-caller-identity --profile $AWS_PROFILE` como verificación explícita en la fase de ejecución; el account devuelto debe ser `AWS_ACCOUNT_ID`.
- **Divergencia entre `project-decisions.md` y este change** → mitigación: `project-decisions.md` se llena en el Paso 1 del workflow (antes o junto con este change) y es la fuente de verdad; cualquier ajuste posterior se hace vía change nuevo.
