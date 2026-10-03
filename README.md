# forecast-sinca-aws — Predicción de Calidad del Aire (SINCA)

Proyecto personal de MLOps de punta a punta para predecir calidad del aire
(MP2.5/MP10) en la zona saturada de Temuco/Padre Las Casas (Región de La
Araucanía), con horizonte de **24 horas**.

## Qué hace

- Ingieta datos horarios de las **3 estaciones activas** de la zona saturada
  (Padre Las Casas II ID 263, Ñielol, Las Encinas) + variables meteorológicas
  height de capa límite (Open-Meteo), feriados y calendario GEC.
- Calcula el target derivado **ICAP_zona** (máximo entre las 3 estaciones,
  fórmula oficial D.S. 12/2011 y equivalente MP2,5) para comparar contra el
  benchmark implícito del MMA.
- Entrena modelos (baseline, LightGBM, y línea autoencoder+DMD/Koopman) y
  produce predicciones a 24h.

## Estructura del repo

```
infra/        # Terraform (IAM, SageMaker Studio, buckets, red, ECR/ECS, OIDC)
infra/scripts/# scripts de operación (on_start.sh del Space, mlflow-server.sh)
docker/       # Dockerfile + entrypoint de la imagen MLflow
pipeline/     # scripts .py del pipeline ML (futuro)
tests/        # unit + integration (futuro)
notebooks/    # solo exploración (futuro)
modules/      # blueprints de decisión por dominio (guías, no specs)
openspec/     # changes activos y specs consolidadas
.github/      # workflows de GitHub Actions (ci, infra-cd, pipeline-cd)
```

## Estado actual

**Fase 1 — Setup (completada).** Infraestructura base (`setup-infra-base`):
buckets S3 (`sinca-data`, `sinca-mlflow`), entorno de desarrollo remoto
(SageMaker Studio Space sobre EFS), y roles IAM de base (`sinca-dev-role`,
`sinca-training-role`). Tracking y CI/CD (`setup-mlflow-ci-cd`): servidor
MLflow self-hosted on-demand (Fargate + SQLite en S3, imagen en ECR), OIDC de
GitHub Actions con dos roles de mínimo privilegio, y los 3 workflows. Aún sin
pipeline de datos ni modelos.

## Servidor MLflow (on-demand)

El tracking usa MLflow self-hosted: backend SQLite (`mlflow.sqlite`) persistido
en `s3://sinca-mlflow/_mlflow/` y artefactos directo a
`s3://sinca-mlflow/_mlflow/mlruns`. No hay nada corriendo 24/7 — el servidor
se levanta como task Fargate solo cuando se quiere ver la UI.

```powershell
# Prender la UI (descarga el estado de S3 y expone la UI en :5000)
$env:AWS_PROFILE="<profile>"
bash infra/scripts/mlflow-server.sh up

# Apagar (dispara SIGTERM; el entrypoint sube el .sqlite de vuelta a S3)
bash infra/scripts/mlflow-server.sh down
```

La imagen custom vive en ECR (`sinca-mlflow`). Se construye y publica a mano
(no hay workflow de build todavía):

```powershell
docker build -t sinca-mlflow docker/
aws ecr get-login-password --profile $env:AWS_PROFILE | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag sinca-mlflow:latest <account>.dkr.ecr.us-east-1.amazonaws.com/sinca-mlflow:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/sinca-mlflow:latest
```

> El security group del servidor permite el puerto 5000 desde
> `var.mlflow_allowed_cidr` (default `0.0.0.0/0`). Conviene restringirlo a la IP
> del operador en `terraform.tfvars`.

## CI/CD (GitHub Actions)

Autenticación a AWS por OIDC (sin Access Keys en GitHub Secrets). Tres workflows:

- **`ci.yml`** — en PR: `terraform fmt -check` + `terraform validate` (sin AWS).
- **`infra-cd.yml`** — en PR: `terraform plan`; al mergear a `main`:
  `terraform apply` (asume `sinca-github-infra-role`).
- **`pipeline-cd.yml`** — skeleton; actualizará la definición del pipeline
  cuando exista `pipeline/` (Fase 2; asume `sinca-github-pipeline-role`).

Requiere configurar en el repo (Settings → Variables):
`AWS_INFRA_ROLE_ARN`, `AWS_PIPELINE_ROLE_ARN` (outputs de Terraform) y
`TFSTATE_BUCKET` (bucket del remote state).

## Cuenta y credenciales

Cuenta personal (Account ID `AWS_ACCOUNT_ID`), profile `AWS_PROFILE` (nunca
`default` ni un profile ajeno). Región `us-east-1`. Ver `PREREQUISITES.md`.

## Flujo de trabajo

Coordinación entre los agentes `arquitecto` (Arquitecto) y `implementador` (Implementador)
con el operador humano. Ver `WORKFLOW.md` y `AGENTS.md`.
