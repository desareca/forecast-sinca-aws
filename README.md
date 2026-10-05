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
infra/        # Terraform (IAM, SageMaker Studio, buckets, red, ECR/ECS, OIDC, scraper)
infra/scripts/# scripts de operación (on_start.sh del Space, mlflow-server.sh)
docker/       # Dockerfile (MLflow) + scraper.Dockerfile (pipeline de datos)
pipeline/     # pipeline de datos (scraper SINCA/Open-Meteo/feriados, ICAP, validación, persistencia)
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
GitHub Actions con dos roles de mínimo privilegio, y los 3 workflows.

**Fase 2 — Pipeline de datos (en progreso).** Change `data-pipeline`: código del
pipeline (`pipeline/`) + imagen (`docker/scraper.Dockerfile`) + infra Terraform
(ECR, task Fargate, IAM, schedule). Pendiente: build/push de la imagen,
`terraform apply`, backfill de 5 años y smoke test (tareas 5.x/6.x). Aún sin
modelos.

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

## Pipeline de datos

Ingiere las 3 estaciones SINCA (Padre Las Casas II ID 263, Ñielol, Las Encinas),
la altura de capa límite de Open-Meteo y los feriados (Nager.Date), calcula el
ICAP por estación (D.S. 12/2011, 3 anclas), valida con pandera (5 dimensiones) y
persiste Parquet en S3. Corre como task Fargate on-demand, disparada por un
schedule EventBridge diario (~01:00 America/Santiago) o manualmente.

```
pipeline/
├── entrypoint.py        # orquestador: scraper → icap → validate → persist
├── config.py            # estaciones, variables/unidades, endpoints, particionado
├── scraper/
│   ├── sinca.py         # CSV de apub.tsindico2.cgi (macropath/macro descubierto)
│   ├── open_meteo.py    # altura de capa límite (JSON)
│   └── feriados.py      # Nager.Date API
├── icap/icap.py         # ICAP piecewise-lineal (MP10/MP2.5)
├── validate/schemas.py  # pandera (validez, completitud, unicidad, oportunidad, consistencia)
└── persist/s3.py        # Parquet raw/ → validated/ → quarantine/
```

Datos en `s3://sinca-data/`:

```
{raw,validated,quarantine}/sinca/estacion=<slug>/year=<yyyy>/month=<mm>/data.parquet
{raw,validated}/open_meteo/year=<yyyy>/month=<mm>/data.parquet
{raw,validated}/feriados/year=<yyyy>/data.parquet
```

En `quarantine/` cada Parquet lleva al lado un `reason.json` con el motivo de
rechazo (dimensión + regla). Modos: `incremental` (últimas 48h, exige frescura)
y `backfill` (rango de fechas).

### Correr localmente (sin AWS, sin escribir)

```powershell
python pipeline/entrypoint.py --dry-run --mode incremental
python pipeline/entrypoint.py --dry-run --mode backfill --from 2021-01-01 --to 2024-12-31
```

### Build + push de la imagen

```powershell
$env:AWS_PROFILE="<profile>"
$acct = aws sts get-caller-identity --query Account --output text
aws ecr get-login-password | docker login --username AWS --password-stdin $acct.dkr.ecr.us-east-1.amazonaws.com
docker build -f docker/scraper.Dockerfile -t sinca-scraper .
docker tag sinca-scraper:latest $acct.dkr.ecr.us-east-1.amazonaws.com/sinca-scraper:latest
docker push $acct.dkr.ecr.us-east-1.amazonaws.com/sinca-scraper:latest
```

### Correr la task (incremental)

```powershell
aws ecs run-task --cluster sinca-scraper --task-definition sinca-scraper --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-ids>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}"
```

### Backfill (override del comando)

El `ENTRYPOINT` de la imagen es `python entrypoint.py`, así que el override solo
agrega los argumentos:

```powershell
aws ecs run-task --cluster sinca-scraper --task-definition sinca-scraper --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-ids>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}" `
  --overrides '{\"containerOverrides\":[{\"name\":\"scraper\",\"command\":[\"--mode\",\"backfill\",\"--from\",\"2021-01-01\",\"--to\",\"2026-10-04\"]}]}'
```

Los IDs de subnet y security group salen de los outputs de Terraform
(`scraper_security_group_id`) y del default VPC.

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
