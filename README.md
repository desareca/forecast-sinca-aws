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
infra/        # Terraform (IAM, SageMaker Studio, buckets, red, ECR/ECS, OIDC, scraper, training)
infra/scripts/# scripts de operación (on_start.sh del Space, mlflow-server.sh)
docker/       # Dockerfile (MLflow) + scraper.Dockerfile (datos) + train.Dockerfile (entrenamiento)
pipeline/     # datos (scraper/ICAP/validación/persistencia) + modelado (features/baseline/train/evaluate)
tests/        # unit + integration (futuro)
notebooks/    # notebook base de análisis (puntuales en .gitignore)
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

**Fase 2 — Pipeline de datos (completada).** Change `data-pipeline`: código del
pipeline (`pipeline/`), imagen (`docker/scraper.Dockerfile`), infra Terraform
(ECR, task Fargate, IAM, schedule), backfill de 5 años y smoke test. Las 6
series horarias validadas viven en `s3://sinca-data/validated/`.

**Fase 3 — Baseline (completada).** Change `baseline`: feature engineering
(`pipeline/features/`), baseline de persistencia (`pipeline/baseline/`),
entrenamiento LightGBM con random search (`pipeline/train/`), evaluación
(`pipeline/evaluate/`), imagen de entrenamiento (`docker/train.Dockerfile`) e
infra Terraform (ECR `sinca-train` + rol de training finalizado). Training Job
completado y verificado en MLflow (7 runs: 6 LightGBM + persistencia). Ver
["Baseline (Fase 3)"](#baseline-fase-3) y [`EXPERIMENT-LOG.md`](EXPERIMENT-LOG.md).

## Entorno de desarrollo (Space + VS Code local)

El trabajo diario se hace **en local con VS Code**, conectado por **SSH sobre
SSM** al Space `sinca-dev` de SageMaker Studio (dominio `sinca-studio`,
`ml.t3.large` + EBS 5 GB). El VS Code local es solo la ventana: el código, el
venv, el `git` y el acceso a S3 corren en el Space (rol `sinca-dev-role`, sin
claves ni credenciales en el repo).

- **Conectar**: VS Code → panel **AWS Toolkit** → SageMaker → Space `sinca-dev`
  → **Start and connect** (extensiones AWS Toolkit + Remote-SSH). El Space
  queda `InService` y factura (~USD 0.05–0.08/h).
- **Apagar**: SageMaker Console → Studio → Spaces → `sinca-dev` → **Stop**
  (el EBS de 5 GB persiste; solo el cómputo deja de cobrar).
- **Git local ↔ Space**: son dos clones. Commit/push en uno y `git pull` en el
  otro para sincronizar.

## Análisis (dashboard y notebooks)

### Dashboard Streamlit (on-demand)

Dashboard interactivo para revisar el baseline por run: series real vs predicho
con bandas ICAP, `ICAP_zona` real vs predicho, feature importance y RMSE
LightGBM vs persistencia. Lee los diagnósticos que persiste cada run
(`predictions.parquet`, `feature_importance.parquet`, `zone_icap.parquet`) desde
`s3://sinca-mlflow/_mlflow/`, sin servidor MLflow. Si un run no tiene
diagnósticos (corridas viejas), muestra solo métricas con un aviso.

**Local en contenedor (contra el fixture, sin AWS):**

```powershell
# Genera el estado MLflow sintético (una vez, o cuando cambie el fixture)
python pipeline/analysis/fixtures/make_fixture.py

docker build -f docker/analysis.Dockerfile -t sinca-analysis .
docker run --rm -p 8501:8501 -v "${PWD}:/app" sinca-analysis `
  pipeline/analysis/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

Abrí http://localhost:8501. El código va montado como volumen (hot-reload sin
rebuild). Para datos reales desde el contenedor, montá `~/.aws` y pasá
`-e AWS_PROFILE -e AWS_DEFAULT_REGION`.

**En el Space (datos reales, port-forward):**

```bash
# Dentro del Space sinca-dev (VS Code -> AWS Toolkit -> Start and connect)
git pull
pip install -r requirements-analysis.txt
streamlit run pipeline/analysis/streamlit_app.py --server.port 8501
```

Luego, en VS Code, port-forward del **8501** y abrí `http://localhost:8501`. Al
terminar, **apagá el Space** (Spaces → `sinca-dev` → Stop) para no dejar cómputo
corriendo (~USD 0.06/h).

**Dependencias:** `requirements-analysis.txt` es la fuente única para el
contenedor local y para el venv del Space (streamlit, plotly, pandas, mlflow,
boto3, pyarrow).

### Notebook de análisis puntual

`notebooks/analisis_baseline.ipynb` es una plantilla versionada para análisis
ad-hoc en JupyterLab del Space (no es código de producción ni fuente de verdad).
Flujo:

1. **Encender**: SageMaker Studio → **Spaces** → `sinca-dev` → **Open** (o
   **Start and connect** desde VS Code/AWS Toolkit).
2. **Trabajar**: abrir el notebook en JupyterLab del Space. Corre con el rol del
   Space (`sinca-dev-role`), sin credenciales explícitas; lee
   `sinca-data/validated/` y `sinca-mlflow/_mlflow/`.
3. **Apagar**: **Spaces** → `sinca-dev` → **Stop** (el EBS persiste; solo deja de
   cobrar el cómputo).

Los notebooks puntuales están en `.gitignore`; versioná el que quieras conservar
con `git add -f`.

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

## Baseline (Fase 3)

Primer modelo del proyecto: **persistencia (naive) + LightGBM**. Predice, por
serie (MP2.5/MP10 × 3 estaciones), el **promedio móvil de 24h a t+24h**
(`mean(MP(t+1 … t+24h))`), que es lo que consume la fórmula oficial del ICAP.
El `ICAP_zona` se deriva siempre en post-proceso.

```
pipeline/
├── features/build.py      # features + target + partición (ventana deslizante + test held-out)
├── baseline/persistence.py# baseline naive (último promedio móvil 24h observado)
├── train/train.py         # 6 modelos LightGBM (RMSE ponderado) + random search + MLflow
├── train/mlflow_log.py    # logging directo a MLflow + sync de estado con S3
├── train/launch.py        # lanzador del SageMaker Training Job (manual)
└── evaluate/evaluate.py   # métricas: RMSE ponderado, ICAP oficial, percentiles propios
```

- **Features**: lags propios (1h/24h/48h/72h/168h), meteorología con lag 24h
  (temp, humedad, presión, velocidad de viento; radiación solo donde la estación
  la publica), altura de capa límite (valor + lag 24h), feriados, día de semana
  cyclic, ventana GEC y Fourier (diaria/semanal/anual).
- **Loss**: RMSE ponderado `w_i = 1 + α·max(0, y_i − umbral_regular)`
  (`α=1`, `umbral_regular` = ancla ICAP 100: MP10 150 / MP2.5 50).
- **Partición**: ventana deslizante (train 2 años / valid 1 mes / step 1 mes)
  + test final held-out `2026-07-01 → 2026-10-04`.
- **Tracking**: MLflow en modo **logging directo** (sin servidor): SQLite local
  + `mlruns/`, subidos al terminar a `s3://sinca-mlflow/_mlflow/`. Convención de
  run `sinca-{estacion}-{modelo}-v{n}`. Modelos también en
  `s3://sinca-mlflow/model-artifacts/`.
- **Resultado (test 3 meses held-out, RMSE)**: LightGBM mejora vs persistencia
  en Las Encinas y Nielol (4/6 series, −10% a −18%); empeora en Padre Las Casas
  II (pm10/pm25). Supervisar en Fase 4: PLC es la serie de mayor variabilidad y
  ahí la radiación es casi toda nula → evaluar features extra (`rain`, viento de
  Open-Meteo) o un baseline por estación.

### Correr local (sin AWS, sin escribir)

```powershell
# Requiere los datos validados en disco (mismo layout que validated/ en S3).
python pipeline/train/train.py --local-validated-dir <dir>/validated --no-upload `
  --search-trials 2 --search-windows 1 --stations "Nielol" --pollutants pm25
```

### Build + push de la imagen de entrenamiento

```powershell
$env:AWS_PROFILE="<profile>"
docker build -f docker/train.Dockerfile -t sinca-train .
$acct = aws sts get-caller-identity --query Account --output text
aws ecr get-login-password | docker login --username AWS --password-stdin $acct.dkr.ecr.us-east-1.amazonaws.com
docker tag sinca-train:latest $acct.dkr.ecr.us-east-1.amazonaws.com/sinca-train:latest
docker push $acct.dkr.ecr.us-east-1.amazonaws.com/sinca-train:latest
```

### Lanzar el Training Job (manual)

La imagen lleva solo el runtime; el código viaja por `source_dir`/`entry_point`
desde `pipeline/`. Requiere el SDK de SageMaker **v2** en el entorno local
(ver `requirements-dev.txt`: `pip install "sagemaker>=2,<3"` — la línea v3
elimina la API `Estimator`). Los ARNs salen de los outputs de Terraform:

```powershell
$env:AWS_PROFILE="<profile>"
$role = terraform -chdir=infra output -raw training_role_arn
$image = "$(terraform -chdir=infra output -raw train_ecr_repository_url):latest"
python pipeline/train/launch.py --role-arn $role --image-uri $image --wait
```

### Ver las métricas

```powershell
# Listar el tracking subido a S3 (sqlite + artefactos por run).
aws s3 ls s3://sinca-mlflow/_mlflow/ --recursive
# O levantar la UI de MLflow (ver sección "Servidor MLflow").
```

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
