## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar acceso a las fuentes: SINCA (`sinca.mma.gob.cl`), Open-Meteo, y Nager.Date (`date.nager.at`) (HTTP público, sin key)

## 1. Escritura del pipeline (`pipeline/`)

- [x] 1.1 Crear `pipeline/scraper/sinca.py` — scraping CSV de `apub.tsindico2.cgi` parametrizado por ID de estación (263, Ñielol, Las Encinas), descubriendo `macropath`/`macro` desde la página de la estación, extrayendo MP2.5/MP10/gases/meteorología horarios
- [x] 1.2 Crear `pipeline/scraper/open_meteo.py` — altura de capa límite horaria desde Open-Meteo (JSON)
- [x] 1.3 Crear `pipeline/scraper/feriados.py` — feriados desde Nager.Date API (`date.nager.at/api/v3/PublicHolidays/{year}/CL`, 1 vez por corrida), filtrando feriados regionales por el campo `counties` (nacionales + La Araucanía)
- [x] 1.4 Crear `pipeline/icap/icap.py` — fórmula piecewise-linear del D.S. 12/2011 con 3 anclas por contaminante (MP10: 0→0, 100→150, 500→330; MP2.5: 0→0, 100→50, 500→170) sobre promedio móvil 24h
- [x] 1.5 Crear `pipeline/validate/schemas.py` — pandera `DataFrameSchema`/`SchemaModel` con las 5 dimensiones (validez, completitud, unicidad, oportunidad, consistencia)
- [x] 1.6 Crear `pipeline/persist/s3.py` — escritura Parquet a `sinca-data/{raw,validated,quarantine}/` con particionado por dataset/estación y año/mes, y motivo de rechazo adjunto en cuarentena
- [x] 1.7 Crear `pipeline/entrypoint.py` — orquestador: scraper → icap → validate → persist, con modo backfill (rango de fechas) e incremental (últimas 48h)

## 2. Escritura de la imagen del scraper

- [x] 2.1 Crear `docker/scraper.Dockerfile` — Python 3.11 + `pandas`, `pandera`, `numpy`, `requests`, `beautifulsoup4`, `boto3`, `pyarrow`, con el código de `pipeline/` copiado y `ENTRYPOINT ["python", "entrypoint.py"]`

## 3. Escritura de Terraform

- [x] 3.1 Crear `infra/ecr-scraper.tf` — repo ECR `sinca-scraper`
- [x] 3.2 Crear `infra/iam-scraper.tf` — rol `sinca-scraper-task-role` (S3 `sinca-data/*` + logs) y rol de ejecución (pull ECR + logs)
- [x] 3.3 Crear `infra/fargate-scraper.tf` — task definition Fargate (imagen ECR, default VPC, CPU/memoria `512`/`1024`)
- [x] 3.4 Crear `infra/schedule.tf` — regla EventBridge Schedule (cron ~01:00 America/Santiago) que dispara `ecs:RunTask`
- [x] 3.5 Actualizar `infra/outputs.tf` con los ARNs de ECR, roles y schedule

## 4. Documentación

- [x] 4.1 Actualizar `README.md` (estado: Fase 2 en progreso; cómo correr el backfill y el incremental; estructura de `pipeline/`)
- [x] 4.2 Actualizar `project-decisions.md` §3 (cómputo del scraper: Fargate on-demand) y §7 (horario del schedule)

## 5. Ejecución real (operador corre y pega output)

- [x] 5.1 Build de la imagen (sin push todavía): `docker build -f docker/scraper.Dockerfile -t sinca-scraper .`
- [x] 5.2 `terraform init` (si hace falta) y `terraform plan` — revisar el plan con el operador antes de aplicar
- [x] 5.3 `terraform apply` (crea ECR, task Fargate, IAM, schedule)
- [x] 5.4 Push de la imagen a ECR (el repo recién existe tras el apply)
- [x] 5.5 Backfill manual de 5 años (2021-01-01 → hoy) ejecutando la task con el rango de fechas

## 6. Verificación funcional

- [x] 6.1 Smoke test manual: correr el pipeline de punta a punta (incremental) y confirmar que los datos validados aparecen en `sinca-data/validated/` (listar con `aws s3 ls`)
- [x] 6.2 Confirmar que los datos rechazados (si los hay) caen en `quarantine/` con motivo de rechazo
- [x] 6.3 Activar el schedule y confirmar la primera corrida automática (o verificar que la regla existe y dispara la task)

## Notas de implementación

- **1.1 (SINCA)**: el endpoint que nombraba la tarea (`apub.htmlindico2.cgi`, HTML) hoy da 500. El endpoint vigente que devuelve datos es el CSV `apub.tsindico2.cgi?outtype=xcl`, construyendo `macro=./<macropath>/<macro>.ic` (con la extensión `.ic`) y pasando `macropath` en la query. El scraper descubre `macropath`/`macro` por parámetro desde la página de la estación (robusto a cambios), descarta los macros `diario` (solo `horario`) y, cuando un parámetro tiene varios macros con coberturas distintas (típico en meteorología), consulta todos los que intersectan el rango y los concatena. Verificado con las 3 estaciones para PM10/PM25, gases (0001/0002/0003/0004/0008/0NOX) y met (TEMP/RHUM/PRES/RAIN/WDIR/WSPD/GLOB), en backfill 2021 y en datos recientes.
- **1.1 (PLC II)**: la página de estación de Padre Las Casas II muestra "Sin datos" en las filas de MP10/MP2.5, pero el endpoint **sí devuelve** el histórico y los datos recientes. No es un bloqueo; el scraper se basa en el descubrimiento de links y la descarga, no en el texto "Sin datos".
- **1.1 (unidades met)**: `RHUM` (humedad relativa) llega como fracción (p.ej. `0,92`) en algunas series, no como porcentaje 0-100. Se persiste el valor crudo; el rango de validez (0-100) no lo rechaza. A revisar en Fase 3 al construir features.
- **1.5 (validez de gases)**: el SINCA entrega valores levemente negativos en horas limpias (ruido instrumental, p.ej. `no` = -2.14). Se amplió el límite inferior de los gases a -50 µg/m³ para no cuarentenar datos válidos. MP2.5/MP10 se mantienen en ≥0 según el design.
- **1.5 (dtype timestamp)**: pandera compara el dtype tz-aware por identidad de objeto tz; `pd.DatetimeTZDtype(tz="America/Santiago")` usa pytz mientras la serie usa `zoneinfo.ZoneInfo`, y falla. Se usa `pd.DatetimeTZDtype(tz=ZoneInfo("America/Santiago"))`.
- **1.7 (umbral de cuarentena)**: se implementó `MAX_QUARANTINE_RATE` (default 0.2, override por env). Si una estación supera el umbral, no se persiste `validated` para esa estación y la corrida termina con exit code 1. `oportunidad` (frescura 48h) solo se exige en modo `incremental` (en `backfill` no aplica).
- **Verificación local**: se corrió el pipeline en modo `--dry-run` (incremental y backfill 2021) con las dependencias reales — las 3 estaciones validan 0 rechazadas, ICAP verificado contra los valores intermedios del design (MP10 195→200/240→300; MP2.5 80→200/110→300), y la escritura S3 se probó con un cliente fake (particiones, Parquet y `reason.json`). `terraform fmt`/`validate` OK.
- **Orden de las tareas 5.x (resuelto)**: el push a ECR requiere que el repo exista, y el repo lo crea `terraform apply`. Se reordenó: 5.1 build (sin push) → 5.2 plan → 5.3 apply (crea ECR) → 5.4 push → 5.5 backfill. Alternativa descartada: `terraform apply -target=aws_ecr_repository.scraper` antes del push (rompe el flujo plan→apply completo).
- **Build context**: se agregó `.dockerignore` en la raíz para que el build del scraper (contexto = raíz del repo) no envíe `.venv/`, `.git/`, `infra/.terraform/`, etc.
- **`.env` / variables de entorno**: se agregó `.env.example` en la raíz (gitignored `.env`, versionado `.env.example`) y `load_dotenv()` en `entrypoint.py` (stdlib, no sobreescribe variables ya seteadas; no-op en Fargate). El flag `--bucket` ahora toma default de `DATA_BUCKET` para honrar la env que setea la task definition.
- **5.5 (backfill real)**: task Fargate exit 0, ~166s para 2021-01-01→2026-10-04. Resultado: PLC II 48.805 válidas / 1.660 rechazadas (3.3%); Ñielol 50.049 / 416 (0.8%); Las Encinas 50.463 / 2 (0.0%). S3: `raw/`=286, `validated/`=286, `quarantine/`=154 objetos. El scraper trae los ~50k registros por estación/parámetro en una sola request (4s por parámetro).
- **5.5 (motivos de rechazo observados)**: `rhum` y `pres` fuera de rango son la mayoría de los rechazos. `rhum` aparece con valores fuera de 0-100 (además de la fracción 0-1 ya anotada): probablemente valores atípicos del sensor (p. ej. >100% o negativos). `pres` con valores fuera de 900-1100 hPa. Son datos basura plausibles, bien cuarentenados. **A revisar en Fase 3** si el rango de `pres` debe ampliarse o si esos registros son ruido genuino; con la tasa de cuarentena <20% por estación, el pipeline no se detiene.
- **6.1 (incremental real)**: task Fargate exit 0. Las 3 estaciones: 71 filas válidas / 0 rechazadas. Datos recientes presentes en `validated/`.
- **6.2 (cuarentena)**: confirmado `reason.json` adjunto a cada Parquet rechazado, con el motivo (dimensión + regla). Ejemplos: `["pm10: in_range(0.0, 1000.0)","pm25: in_range(0.0, 1000.0)"]` y `["pres: in_range(900.0, 1100.0)","rhum: in_range(0.0, 100.0)"]`.
- **6.3 (schedule)**: `sinca-scraper-daily` creado, `state=ENABLED`, `cron(0 1 * * ? *)` America/Santiago, target el cluster `sinca-scraper` con el rol `sinca-scraper-scheduler-role`. La primera corrida automática queda programada para ~01:00 local del día siguiente; no se forzó una ejecución del schedule (las tareas 5.5/6.1 ya validaron la task).
- **Perfil AWS (trampa operativa)**: el shell trae `AWS_PROFILE` apuntando a **otra cuenta** (una cuenta de terceros, no la del proyecto). El profile del proyecto es el de la cuenta personal (`AWS_ACCOUNT_ID`, según `terraform.tfvars`). Todos los comandos de ejecución real (push a ECR, `ecs run-task`, `s3 ls`, `scheduler get-schedule`) se corrieron con el profile del proyecto explícito. Nunca usar el `AWS_PROFILE` del entorno sin verificar la cuenta con `sts get-caller-identity`.
