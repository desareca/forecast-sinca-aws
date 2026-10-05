## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [ ] 0.2 Confirmar acceso a las fuentes: SINCA (`sinca.mma.gob.cl`), Open-Meteo, y Nager.Date (`date.nager.at`) (HTTP público, sin key)

## 1. Escritura del pipeline (`pipeline/`)

- [ ] 1.1 Crear `pipeline/scraper/sinca.py` — scraping CSV de `apub.tsindico2.cgi` parametrizado por ID de estación (263, Ñielol, Las Encinas), descubriendo `macropath`/`macro` desde la página de la estación, extrayendo MP2.5/MP10/gases/meteorología horarios
- [ ] 1.2 Crear `pipeline/scraper/open_meteo.py` — altura de capa límite horaria desde Open-Meteo (JSON)
- [ ] 1.3 Crear `pipeline/scraper/feriados.py` — feriados desde Nager.Date API (`date.nager.at/api/v3/PublicHolidays/{year}/CL`, 1 vez por corrida), filtrando feriados regionales por el campo `counties` (nacionales + La Araucanía)
- [ ] 1.4 Crear `pipeline/icap/icap.py` — fórmula piecewise-linear del D.S. 12/2011 con 3 anclas por contaminante (MP10: 0→0, 100→150, 500→330; MP2.5: 0→0, 100→50, 500→170) sobre promedio móvil 24h
- [ ] 1.5 Crear `pipeline/validate/schemas.py` — pandera `DataFrameSchema`/`SchemaModel` con las 5 dimensiones (validez, completitud, unicidad, oportunidad, consistencia)
- [ ] 1.6 Crear `pipeline/persist/s3.py` — escritura Parquet a `sinca-data/{raw,validated,quarantine}/` con particionado por dataset/estación y año/mes, y motivo de rechazo adjunto en cuarentena
- [ ] 1.7 Crear `pipeline/entrypoint.py` — orquestador: scraper → icap → validate → persist, con modo backfill (rango de fechas) e incremental (últimas 48h)

## 2. Escritura de la imagen del scraper

- [ ] 2.1 Crear `docker/scraper.Dockerfile` — Python 3.11 + `pandas`, `pandera`, `numpy`, `requests`, `beautifulsoup4`, `boto3`, `pyarrow`, con el código de `pipeline/` copiado y `ENTRYPOINT ["python", "entrypoint.py"]`

## 3. Escritura de Terraform

- [ ] 3.1 Crear `infra/ecr-scraper.tf` — repo ECR `sinca-scraper`
- [ ] 3.2 Crear `infra/iam-scraper.tf` — rol `sinca-scraper-task-role` (S3 `sinca-data/*` + logs) y rol de ejecución (pull ECR + logs)
- [ ] 3.3 Crear `infra/fargate-scraper.tf` — task definition Fargate (imagen ECR, default VPC, CPU/memoria `512`/`1024`)
- [ ] 3.4 Crear `infra/schedule.tf` — regla EventBridge Schedule (cron ~01:00 America/Santiago) que dispara `ecs:RunTask`
- [ ] 3.5 Actualizar `infra/outputs.tf` con los ARNs de ECR, roles y schedule

## 4. Documentación

- [ ] 4.1 Actualizar `README.md` (estado: Fase 2 en progreso; cómo correr el backfill y el incremental; estructura de `pipeline/`)
- [ ] 4.2 Actualizar `project-decisions.md` §3 (cómputo del scraper: Fargate on-demand) y §7 (horario del schedule)

## 5. Ejecución real (operador corre y pega output)

- [ ] 5.1 Build de la imagen: `docker build -f docker/scraper.Dockerfile -t sinca-scraper .` y push a ECR
- [ ] 5.2 `terraform init` (si hace falta) y `terraform plan` — revisar el plan con el operador antes de aplicar
- [ ] 5.3 `terraform apply` (crea ECR, task Fargate, IAM, schedule)
- [ ] 5.4 Backfill manual de 5 años (2021-01-01 → hoy) ejecutando la task con el rango de fechas

## 6. Verificación funcional

- [ ] 6.1 Smoke test manual: correr el pipeline de punta a punta (incremental) y confirmar que los datos validados aparecen en `sinca-data/validated/` (listar con `aws s3 ls`)
- [ ] 6.2 Confirmar que los datos rechazados (si los hay) caen en `quarantine/` con motivo de rechazo
- [ ] 6.3 Activar el schedule y confirmar la primera corrida automática (o verificar que la regla existe y dispara la task)
