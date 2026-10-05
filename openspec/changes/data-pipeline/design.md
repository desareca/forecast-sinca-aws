## Context

Proyecto `forecast-sinca-aws` (predicción ICAP zona saturada Temuco/PLC, horizonte 24h). Cuenta AWS personal (`AWS_ACCOUNT_ID`, profile `AWS_PROFILE`, región `us-east-1`). Repo GitHub `desareca/forecast-sinca-aws`.

Fase 1 (Setup) cerrada: buckets `sinca-data`/`sinca-mlflow`, SageMaker Studio Space, MLflow self-hosted (Fargate+ECR), OIDC + 3 workflows. Este change (Fase 2) construye el pipeline de datos que alimentará el modelado.

Decisiones ya cerradas en `project-decisions.md`: 6 series target (MP2.5/MP10 × 3 estaciones), ICAP_zona derivado en post-proceso, tracking MLflow self-hosted, región `us-east-1`. El cómputo del scraper estaba **pendiente** y se resuelve acá (Fargate on-demand, confirmado con el operador).

## Goals / Non-Goals

**Goals:**
- Ingerir de forma 100% automática las 3 estaciones SINCA (MP2.5/MP10/gases/meteorología) + Open-Meteo (altura de capa límite) + feriados.
- Calcular el ICAP por estación (fórmula oficial D.S. 12/2011) y derivar `ICAP_zona` en post-proceso.
- Validar con pandera (schema/rangos) antes de persistir, con cuarentena para lo rechazado.
- Persistir en Parquet bajo `raw/` → `validated/` → `quarantine/`.
- Backfill inicial de 5 años (2021-01-01 → hoy) + modo incremental diario.
- Schedule EventBridge diario (~01:00 hora local).

**Non-Goals:**
- Entrenamiento de modelos (Fase 3+).
- SageMaker Pipeline de entrenamiento (Fase 3+; `pipeline-cd.yml` sigue skeleton).
- Scraping de comunicados oficiales del MMA (el estado se deriva, no se scrapea).
- Calendario escolar Mineduc (descartado, no automatizable de forma confiable).

## Decisions

### 1. Cómputo del scraper: Fargate on-demand

El scraper corre como task Fargate on-demand (no Lambda). Motivo: (a) el backfill inicial de 5 años puede superar el límite de 15 min de Lambda; (b) las dependencias (`pandas` + `pandera` + `numpy` + `requests` + `bs4`) son pesadas para layers de Lambda; (c) reutiliza el patrón ECR+Fargate ya montado en Fase 1.

- Imagen ECR `sinca-scraper`: Python 3.11 + `pandas`, `pandera`, `numpy`, `requests`, `beautifulsoup4`, `boto3`, `pyarrow`.
- El código del pipeline vive en `pipeline/` y se empaqueta en la imagen (rebuild al cambiar el código). Alternativa considerada: "entrypoint genérico + código en S3" (patrón `compute`) → descartada por ahora, agrega coordinación S3 sin beneficio claro para un solo script orquestador.
- Task definition Fargate (no service), reutiliza el default VPC, CPU/memoria `512`/`1024` (a ajustar si el backfill lo exige).

### 2. Backfill inicial de 5 años

Backfill `2021-01-01` → hoy, más modo incremental diario (últimas 24-48h). Motivo del operador: medidas de mitigación de emisiones a lo largo de los años cambian la dinámica, y un histórico muy largo introduce más ruido que señal; además se busca un flujo más simple. El scraper soporta ambos modos vía parámetro de rango de fechas.

### 3. Schedule EventBridge diario

Regla de EventBridge Schedule (cron) que dispara `ecs:RunTask` sobre la task del scraper a las **~01:00 hora local** (America/Santiago), para capturar el día anterior completo (SINCA actualiza cada hora). El horario exacto se define en la implementación; el smoke test manual precede a la activación del schedule.

### 4. Validación pandera (5 dimensiones)

Se validan 5 de las 6 dimensiones del blueprint `data-quality` (la "exactitud" contra fuente externa no aplica: no hay fuente de referencia independiente):

- **Validez**: tipos correctos; rangos plausibles (MP2.5/MP10 ≥ 0 y ≤ 1000 µg/m³; humedad 0-100%; etc.).
- **Completitud**: % de nulos por columna; umbral crítico en MP2.5/MP10 (< 5% nulos en la ventana ingerida).
- **Unicidad**: sin duplicados `(estación, timestamp)`.
- **Oportunidad**: el último timestamp ingerido está dentro de las últimas N horas (N=48 para tolerar retrasos de validación operacional del SINCA).
- **Consistencia**: unidades y encoding uniformes (µg/m³, UTC o hora local consistente).

Lo que aprueba pasa a `validated/`; lo que falla va a `quarantine/` con el motivo de rechazo adjunto (dimensión + regla que falló). El umbral de detención (tasa de cuarentena) se define en la implementación; si la calidad cae por debajo, el pipeline no persiste `validated/` para esa corrida.

### 5. Estructura del código en `pipeline/`

```
pipeline/
├── entrypoint.py        # orquestador: scraper → icap → validate → persist
├── scraper/
│   ├── sinca.py         # scraping CSV de apub.tsindico2.cgi (3 estaciones, macropath/macro descubierto desde la página de la estación)
│   ├── open_meteo.py    # altura de capa límite (JSON)
│   └── feriados.py      # Nager.Date API (date.nager.at/api/v3/PublicHolidays/{year}/CL)
├── icap/
│   └── icap.py          # fórmula piecewise-linear D.S. 12/2011 (3 anclas)
├── validate/
│   └── schemas.py       # pandera DataFrameSchema/SchemaModel
└── persist/
    └── s3.py            # escritura Parquet raw/ → validated/ → quarantine/
```

### 6. Cálculo de ICAP

El ICAP se calcula con la fórmula oficial del D.S. 12/2011 (y su equivalente MP2.5), aplicada al promedio móvil de 24h de MP10/MP2.5. La fórmula es piecewise-lineal con **3 anclas por contaminante** (interpolación lineal entre anclas consecutivas), verificadas contra la fuente oficial (SESMA/MMA):

| Contaminante | ICAP 0 | ICAP 100 | ICAP 500 |
|---|---|---|---|
| MP10 | 0 µg/m³ | 150 µg/m³ | 330 µg/m³ |
| MP2.5 | 0 µg/m³ | 50 µg/m³ | 170 µg/m³ |

Los niveles intermedios (alerta/preemergencia) caen exactamente sobre la recta 100→500 (MP10: ICAP 200→195, ICAP 300→240; MP2.5: ICAP 200→80, ICAP 300→110), por lo que 3 anclas bastan y son equivalentes a la tabla completa de 5 tramos. El `ICAP_zona` se deriva siempre en post-proceso como `max` entre las 3 estaciones y el peor contaminante (no se persiste como serie, se calcula al consumir).

### 7. Modelo de datos y particionado

- **Series target**: 6 series horarias (MP2.5 y MP10 × 3 estaciones) + meteorología por estación + altura de capa límite + feriados + día de semana + ventana GEC (1 abr–15 sep).
- **Particionado Parquet**: por dataset y estación, con partición temporal (año/mes) para lecturas eficientes. Estructura exacta en `sinca-data/{raw,validated,quarantine}/<dataset>/...`.

### 8. IAM del scraper

Rol de ejecución de la task Fargate (`sinca-scraper-task-role`): S3 `sinca-data/*` (GetObject/PutObject/ListBucket) + `logs` en su log group propio. Sin permisos IAM ni managed policies amplias. (El acceso a Open-Meteo/feriados es HTTP público, no requiere IAM.)

## Risks / Trade-offs

- **Scraping frágil (sin API estable)** → mitigación: el scraper se parametriza por ID de estación y descarga el CSV (`apub.tsindico2.cgi`) descubriendo `macropath`/`macro` desde la página de la estación; si SINCA cambia la estructura, el cambio se detecta en la validación (oportunidad/completitud) y se corrige como change nuevo.
- **Backfill de 5 años puede tardar** → mitigación: Fargate sin límite de 15 min; si excede la memoria, se sube CPU/memoria (variables).
- **Datos operacionalmente no validados por el SINCA** (los registros "pueden variar una vez validados") → mitigación: el modo incremental re-ingiere una ventana solapada (últimas 48h) para corregir retroactivamente.
- **Schedule dispara corridas que fallan silenciosamente** → mitigación: CloudWatch Alarms sobre fallos de la task (Fase 3+, según `project-decisions.md` §6); por ahora el smoke test manual y los logs de la task.
- **Concurrencia de escritura en S3** → mitigación: una sola corrida diaria; el backfill es one-shot antes de activar el schedule.

## Migration Plan

1. Escribir el código del pipeline (`pipeline/`) y la imagen (`docker/`).
2. Escribir Terraform (ECR, task Fargate, IAM, schedule).
3. Build + push de la imagen; `terraform plan`/`apply`.
4. Backfill manual (5 años) y smoke test manual de punta a punta.
5. Activar el schedule y verificar la primera corrida automática.
6. Rollback: `terraform destroy` de los recursos nuevos (no toca buckets ni Fase 1).

## Open Questions

- Ninguna pendiente. Cómputo (Fargate), backfill (5 años), schedule (~01:00 local), validación (5 dimensiones) y estructura (`pipeline/`) quedaron cerrados con el operador.
