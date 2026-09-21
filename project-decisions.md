# Predicción Calidad del Aire (SINCA) — Plan de arquitectura (contexto para OpenSpec)

**Fecha:** 2026-09-21
**Estado:** Planificación — Fase 1 (Setup) en progreso

Este documento es la fuente de verdad de arquitectura del proyecto. Las
decisiones acá cerradas no se re-discuten salvo que cambien explícitamente, y
`AGENTS.md` obliga a leer este documento antes de generar cualquier artefacto
de OpenSpec.

---

## 1. Contexto y objetivo

Predicción de calidad del aire (MP2.5/MP10) en la zona saturada de
Temuco/Padre Las Casas (La Araucanía), horizonte **24 horas** (mismo que el
sistema oficial del MMA, para comparar contra ese benchmark implícito).

Target derivado **ICAP_zona**: máximo entre las 3 estaciones representativas
(Ñielol, Las Encinas, Padre Las Casas II), calculado por MP10 y MP2.5 con la
fórmula oficial y tomando el peor contaminante. El pipeline produce **6 series
continuas** (MP2.5 y MP10 × 3 estaciones) y el `ICAP_zona` se deriva siempre en
post-proceso. Detalle completo en `prompt-sinca-mlops.md`.

## 2. Cuentas AWS y entorno de desarrollo

Ver `PREREQUISITES.md`. Cuenta personal (Account ID `AWS_ACCOUNT_ID`), profile
`AWS_PROFILE` (nunca `default` ni un profile ajeno), región **`us-east-1`**.
Dev environment remoto según `modules/dev-environment/blueprint.md`.

## 3. Blueprints de dominio seleccionados

| Blueprint | ¿Aplica? | Parámetros de este proyecto |
|---|---|---|
| `compute` | Sí | Entrenamiento en SageMaker Training Job (CPU `ml.m5.large` tabular / GPU `ml.g4dn.xlarge` AE+DMD). Cómputo del scraper **pendiente** (decidir en Fase 2 tras analizar páginas SINCA). |
| `dev-environment` | Sí | SageMaker Studio Space `ml.t3.large`, EFS persistente, SSH over SSM, lifecycle config reinstala Node+OpenCode+aws+terraform+repo. |
| `data-quality` | Sí | pandera; convención `raw/`/`validated/`/`quarantine/`; dimensiones y umbrales se definen en Fase 2. |
| `tracking-experimentos` | Sí | MLflow self-hosted (SQLite+S3), prefijo `sinca-mlflow/_mlflow/`; logging directo sin servidor como modo principal. |
| `costos` | Sí | Checklist "¿qué queda corriendo?" por servicio; estimación mensual < USD 5 (Space apagable). |
| `cicd` | Sí | Terraform como código, remote state S3+DynamoDB, feature branches por change, OIDC GitHub Actions. |

### Naturaleza del proyecto

- **Tipo**: MLOps completo (ingesta → validación → entrenamiento → evaluación → productivización).
- **Stack ML**: tabular (LightGBM, scikit-learn) + red neuronal (autoencoder + DMD/Koopman, PyTorch/TF). Enfoque loss detallado en `prompt-sinca-mlops.md` (RMSE ponderado por serie; loss conjunta solo como variante experimental de AE+DMD).

## 4. Storage y datos

- Bucket **`sinca-data`**: `raw/` → `validated/` → `quarantine/` (Parquet).
- Bucket **`sinca-mlflow`**: `_mlflow/` (tracking: `mlflow.sqlite` + `mlruns/`) y `model-artifacts/` (modelos versionados).
- Target del pipeline: 6 series horarias MP2.5/MP10 (una por estación). Partición train/valid/test a definir en Fase 2/3.

## 5. IAM — mínimo privilegio

| Rol | Permisos acotados |
|---|---|
| `sinca-dev-role` (dev env / Studio) | S3 `sinca-data/*` + `sinca-mlflow/_mlflow/*` (r/w), logs `/aws/sagemaker/*`, acciones Studio mínimas (presigned url, describe, apps) |
| `sinca-training-role` (training) | S3 read `sinca-data/validated/*`, r/w `sinca-mlflow/model-artifacts/*`, logs `/aws/sagemaker/*` |
| OIDC infra (GitHub → Terraform) | Pemitos `plan`/`apply` acotados a los recursos gestionados (Fase 1b) |
| OIDC pipeline (GitHub → modelo) | Actualizar definición de pipeline, escribir artefactos, registrar modelos (Fase 1b) |

Nunca policies `*FullAccess` ni `AdministratorAccess`. Roles OIDC en Fase 1b.

## 6. Monitoreo y observabilidad

- CloudWatch Alarms sobre fallos de Training Job / pipeline (Fases 3+).
- Métricas de modelo y de calidad de datos → MLflow self-hosted, por corrida.
- Alerta selectiva (solo condiciones que requieren acción humana).

## 7. Ejecución y schedule

Nada automático en Fase 1. A partir de Fase 2: pipeline de datos disparado por
EventBridge Schedule (horario a definir); inferencia diaria a 24h en Fase 6.

## 8. Estructura OpenSpec

1. `setup-infra-base` — buckets, SageMaker Studio Space, IAM base, remote state. **(activo)**
2. `setup-mlflow-ci-cd` — MLflow self-hosted (Fargate), OIDC + 3 workflows.
3. `data-pipeline` (Fase 2) — scraper SINCA/Open-Meteo/feriados + ICAP + pandera.
4. `baseline` (Fase 3) — persistencia + modelo simple (LightGBM lags).
5. `experimentos-avanzados` (Fase 4) — autoencoder + DMD/Koopman.
6. `evaluacion` (Fase 5) — comparar baseline + referencia pública.
7. `productivizacion` (Fase 6) — reentrenamiento semanal + inferencia diaria.

## 9. Decisiones cerradas

- Región `us-east-1`; cuenta `AWS_ACCOUNT_ID`; profile `AWS_PROFILE`.
- Repo `desareca/forecast-sinca-aws`; OIDC `sub` restringido a `refs/heads/main`.
- Ramas: feature branches por change (PR + plan/apply).
- Tracking y Model Registry: MLflow self-hosted (SQLite+S3); reabrir Registry en Fase 6.
- Space `ml.t3.large`, EFS, SSH over SSM.
- 6 series target (MP2.5/MP10 × 3 estaciones); ICAP_zona derivado en post-proceso.
- No Secrets Manager en Fase 1 (fuentes públicas HTTP).

Ver `WORKFLOW.md` para el protocolo de coordinación entre Arquitecto,
Implementador y operador humano.
