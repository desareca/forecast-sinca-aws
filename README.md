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
infra/        # Terraform (IAM, SageMaker Studio, buckets, red)
pipeline/     # scripts .py del pipeline ML (futuro)
tests/        # unit + integration (futuro)
notebooks/    # solo exploración (futuro)
modules/      # blueprints de decisión por dominio (guías, no specs)
openspec/     # changes activos y specs consolidadas
```

## Estado actual

**Fase 1 — Setup (en progreso).** Infraestructura base **completada**
(`setup-infra-base`): buckets S3 (`sinca-data`, `sinca-mlflow`), entorno de
desarrollo remoto (SageMaker Studio Space sobre EFS), y roles IAM de base
(`sinca-dev-role`, `sinca-training-role`). Próximo paso: `setup-mlflow-ci-cd`
(MLflow self-hosted + OIDC + workflows de GitHub Actions). Aún sin pipeline de
datos ni modelos.

## Cuenta y credenciales

Cuenta personal (Account ID `AWS_ACCOUNT_ID`), profile `AWS_PROFILE` (nunca
`default` ni un profile ajeno). Región `us-east-1`. Ver `PREREQUISITES.md`.

## Flujo de trabajo

Coordinación entre los agentes `arquitecto` (Arquitecto) y `implementador` (Implementador)
con el operador humano. Ver `WORKFLOW.md` y `AGENTS.md`.
