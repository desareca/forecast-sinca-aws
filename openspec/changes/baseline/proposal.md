## Why

Fase 3 del plan. El pipeline de datos (Fase 2) ya produce las 6 series horarias validadas en `sinca-data/validated/`. Falta el primer modelo: un **baseline** que establezca el piso a superar antes de los experimentos avanzados (Fase 4). Sin baseline no hay referencia para saber si el autoencoder + DMD/Koopman aporta algo real.

## What Changes

- **Persistencia baseline** (naive): predicción = último valor observado (promedio móvil 24h), sin entrenamiento. Referencia trivial.
- **LightGBM baseline**: 6 modelos independientes (MP2.5/MP10 × 3 estaciones), con lags + meteorología + Fourier, cada uno minimizando su RMSE ponderado.
- **Feature engineering**: construcción de features desde `validated/` (lags, meteorología, altura de capa límite, feriados, día de semana, ventana GEC, Fourier).
- **Evaluación**: ventana deslizante de tamaño fijo (train 2 años / valid 1 mes / step 1 mes) + test final 3 meses held-out.
- **Tracking MLflow**: logging directo (modo 1), convención `sinca-{estacion}-{modelo}-v{n}`.
- **Cómputo**: SageMaker Training Job (CPU `ml.m5.large`), imagen de entrenamiento en ECR.
- **IAM**: finalizar `sinca-training-role` (hoy skeleton).

**No incluye** (Fases 4+): autoencoder + DMD/Koopman, SageMaker Pipeline de entrenamiento, features cruzadas entre estaciones, ni inferencia productivizada.

## Capabilities

### New Capabilities

- `modelado`: feature engineering, entrenamiento de modelos, evaluación y baseline de persistencia sobre los datos validados.

### Modified Capabilities

- `iam`: se finaliza el rol de training (`sinca-training-role`), hoy skeleton, con acceso de lectura a `sinca-data/validated/*` y escritura a `sinca-mlflow/_mlflow/*` y `sinca-mlflow/model-artifacts/*`.
- `tracking-experimentos`: se agrega el modo de logging directo (sin servidor) usado por el Training Job, además del servidor interactivo ya especificado.

## Impact

- **Infraestructura AWS**: repo ECR para la imagen de entrenamiento, finalización del rol `sinca-training-role`. Región `us-east-1`. Training Job on-demand (no queda nada corriendo).
- **Repo**: `pipeline/` ampliado (features, train, evaluate, baseline), `docker/` ampliado (imagen de entrenamiento), `infra/` ampliado (ECR + IAM).
- **Dependencia previa**: `sinca-data/validated/` poblado (backfill Fase 2) y MLflow self-hosted operativo.
- **Costo mensual**: Training Job on-demand (corridas cortas, ~$0.05-0.10 por corrida), ECR/S3 despreciable. Total < USD 5/mes.
