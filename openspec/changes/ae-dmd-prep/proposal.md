# Proposal: `ae-dmd-prep` (Fase 4, parte 2, change 7.0)

## Why

`dmd-operador-global` (7.1) debe elegir representación y operador en una validación que ningún
modelo usó para entrenar ni para detenerse (`project-decisions.md` §9, "Partición de `ae-dmd`").
Hoy no existe esa partición, persistencia no guarda predicciones, los nombres de run se repiten
y `ensure_state` baja todo el estado de MLflow (~355 MiB) en cada arranque.

## What Changes

- **Partición `ae-dmd`** con nombre y tramos: train 2021-01 → 2026-02, early stopping 2026-03,
  validación de selección (`sel`) 2026-04 → 2026-06, test 2026-07-01 → 2026-10-04. La partición
  actual (`holdout`) queda igual y sigue siendo la de defecto.
- **Reentrenamiento** de conv1d y GRU `L = 24` con la configuración del AE de referencia, 3
  semillas cada uno (6 Training Jobs, uno por semilla, CPU `ml.m5.large`). Se guardan latentes de
  los 4 tramos. No se calculan métricas ni predicciones de test.
- **Pase DMD global** (DMD exacto numpy, el actual) encadenado a cada una de las 6 corridas, con
  `K` ajustado en train y pronóstico en `sel`. Es la fila base del método (continuidad con 33.28).
- **Referencias en `sel`**, mismas filas: `predictions.parquet` de persistencia y de LightGBM.
  LightGBM se reajusta con el mismo corte (train hasta 2026-02, early stopping 2026-03) usando
  los hiperparámetros de los runs `lightgbm-v1`, sin nueva búsqueda.
- **Nombres de run**: semilla (`-s{seed}`) y token de partición en los runs nuevos; referencias
  con `L`. Plan opcional para renombrar los duplicados históricos (`ref-mean`, `ae_conv1d-v2`).
- **Descarga selectiva**: `ensure_state` y los Training Jobs pueden bajar solo `mlflow.sqlite` o
  el sqlite más los artefactos de los runs pedidos.
- **Mantenimiento de MLflow**: script con plan de solo lectura (runs `lightgbm` v99, renombres) y
  aplicación sobre copia local; la ejecución contra S3 es del operador.
- **`pydmd` a prueba**: imagen nueva `tf-cpu-v2` (= `tf-cpu-v1` + `pydmd`); cada pase DMD
  contrasta numpy contra `pydmd` sobre el latente. Plan B: entorno aparte que lee latentes.
- **Resumen de `sel`** sobre filas comunes y entrada en `EXPERIMENT-LOG.md` (las cifras de
  referencia cambian por la partición; no es regresión).
- **BREAKING (solo nombres de runs nuevos)**: en `holdout`, los AE pasan a
  `sinca-zona-ae_{arq}-v{n}-s{seed}` y las referencias a `ref-mean-L{L}-v{n}` /
  `ref-pca-d{d}-L{L}-v{n}`. Los runs existentes no cambian salvo que el operador aplique el plan.

## Capabilities

### New Capabilities

Ninguna.

### Modified Capabilities

- `modelado-ae`: partición `ae-dmd`, reentrenamiento por semilla, pase DMD sin compuerta en esa
  partición, referencias de pronóstico en `sel`, contraste numpy/`pydmd`.
- `tracking-experimentos`: convención de nombres con semilla y partición, sincronización parcial
  del estado en los jobs, mantenimiento del estado.
- `analisis-modelos`: descarga selectiva en `ensure_state`, exclusión de runs borrados, resumen de
  la validación de selección.

## Impact

- Código: `pipeline/features/partition.py` (nuevo), `pipeline/ae/{data,run,dmd}.py`,
  `pipeline/train/{train,mlflow_log}.py`, `pipeline/analysis/load.py`,
  `pipeline/analysis/sel_summary.py` (nuevo), `scripts/mlflow/maintenance.py` (nuevo),
  `scripts/ae/run_ae_dmd_prep.ps1` (nuevo), `scripts/ae/pydmd_check.py` (nuevo),
  `docker/train-tf-v2.Dockerfile` (nuevo), `requirements-dmd.txt` (nuevo, solo plan B), tests.
- AWS: un tag ECR nuevo (`sinca-train:tf-cpu-v2`), 8 Training Jobs (1 humo, 1 LightGBM, 6 AE),
  escritura serializada en `sinca-mlflow/_mlflow/`. Sin IAM, red ni Terraform nuevos.
- Costo estimado: < USD 1 (detalle en `design.md`).
