# Proposal: `dmd-operador-global` (Fase 4, parte 2, change 7.1)

## Why

El DMD global sobre el latente pierde en RMSE ponderado contra LightGBM y persistencia en `sel`
(`EXPERIMENT-LOG.md`, 2026-11-15, v2: cumple 2/6 series). `ae-dmd-prep` dejó la partición, los AE
`version = 2` y las referencias en `sel`; falta saber si el techo es la representación o la
dinámica, y cuál es el mejor operador lineal global. Plan vinculante: `project-decisions.md` §9
("Plan de `ae-dmd`", 7.1); razonamiento: `docs/ae-dmd-propuesta-experimentos.md` §5, §8 y §9.

## What Changes

- **Paquete `pipeline/operador/`** (post-hoc, solo CPU local): una sola cadena estado → K →
  rollout → decode → promedio 24 h → métrica, común a Hankel, PCA y AE, sobre latentes ya
  entrenados (`sinca-zona-ae_{arq}-aedmd-v2-s{seed}`, conv1d y GRU × 3 semillas) o sobre las series
  crudas. Sin Training Jobs.
- **Métricas en `sel`** sobre las mismas filas que LightGBM v2 y persistencia v2: RMSE ponderado por
  serie, skill vs persistencia, intervalos por bootstrap de bloques de días; diagnósticos
  obligatorios (horizonte 1…24, episodios ≥ ICAP100, espectro, pendiente y sesgo por tramo).
- **Experimentos con compuertas internas**, cada una con pausa y decisión del operador:
  oráculo (1.0) → escalera de representaciones (1.1a–g, salida ≤ 2) → picos (1.2) → exógenas (1.3)
  → horizonte (1.4, K horario y A₂₄ de referencia; multi-paso diferido a `deep-koopman`) → estimador y bootstrap por modo (1.5) → rango y modos (1.6).
- **Cierre**: reajuste del ganador con datos hasta 2026-06 y una única evaluación en test
  (2026-07-01 → 2026-10-04), veredicto único del criterio de éxito (referencia = la mejor de
  LightGBM y persistencia por serie; `design.md` D12) y entregables de interpretabilidad.
- **Runs MLflow** del operador con semilla y versión en el nombre, escritos en un estado local y
  publicados con un script simple (respaldo con fecha y hora, control de `LastModified`).

## Capabilities

### New Capabilities

- `operador-dmd`: operador lineal global post-hoc sobre representaciones (Hankel, PCA, AE):
  evaluación común, información permitida, experimentos de la escalera, estimadores, selección de
  modos, compuertas, confirmación en test e interpretabilidad.

### Modified Capabilities

- `tracking-experimentos`: convención de runs del operador (nombres, tags y artefactos en el estado local).

## Impact

- Código nuevo: `pipeline/operador/` (estado, estimadores, rollout, evaluación, modos, CLI, README),
  `scripts/mlflow/publish_state.py`, `tests/operador/`, notebook de interpretabilidad.
  Reutiliza `ae/{data,dmd,models}.py`, `features/partition.py`, `evaluate/evaluate.py`,
  `analysis/{load,sel_summary}.py`, `train/mlflow_log.py` sin cambiar su comportamiento.
- AWS: solo lectura de `sinca-data/validated/` y `sinca-mlflow/_mlflow/`; escritura del sqlite y
  `mlruns/` nuevos por el operador. Sin IAM, red, imágenes ni Terraform nuevos. Costo ≈ USD 0
  (solo CPU local).
- Dependencias: `pydmd` y `scipy` ya presentes (`requirements-dmd.txt`, `.venv`).
