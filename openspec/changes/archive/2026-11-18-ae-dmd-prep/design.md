# Design: `ae-dmd-prep`

## Context

- Vinculante: `project-decisions.md` §9 "Plan de `ae-dmd`". Razonamiento: `docs/ae-dmd-propuesta-experimentos.md`
  (v5) §4 y §8. El borrador de `docs/referencia/ae-dmd-borrador/` no se continúa.
- `ae/run.py` usa hoy `holdout_spans` (train = todo antes del test) y, para early stopping, la última
  ventana del rolling (junio 2026), que además queda dentro del train.
- El pase DMD se encadena solo a la semilla 0 y solo si la compuerta aprueba; evalúa en test.
- `train/train.py` reentrena LightGBM con todo lo anterior al test; persistencia guarda solo métricas.
- `run_name_ae` no lleva semilla; `ref-mean`/`ref-pca-d16` se repiten entre `L`.
- `mlflow_log.download_state` y `analysis/load.ensure_state` bajan todo `_mlflow/` (~370 objetos).
- `list_runs` no filtra `lifecycle_stage`: un `delete_run` no saca el run del dashboard.
- AE de referencia (§9): conv1d `L = 24`, `d = 16` (`d_aux = 4`), hidden 64, dropout 0.1, lr 0.003,
  batch 128, `w_aux` 0.1, `aux_input` on, 60 epochs, patience 8. El GRU de 11.3 usó los mismos valores.

## Goals / Non-Goals

**Goals:** dejar en MLflow, con la partición `ae-dmd`, 6 AE (conv1d y GRU × 3 semillas) con sus
latentes, 6 pases DMD evaluados en `sel` y las referencias persistencia/LightGBM en `sel`, más las
herramientas (descarga selectiva, resumen, mantenimiento, `pydmd`) que usa 7.1.

**Non-Goals:** oráculo, Hankel, PCA nuevo, ponderación, exógenas, estimadores, LPV, Deep Koopman,
bootstrap, rolling. Cambios al dashboard más allá de no romperse. Re-búsqueda de hiperparámetros.

## Decisions

### D1. Partición con nombre, en `pipeline/features/partition.py`

`Partition(name, bounds)` con tramos `train`, `es`, `sel`, `test` como `[inicio, fin)` en fechas
locales (`America/Santiago`, igual que `features.build`):

| Tramo | `[inicio, fin)` |
|---|---|
| `train` | 2021-01-01 → 2026-03-01 |
| `es` | 2026-03-01 → 2026-04-01 |
| `sel` | 2026-04-01 → 2026-07-01 |
| `test` | 2026-07-01 → 2026-10-05 (= `TEST_END` + 1 día) |

- Vive en `features/` porque la usan AE y LightGBM. `holdout` no se modela como `Partition`: es el
  camino actual y queda intacto (`--partition holdout`, defecto).
- Ventanas AE: regla actual (la ventana completa dentro del tramo), con `aed.partition_spans`.
- **Embargo de 24 h** para filas con target (`target(t) = mean(MP(t+1…t+24))`): una fila de un
  tramo entra solo si `inicio ≤ t < fin − 24 h`. Así ningún target de train usa marzo, ninguno de
  `es` usa abril y ninguno de `sel` usa julio (test). Aplica a LightGBM (train, es), a persistencia
  y a las filas de pronóstico DMD en `sel`. Ver Pregunta 1.
- Alternativa descartada: reutilizar `make_rolling_windows`/`final_holdout_split` con parámetros.
  No expresa 4 tramos ni el embargo.

### D2. AE en la partición `ae-dmd` (`run_final_partition`)

Función nueva en `ae/run.py`; `run_final` (holdout) no cambia salvo el nombre (D4).

- Normalización con el train del tramo; early stopping con `es`; métricas de reconstrucción,
  referencias PCA(d)/promedio, compuerta, ablación de `z_aux` y diagnósticos del latente sobre `sel`.
- Test: solo latentes (`latent.parquet` con `split ∈ {train, es, sel, test}`). Sin métricas, sin
  filas en `reconstruction.parquet`, sin pronóstico. Ver Pregunta 4.
- Prefijos de métricas: `es_recon_`, `sel_recon_`, `ref_pca_sel_`, `ref_mean_sel_`,
  `sel_recon_rmse_mean`, `es_recon_rmse_mean`, `ablation_zaux_delta_rmse_sel`. Gate y tiempos con
  los mismos nombres que hoy.
- Un job por semilla (`--seeds N`): si un job falla, se repite solo esa semilla.

### D3. Pase DMD en la partición `ae-dmd` (`run_dmd_pass_partition`)

- Mismo estimador que hoy (`dmd.fit_dmd`, rango completo y truncado `d/2`), `K` ajustado con los
  latentes de train. Linealizabilidad en `es` y `sel`.
- Pronóstico `mean(MP(t+1…t+24))` decodificando `K^24 z_t` para `t` en `sel` con embargo (D1).
- Encadenado a **cada** semilla del job, **sin exigir la compuerta** (se registra igual). §9 pide
  DMD para todas las semillas y el GRU. Ver Pregunta 5. `--mode dmd --partition ae-dmd` repite el
  pase sin reentrenar, también sin compuerta.
- Runs por serie con el contrato de LightGBM (`predictions.parquet` con `split = sel`,
  `zone_icap.parquet`, `rolling_scores.parquet` vacío), métricas `sel_*`.
- En el run del AE: `dmd_es_latent_rmse_{1,24}`, `dmd_sel_latent_rmse_{1,24}`,
  `dmd_sel_trunc_latent_rmse_{1,24}`, `dmd_trunc_rank`, `dmd_operator.npz`, `dmd_operator_trunc.npz`
  y el contraste con `pydmd` (D8).

### D4. Nombres de run y tags

| Run | Nombre | Tags/params clave |
|---|---|---|
| AE holdout | `sinca-zona-ae_{arq}-v{n}-s{seed}` | como hoy |
| AE ae-dmd | `sinca-zona-ae_{arq}-aedmd-v{n}-s{seed}` | `stage=ae-dmd-prep`, `particion=ae-dmd`, `seed` |
| Referencias holdout | `ref-mean-L{L}-v{n}`, `ref-pca-d{d}-L{L}-v{n}` | como hoy |
| DMD por serie holdout | `sinca-{slug}-ae-{arq}-dmd-v{n}-s{seed}` | como hoy + `seed` |
| DMD por serie ae-dmd | `sinca-{slug}-{pollutant}-ae-{arq}-dmd-aedmd-v{n}-s{seed}` | `stage=ae-dmd-prep`, `particion=ae-dmd`, `seed`, `ae_run_id` |
| LightGBM ae-dmd | `sinca-{slug}-{pollutant}-lightgbm-aedmd-v{n}` | `modelo=lightgbm-aedmd`, `source_run_id`, `particion` |
| Persistencia ae-dmd | `sinca-zona-persistence-aedmd-v{n}` | param `tipo=referencia`, tag `modelo=persistence-aedmd` |

- Token de partición = nombre sin guiones (`ae-dmd` → `aedmd`).
- En la partición con nombre, los runs por serie llevan el contaminante (corregido tras el humo de
  11.4: `pm10` y `pm25` de una estación compartían nombre). `holdout` conserva la convención del
  baseline `sinca-{estacion}-{modelo}-v{n}`.
- `modelo` distinto para LightGBM y persistencia de `ae-dmd`: el dashboard agrupa por
  `(modelo, version)` y no debe mezclarlos con el baseline `lightgbm v1`.
- `is_ae_run` pasa a reconocer el run conjunto por `modelo` (`ae-*` que no termina en `-dmd`, sin
  `estacion`), no por `stage`; `is_series_run` excluye con `is_ae_run`. Así los runs por serie con
  `stage=ae-dmd-prep` siguen siendo "por serie".
- El directorio local del modelo es `model_dir/<nombre>` (el nombre ya trae la semilla).

### D5. LightGBM y persistencia en `sel` (`train.py --mode refit`)

- Hiperparámetros: los del run `lightgbm-v1` de cada serie, leídos del sqlite (claves de
  `HYPERPARAM_SPACE`, con su tipo). El operador fija los 6 run IDs con `--source-run-ids`
  (`slug:pollutant=run_id`, separados por coma); el job falla si falta una serie o si
  `estacion/pollutant` del run fuente no coinciden. Motivo: hay 2 runs v1 por serie (13.1 de
  `ae-reconstruccion`); elegir a mano evita una regla implícita. Ver Pregunta 3.
- Ajuste: `fit_model(train con embargo, params, valid = es con embargo, seed=0)`; predicción en
  `sel` con embargo. Se registran `best_iteration`, métricas `sel_*`, `predictions.parquet`
  (`split = sel`), `feature_importance.parquet`, `zone_icap.parquet`, `rolling_scores.parquet`
  vacío y el booster. No se sube a `model-artifacts/`.
- Persistencia: `baseline.persistence.predict_frame` filtrado a `sel` con embargo, un run para las
  6 series con `predictions.parquet`.
- Mismo job, imagen `lgbm-v1` (sin cambios de dependencias).
- Sesgo conocido: los hiperparámetros v1 se eligieron con ventanas rolling cuyas valid llegan a
  2026-06 (`ROLLING_END`). Favorece a LightGBM en `sel`, lo que hace la referencia más exigente.
  Ver Pregunta 2.

### D6. Descarga selectiva

- `mlflow_log.download_state(..., run_ids=None)`: `None` = todo (actual); `[]` = solo
  `mlflow.sqlite`; lista = sqlite + artefactos de esos runs. Prefijo del run:
  `{prefix}/mlruns/{experiment_id}/{run_id}/` (experiment_id leído del sqlite recién bajado), con
  respaldo `{prefix}/mlruns/{run_id}/`. Omite archivos ya presentes con el mismo tamaño.
- `analysis.load.ensure_state(..., run_ids=None)`: con `run_ids=None` se comporta igual que hoy;
  con lista, baja (sqlite siempre, para refrescar) al `data_dir` dado o a un temporal.
- Jobs: hiperparámetro `--mlflow-runs` (`all` | `none` | IDs separados por coma), defecto `all`.
  `--mode dmd` agrega siempre `--ae-run-id`. Los jobs de este change usan `none`: solo agregan runs,
  y `upload_state` sube solo lo que hay en local (no borra nada en S3).
- `list_runs` agrega `start_time`, `end_time`, `lifecycle_stage` y por defecto omite los borrados.

### D7. Mantenimiento de MLflow (`scripts/mlflow/maintenance.py`, fuera de `pipeline/`)

- `plan` (solo lectura, abre el sqlite en modo `ro`): runs activos `lightgbm` con `version = 99`
  y prefijos S3 de sus artefactos; grupos de nombres duplicados entre runs `sinca-zona-ae_*` y
  `ref-*`, con nombre nuevo: `ref-mean` → `ref-mean-L{L}-v{n}`; `ref-pca-d{d}` →
  `ref-pca-d{d}-L{L}-v{n}`; `sinca-zona-ae_*` → `+ -s{seed}`; si sigue duplicado,
  `+ -d{d}-ai{0|1}-wa{w_aux}`; si aún sigue, `+ -r{run_id[:6]}`. Escribe `plan.json` e imprime una
  tabla. Si no hay nada que hacer, lo dice y termina con 0.
- `apply --plan plan.json --yes`: sobre una **copia local** del sqlite, `delete_run` y
  `update_run(name=…)`; imprime los comandos del operador (respaldo del sqlite en S3, `mlflow gc`,
  `aws s3 rm --dryrun` y real, subida del sqlite), igual que 13.1 de `ae-reconstruccion`.
- No se tocan los runs `lightgbm-v1` (2 por serie es estructural).

### D8. `pydmd` a prueba

- Imagen `sinca-train:tf-cpu-v2` desde `docker/train-tf-v2.Dockerfile`: `FROM` la imagen publicada
  `tf-cpu-v1` (`ARG BASE_IMAGE`, por digest o tag) más una capa `pip install "pydmd==2025.8.1"`. No se
  reconstruye desde `python:3.11-slim`: los rangos sin fijar de v1 resolverían versiones nuevas
  (medido en 11.2: mlflow 3.17.2 → 3.18.0, boto3, SQLAlchemy, …). Un mlflow distinto escribiendo el
  sqlite compartido puede migrar su esquema y dejar fuera a `lgbm-v1` y al dashboard.
- Aceptación de la imagen: `pip freeze` de v2 = el de v1 más líneas nuevas, sin cambiar versión de
  `tensorflow-cpu`, `keras`, `numpy`, `pandas`, `protobuf`, `h5py`, `scikit-learn`; `pip check`
  igual al de v1 (v1 ya reporta el conflicto conocido de `sagemaker-training` con `protobuf`).
- `dmd.pydmd_crosscheck(segmento, rank)`: ajusta `fit_dmd` (numpy) y `pydmd.DMD(svd_rank, exact=True)`
  sobre el mismo segmento contiguo más largo del latente de train; compara autovalores (distancia
  máxima al más cercano, en ambos sentidos) y, en rango completo, la predicción a 1 paso. Tolerancia
  `1e-6`. Import perezoso: si `pydmd` no está, se registra el tag `pydmd_version=no-disponible`.
  Una discrepancia no corta el job: se registra `dmd_pydmd_ok = 0` y un warning.
- Se compara en un segmento contiguo y no con todos los pares porque `pydmd` ajusta una trayectoria
  de snapshots; el `K` multi-segmento de producción no cambia.
- Plan B (si la imagen v2 no cumple): los jobs usan `tf-cpu-v1`; el contraste corre con
  `scripts/ae/pydmd_check.py` en un venv aparte (`requirements-dmd.txt`) sobre `latent.parquet`
  bajado con descarga selectiva.

### D9. Orquestación

`scripts/ae/run_ae_dmd_prep.ps1`, mismo patrón que `run_11_3_finales.ps1` (cuenta verificada desde
`.env`, serie, salta `Completed`, sufijo `-rN` tras fallo, se detiene en el primer fallo, `-DryRun`).
Switch `-Smoke`: un solo job conv1d semilla 0, `epochs = 2`, experimento `sinca-smoke`.
Parámetros `-TfTag` (defecto `tf-cpu-v2`) y `-SourceRunIds`.

### D11. Presupuesto fijo, sin early stopping (corrección tras 11.5, v2)

- Medido en v1: early stopping con marzo corta muy pronto (AE: mejor época 3–24, `gate_pass = 0` en
  los 6; LightGBM: 41–145 rondas, peor que persistencia en 4/6 series de `sel`). No hay error de
  máscaras ni fuga (diagnóstico de solo lectura en las notas de implementación).
- AE (`--early-stopping off`): `fit` sin `valid` → 60 épocas completas, sin restaurar pesos. Marzo
  se sigue reportando (`es_recon_*`) como diagnóstico. `holdout` no cambia (defecto `on`).
- LightGBM (`--rounds-from-source`): `num_boost_round` = `num_trees()` del booster `.txt` del run
  fuente: el **baseline** `sinca-{estacion}-lightgbm-v1` (el de la búsqueda, el que dio 28.17 en
  test), **no** el refit `lightgbm-aedmd-v1` (ese usó early stopping con marzo, justo lo que D11
  elimina). Sin conjunto de validación. Marzo no
  entra al train (se mantiene la partición).
- Alternativa descartada: mediana de rondas de las ventanas rolling previas a 2026-03 (sin fuga,
  pero el dato no se guardó por ventana). Early stopping con bloques del train: más código y riesgo.
- Se re-corren los 7 jobs como `version = 2`; los runs v1 quedan como evidencia del problema.
  `sel_summary.py` filtra por `--version`.

### D10. Resumen de `sel` (`pipeline/analysis/sel_summary.py`)

Baja con D6 solo los runs de `particion = ae-dmd`, toma los `predictions.parquet` (`split = sel`) de
persistencia, LightGBM y los 6 DMD, intersecta filas por serie (`timestamp`) entre **todos** y
reporta RMSE y RMSE ponderado por serie y su media, con `n` común; más una tabla de los 6 AE
(`gate_pass`, `sel_recon_rmse_mean`, `dmd_sel_latent_rmse_{1,24}`, `dmd_pydmd_ok`). Salida markdown
para `EXPERIMENT-LOG.md`. Sin bootstrap (7.1).

## Costo y tiempo

**v2 (D11):** 60 épocas fijas → conv1d ~15–18 min de fit y GRU ~38–45 min. Estimado de la serie
v2: ~3,3 h, ~USD 0,40. El real de v1 fue 4 637 s (USD 0,16).

Base: 11.3 y calibración 10.4 (conv1d `L = 24` ~15–16 min de fit en 60 epochs; GRU ~38 min), train
~6 % más corto, + ~5–8 min por job de arranque, datos y DMD. Tarifa `ml.m5.large` Training
on-demand us-east-1 ≈ USD 0.115/h (verificar en la consola).

| Job | n | Tiempo c/u | Total | Costo |
|---|---|---|---|---|
| Humo conv1d (2 epochs) | 1 | ~8 min | 0.13 h | ~0.02 |
| LightGBM + persistencia | 1 | ~10–15 min | 0.25 h | ~0.03 |
| conv1d s0–s2 | 3 | ~25 min | 1.25 h | ~0.15 |
| GRU s0–s2 | 3 | ~45 min | 2.25 h | ~0.26 |
| **Total** | 8 | — | **~3.9 h en serie** | **~USD 0.46** |

ECR: +~1 GB de imagen (< USD 0.10/mes). Con reintentos, el techo razonable es < USD 1.
`max_run`: humo y LightGBM 3600 s, conv1d 7200 s, GRU 10800 s.

## Risks / Trade-offs

- [`pydmd` cambia versiones del runtime] → criterio de freeze de D8; si falla, plan B.
- [GRU inestable entre semillas (11.3)] → se registra la dispersión; descartarlo es decisión de 7.1.
- [Upload con estado parcial] → `upload_state` solo sube archivos locales; los jobs corren en serie.
- [Renombre o borrado sobre el sqlite real] → plan de solo lectura, copia local, respaldo en S3 y
  ejecución del operador con output a la vista.
- [Runs nuevos sin métricas `test_`] → el dashboard muestra "sin predicciones en este split"
  (degradación ya especificada); vistas de `sel` quedan para 7.1.
- [Cifras distintas a 33.28 / 31.09 / 28.17] → esperado: otro corte y otro tramo; se anota en
  `EXPERIMENT-LOG.md`.

## Migration Plan

Código y tests (grupos 1–10) → mantenimiento MLflow (operador) → imagen `tf-cpu-v2` (operador) →
IDs fuente de LightGBM (operador confirma) → humo → LightGBM/persistencia → 6 AE → resumen y
registro. Rollback: los runs nuevos se identifican por `particion = ae-dmd` y se pueden borrar con el
mismo script; `tf-cpu-v1` y `lgbm-v1` no se tocan.

## Preguntas al operador

No cambian §9; las respuestas por defecto están ya en las decisiones.

1. **Embargo de 24 h** en el borde de cada tramo para filas con target (D1). §9 no lo dice; sin él,
   los targets de fines de junio en `sel` usan observaciones de julio (test). Defecto: aplicar.
2. **Hiperparámetros heredados de LightGBM v1** (D5): su búsqueda usó valid hasta 2026-06, que
   incluye `sel`. §9 pide no re-buscar. Defecto: aceptar y anotar el sesgo (a favor de LightGBM).
3. **Run fuente por serie** (D5): hay 2 runs `lightgbm-v1` por serie. Defecto: el que tiene
   `predictions.parquet` (el usado en 11.5); la tarea 11.3 imprime el mapeo para confirmar.
4. **Test en este change** (D2): se guardan latentes de test, sin métricas ni pronóstico. §9: el
   test solo confirma con el ganador reajustado hasta 2026-06. Defecto: así.
5. **Compuerta en `ae-dmd`** (D3): se calcula en `sel` y es informativa; el DMD corre en las 6
   corridas aunque falle. Contradice la regla de elegibilidad de `ae-reconstruccion` (spec
   modificada). Defecto: así.
6. **Runs v99**: la tarea 13.1 de `ae-reconstruccion` figura cerrada; §9 la repite. El plan de D7 lo
   comprueba; si hay 0, no se hace nada. ¿Quedó algo pendiente que no esté en el sqlite?
7. **Duplicados históricos**: ¿aplicar el renombre del plan o solo corregir hacia adelante?
   Defecto: el plan se genera; el operador decide al aplicar (`--skip-renames`).
