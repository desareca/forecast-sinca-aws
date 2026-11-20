# `pipeline/koopman/` — Koopman entrenado (`deep-koopman`)

Encoder (GRU o conv1d somero) → `z` (`d = 10`) → `K` por bloques (diario 24 h fijo, 2 libres, 4 reales,
`ρ = exp(−(δ + softplus(a)))`, vida media ≤ 1000 h) + `B·u` (calendario de `t+1`) → decoder lineal
(o GRU). Rollout de 24 pasos sin meteo futura → promedio 24 h → µg/m³. Todo el cómputo de
experimentos corre en SageMaker Training Jobs orquestados por Step Functions; local solo tests.

| Módulo | Qué hace |
|---|---|
| `modelo.py` | redes, `OperadorBloques` (variantes `kden`, `perlib`, `sinbu`), entradas, `pronostico` |
| `perdidas.py` | `L_rec`, `L_lin`, `L_pred(H)`, `L_tgt` (y `icap` con máximo suave), `L_norma` (escala por bloque del decoder lineal), `L_cons`, `L1` |
| `entrenar.py` | `Datos`, fases y presupuesto fijo, calentamiento, currículo, paso compilado (`tf.function`, un trazado por fase), checkpoints, orden canónico |
| `inicio.py` | arranque del GRU desde el AE v2, decoder lineal por LS y reescalado por bloque, `K` desde `e15a` |
| `evaluar.py` | arnés de `operador` (filas fijas de 7.1, bootstrap, Duan, diagnósticos) |
| `identificabilidad.py` | aportes, dominantes, bootstrap del latente, coherencia, criterio de identificabilidad (D11), eliminación de bloques |
| `seleccion.py` | selección preregistrada (D14) → `seleccion.json` en S3 |
| `run.py` | entry point de los jobs (`--fase …`) |

## Campaña (lista fija, sin ramas ni pausas)

`smoke` (manual) · campaña en serie: F1 (GRU, conv1d, GRU+decoder GRU, variantes de K) → B1–B4 (siempre)
→ F2 (currículo H 1/6/24; GRU y conv1d) → F3 (+`L_tgt`, y variante `icap`; GRU y conv1d) → selección →
cierre (solo `K`, `B`, train+es+sel) → test (una vez; `--force-test` para repetir). 48 jobs.

- F2/F3 toman su padre del marcador del job de F1/F2 con el mismo encoder y semilla (`--parent-from`).
- `--fase seleccion` aplica D14 a los runs de la campaña (candidatas, orden de simplicidad, filtro de
  identificabilidad, reemplazo solo con mejora significativa; controles y `decgru` se reportan), registra
  `sinca-zona-koop-seleccion-aedmd-v{n}` y, tras publicar, escribe `<campaña>/seleccion.json`.
- `--fase cierre` y `--fase test` leen `seleccion.json` (`--seleccion` para una ruta local) y fallan sin
  ella; escriben `<campaña>/cierre.json` y `test.json` (con `resumen` para el aviso SNS).

## Jobs, marcadores y reanudación

- Cada job publica sus runs con `train.mlflow_log.publish_new_runs` (chequeo de `LastModified`,
  respaldo, solo runs nuevos; conflicto → copia en `/opt/ml/model/conflicto` y error) y solo después
  escribe su marcador `<campaña>/jobs/<clave>.json`. La máquina salta los jobs con marcador.
- Checkpoint por época en `/opt/ml/checkpoints/<variante>/` (S3 vía `CheckpointConfig`): al reiniciar,
  retoma en la época siguiente sin reiniciar lr, calentamiento ni currículo.

## Modo seco

`--dry`: sin MLflow ni entrenamiento; selección, cierre y test escriben JSON sintéticos (cierre y test
exigen la selección). `--dry-ckpt`: 4 épocas sintéticas con checkpoints; `--dry-fallar-en N`: falla tras
guardar la época N. `preparar_ejecucion.py --modo seco` usa la semilla 0 (18 jobs).

## Comandos

```powershell
# entrada de la campaña (sin AWS) e impresión del start-execution
.\.venv\Scripts\python scripts/koopman/preparar_ejecucion.py --modo real --campana <id> --plan   # seco: --modo seco --dry-ckpt --fallar-en f1:3:2
# humo manual (tarea 8.1)
.\.venv\Scripts\python pipeline/train/launch.py --profile <AWS_PROFILE> --role-arn <training_role_arn> --image-uri <registro>/sinca-train:tf-cpu-v2 --entry-point koopman/run.py --hp fase=smoke --hp experiment=sinca-smoke --hp campana=humo --max-run 3600 --wait
# publicación manual tras un conflicto
.\.venv\Scripts\python scripts/mlflow/publish_state.py apply --yes --stage deep-koopman --state-dir <dir> --profile <AWS_PROFILE>
```

Runs: `sinca-zona-koop-{enc}-{fase}{var}-aedmd-v{n}-s{seed}` (tags `stage=deep-koopman`, `fase`,
`variante`, `seed`, `campana`, `parent_run_id`); test por serie `sinca-{slug}-{pollutant}-koop-{enc}-aedmd-v{n}-s{seed}`.
