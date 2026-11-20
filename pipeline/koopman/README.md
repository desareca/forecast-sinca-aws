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
| `identificabilidad.py` | aportes, dominantes, bootstrap del latente, coherencia, compuerta, eliminación de bloques |
| `compuerta.py` | reglas → JSON de decisión (`rama`) en S3 |
| `run.py` | entry point de los jobs (`--fase …`) |

## Fases y compuertas

`smoke` → F1 (GRU, conv1d, GRU+decoder GRU, variantes de K) → compuerta → (B1 λ_lin → B2 consistencia →
B3 L1 → B4 `d = 8`) → F2 (currículo H 1/6/24) → F3 (+`L_tgt`, variante `icap`) → cierre (solo `K`, `B`,
train+es+sel) → ⏸ operador → test (una vez; `--force-test` para repetir).

La compuerta (`--fase compuerta --de <fase>`) evalúa los runs de la fase en la campaña y escribe
`model-artifacts/orquestacion/deep-koopman/<campaña>/<fase>.json` (`rama` ∈ `pasa`, `b1`–`b4`, `c`,
`pausa`, `continuar`; ganadores por semilla). Con `pasa` escribe además `arranque_f2.json` (padre de F2).

## Jobs, marcadores y reanudación

- Cada job publica sus runs con `train.mlflow_log.publish_new_runs` (chequeo de `LastModified`,
  respaldo, solo runs nuevos; conflicto → copia en `/opt/ml/model/conflicto` y error) y solo después
  escribe su marcador `<campaña>/jobs/<clave>.json`. La máquina salta los jobs con marcador.
- Checkpoint por época en `/opt/ml/checkpoints/<variante>/` (S3 vía `CheckpointConfig`): al reiniciar,
  retoma en la época siguiente sin reiniciar lr, calentamiento ni currículo.

## Modo seco

`--dry`: sin MLflow ni entrenamiento; la compuerta escribe `--rama-forzada`. `--dry-ckpt`: 4 épocas
sintéticas con checkpoints; `--dry-fallar-en N`: falla tras guardar la época N.

## Comandos

```powershell
# entrada de la campaña (sin AWS) e impresión del start-execution
.\.venv\Scripts\python scripts/koopman/preparar_ejecucion.py --modo real --campana <id> --plan
# humo manual (tarea 8.1)
.\.venv\Scripts\python pipeline/train/launch.py --profile <AWS_PROFILE> --role-arn <training_role_arn> --image-uri <registro>/sinca-train:tf-cpu-v2 --entry-point koopman/run.py --hp fase=smoke --hp experiment=sinca-smoke --hp campana=humo --max-run 3600 --wait
# publicación manual tras un conflicto
.\.venv\Scripts\python scripts/mlflow/publish_state.py apply --yes --stage deep-koopman --state-dir <dir> --profile <AWS_PROFILE>
```

Runs: `sinca-zona-koop-{enc}-{fase}{var}-aedmd-v{n}-s{seed}` (tags `stage=deep-koopman`, `fase`,
`variante`, `seed`, `campana`, `parent_run_id`); test por serie `sinca-{slug}-{pollutant}-koop-{enc}-aedmd-v{n}-s{seed}`.
