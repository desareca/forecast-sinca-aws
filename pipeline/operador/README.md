# `pipeline/operador/` — operador lineal global (`dmd-operador-global`)

Post-hoc, solo CPU local (`.venv`). Una cadena común: estado → K → rollout 24 h → decodificación →
promedio 24 h → RMSE ponderado en `sel` sobre `eval_rows` (D3), con bootstrap de bloques de 7 días
(B = 2000, semilla 0, emparejado). Sin Training Jobs.

| Módulo | Qué hace |
|---|---|
| `estado.py` | `Contexto` (normalizador solo con los tramos de ajuste) y estados `hankel`, `hankelmet`, `hankel48`, `hankelest` (bloques), `pca`, `ae` |
| `estimadores.py` | `exacto` (envuelve `ae.dmd.fit_dmd`), `truncado`, Gavish–Donoho, `ponderado`, `directo24`, `recorte`, `fbdmd`, `por_bloques` |
| `rollout.py` | `autonomo`, `oraculo` (y exógeno, Duan en etapas posteriores) |
| `evaluacion.py` | `eval_rows`, `Evaluador` (métricas, razones, skill, bootstrap), diagnósticos |
| `modos.py` | espectro (par conjugado = 1 modo), selección de modos |
| `registro.py` | runs MLflow en el estado local, resúmenes `output/`, `decisiones.json` |
| `run.py` | CLI por etapa |

## Flujo

```powershell
# base: única etapa que descarga (sqlite + artefactos de los runs v2 y lightgbm-v1); rechaza si ya hay sqlite local
.\.venv\Scripts\python pipeline/operador/run.py --etapa base --profile <AWS_PROFILE> --state-dir .\mlflow-op --local-validated-dir .\validated-op
# etapas siguientes (1.0, 1.1, …, 1.6, cierre, test), sin AWS
.\.venv\Scripts\python pipeline/operador/run.py --etapa 1.0 --state-dir .\mlflow-op --local-validated-dir .\validated-op --no-download
# publicación tras cada compuerta
.\.venv\Scripts\python scripts/mlflow/publish_state.py plan --state-dir .\mlflow-op --profile <AWS_PROFILE>
.\.venv\Scripts\python scripts/mlflow/publish_state.py apply --yes --state-dir .\mlflow-op --profile <AWS_PROFILE>
```

- `base` fija `output/dmd-operador-global/eval_rows.parquet` y la tabla de continuidad con el DMD v2 (sin runs).
- Cada etapa escribe `output/dmd-operador-global/<etapa>.md` (+ `.json`) y sus runs.
- **`decisiones.json`**: `{etapa: {ganador, config, fecha, nota}}`. Una etapa exige la decisión de la
  anterior (1.1 exige 1.0, …, `cierre` exige 1.6); la `config` se arrastra.
- **Publicación**: `apply` compara el `LastModified` remoto con `<state-dir>/.download.json`, respalda el
  sqlite en `_mlflow/backups/`, sube el sqlite y solo `mlruns/<run_id>/` de runs nuevos; sin borrados.
- `--etapa test` evalúa una sola vez; repetir exige `--force-test` (orden del operador).

## Runs (D10)

- Conjuntos: `sinca-zona-dmdop-{rep}-{exp}-aedmd-v{n}-s{seed}` (`exp`: `e10`, `e11a`…`e16`, `cierre`),
  tags `modelo=dmdop-{rep}`, `stage=dmd-operador-global`, `particion`, `etapa`, `seed`, `ae_run_id`;
  params `tipo` (`operador`; oráculo `diagnostico`; `A₂₄` `referencia`), `etapa`, `estimador`, `rollout`, `rank`, `version`.
- Test por serie: `sinca-{slug}-{pollutant}-dmdop-{rep}-aedmd-v{n}-s{seed}` (contrato LightGBM).
- Artefactos en `<state-dir>/mlruns/<run_id>/artifacts/`: `predictions`, `horizon`, `episodes`,
  `spectrum`, `tramos`, `bootstrap` (`.parquet`), `operator.npz`.
- `output/`, `mlflow-op/` y `validated-op/` no se versionan.
