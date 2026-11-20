# Tasks: `dmd-operador-global`

> Fuente de verdad: `project-decisions.md` §9; detalle en `design.md` (D1–D12). Comandos desde la raíz
> en PowerShell; tests con `.\.venv\Scripts\python -m pytest`, sin AWS. Solo CPU local; sin Training Jobs.
> **[OPERADOR]** = descarga o escritura en S3, mutación de MLflow o test; puede delegarla al
> implementador con confirmación previa, como en `ae-dmd-prep`.
> **⏸ PAUSA** = el implementador se detiene, deja los números en `EXPERIMENT-LOG.md` y espera la
> decisión del operador; la registra en `pipeline/operador/decisiones.json` antes de seguir.
> Comando de etapa (1.9): `.\.venv\Scripts\python pipeline/operador/run.py --etapa <id> --state-dir
> .\mlflow-op --local-validated-dir .\validated-op --no-download` (solo `base` descarga, con `--profile`).
> Publicación (1.7): `.\.venv\Scripts\python scripts/mlflow/publish_state.py {plan|apply --yes}
> --state-dir .\mlflow-op --profile <AWS_PROFILE>`. `output/` no se versiona.
> Si algo no calza, anotarlo en `## Notas de implementación`; si bloquea, volver al arquitecto.

## 0. Preparación

- [x] 0.1 **[OPERADOR]** Rama `feat/dmd-operador-global` desde `main`. Verificar: `git branch --show-current`.
- [x] 0.2 Línea base de tests. Verificar: `.\.venv\Scripts\python -m pytest tests -q` en verde; anotar el conteo.
- [x] 0.3 **[OPERADOR]** Datos validados: `aws s3 sync s3://sinca-data/validated/ .\validated-op --profile <AWS_PROFILE>` (solo lectura). Verificar: existen `validated-op\sinca`, `open_meteo`, `feriados`.

## 1. Grupo 0 — base común (D1–D4, D8, D10)

- [x] 1.1 Crear `pipeline/operador/` (`__init__.py` y módulos de D2) y `tests/operador/conftest.py` (agrega `pipeline/` al path; fixtures: sistema lineal estable `d = 6`, sinusoide amortiguada con período 24 h y vida media 36 h, grilla de 280 días de `tests/ae/conftest.synthetic_validated`, estado MLflow con `artifact_location = file:///opt/ml/mlflow/mlruns`). Agregar `mlflow-op/` y `validated-op/` a `.gitignore`. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`; `git check-ignore mlflow-op validated-op`.
- [x] 1.2 `estado.py`: `Estado` (`z [N, d]`, `index`, `segments`, `d_mp`, `decode(z) -> [N, 24, 6]` en µg/m³) y constructores `hankel(grid, norm, h, canales)`, `hankel_por_estacion`, `pca_state(…)` (PCA(12) MP + PCA(4) aux, ajuste en train) y `ae_state(latent_df, decoder, norm)` desde `latent.parquet`. Ventanas con `ae.data`; normalizador reconstruido desde `--local-validated-dir` con el `train` de la partición. Tests: Hankel decodifica exacto la ventana; `pca_state` reconstruye un dato de rango 12 sin error; ninguna fila fuera de `train` entra al ajuste. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_estado.py -q`.
- [x] 1.3 `estimadores.py`: `exacto` (envuelve `ae.dmd.fit_dmd`), `truncado(rank)`, `gavish_donoho_rank(sigma, shape)`; `modos.espectro(K) -> DataFrame` (λ, período h, vida media h; par conjugado = 1 modo). Tests: Hankel de la sinusoide amortiguada recupera período y vida media (±2 %); GD = rango de `pydmd` con `svd_rank=0` en un sintético; exacto = `pydmd.DMD` en autovalores ≤ 1e-6 (`importorskip`). Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_estimadores.py -q`.
- [x] 1.4 `rollout.py`: `autonomo(K, estado, t_idx, pasos=24)` con `matrix_power` (o `Ã²⁴` en rango reducido) → `decode` → promedio 24 h `[N, 6]` (D6). Test: sobre un sistema lineal exacto el pronóstico iguala la verdad. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_rollout.py -q`.
- [x] 1.5 `evaluacion.py`: `eval_rows(…)` (D3), `metricas(preds, refs, rows)` (RMSE ponderado con `evaluate.sample_weights`, razón vs mejor referencia y vs cada una, skill vs persistencia, por serie y media), `bootstrap_bloques(…, dias=7, B=2000, seed=0)` emparejado (D8), `diferencia(a, b)` con intervalo, comparación sobre intersección para `h = 48`, y diagnósticos (horizonte 1…24, episodios ≥ ICAP100, pendiente/sesgo por tramo). Error si una variante no cubre `eval_rows`. Tests: misma semilla → intervalos idénticos; variante = referencia → diferencia 0 con intervalo [0, 0]; fila faltante → error con serie y variante; mejor referencia elegida por serie. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_evaluacion.py -q`.
- [x] 1.6 `registro.py`: nombres y tags de D10; `log_variante(…)` con artefactos en `<state-dir>/mlruns/<run_id>/artifacts/` (D1) y `resumen_md(etapa, …)` → `output/dmd-operador-global/<etapa>.md`. Tests (fixture con `artifact_location` de `/opt/ml`): artefactos legibles con `analysis.load.artifact_path`; 3 semillas × 2 etapas → 6 nombres distintos; oráculo con `tipo=diagnostico` e `is_series_run` falso. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_registro.py -q`.
- [x] 1.7 `scripts/mlflow/publish_state.py` (script simple, D1): `plan` (runs nuevos con `stage=dmd-operador-global`; falla si alguno no tiene artefactos) y `apply --yes` (aborta si `LastModified` remoto ≠ `.download.json`; respaldo `_mlflow/backups/mlflow-<AAAAMMDD-HHMM>.sqlite`; sube sqlite + `mlruns/` de runs nuevos; reescribe `.download.json` con `head_object`). `--profile` obligatorio; sin `--yes` no escribe. Tests con S3 falso: `LastModified` distinto → 0 subidas; tras `apply`, `.download.json` tiene el nuevo valor; solo se suben runs nuevos. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_publish.py -q`.
- [x] 1.8 `decisiones.json` (vacío, esquema `{etapa: {ganador, config, fecha, nota}}`) y lectura en la CLI: una etapa sin la decisión de la anterior termina con error. Test. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_decisiones.py -q`.
- [x] 1.9 `run.py`: CLI `--etapa {base,1.0,1.1,1.2,1.3,1.4,1.5,1.6,cierre,test}`, `--state-dir`, `--local-validated-dir`, `--no-download`, `--profile` (solo `base`), `--version` (defecto 1), `--seeds 0,1,2`. `base`: rechaza si existe sqlite local; descarga selectiva de D1 (incluye `lightgbm-v1` vía `source_run_id` de LightGBM v2) y `.download.json`; arma `eval_rows.parquet`; evalúa DMD exacto autónomo de los 6 AE y escribe la tabla de continuidad en `output/` (sin runs, D10). Test e2e sintético (`--no-download`, patrón `tests/ae/test_run_partition_e2e.py`). Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`; `run.py --help`.
- [x] 1.10 `pipeline/operador/README.md`: flujo por etapas, comandos, publicación, `decisiones.json`, nombres de run, `output/` no versionado. Verificar: los comandos del README corren con `--help`.
- [x] 1.11 **[OPERADOR]** `run.py --etapa base --profile <AWS_PROFILE> --state-dir .\mlflow-op --local-validated-dir .\validated-op` (solo lectura en S3). Pegar en notas el conteo de objetos y el `LastModified`.
- [x] 1.12 Continuidad (spec): para conv1d y GRU s0–s2, K reajustado vs `dmd_operator.npz` (≤ 1e-8 relativo) e `y_pred` vs `predictions.parquet` DMD v2 en `eval_rows` (≤ 1e-4 relativo). Pegar la tabla y `n` por serie (base 1407). Verificar: tabla en notas.
- [x] 1.13 Suite en verde. Verificar: `.\.venv\Scripts\python -m pytest tests -q` (≥ conteo de 0.2 + nuevos).

## 2. Compuerta 1 — oráculo (1.0)

- [x] 2.1 `rollout.oraculo(estado, t_idx)`: decodifica el estado real en `t+24` (AE: latente de `t+24` de `latent.parquet`; PCA: proyección de la ventana real) y promedia. Hankel no tiene oráculo. Test: con decoder identidad el oráculo = target. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_rollout.py -q`.
- [x] 2.2 `run.py --etapa 1.0`: oráculo de conv1d y GRU (3 semillas) y PCA; tabla por serie y media (razón vs mejor referencia, vs LightGBM y vs persistencia, con intervalo) y lectura por intervalo de D9. Verificar: 7 runs `…-e10-…` con `tipo=diagnostico` y `output/dmd-operador-global/1.0.md`.
- [x] 2.3 Entrada en `EXPERIMENT-LOG.md` (plantilla del archivo): oráculo 1.0, filas, cifras, lectura. Verificar: sin IDs de cuenta ni ARNs.
- [x] 2.4 ⏸ **PAUSA** — decisión del operador: AE siguen o salen (Hankel línea principal). Registrar en `decisiones.json` y en el log.
- [x] 2.5 **[OPERADOR]** Publicar: `plan` y `apply --yes` (pegar plan y objetos subidos).

## 3. Escalera de representaciones (1.1a–g)

- [x] 3.1 Estados de 1.1b (`hankelmet`), 1.1f (`hankelest`) y 1.1g (`hankel48`) en `estado.py` si faltan, con tests de forma y decodificación exacta. Verificar: `.\.venv\Scripts\python -m pytest tests/operador/test_estado.py -q`.
- [x] 3.2 `run.py --etapa 1.1`: DMD exacto autónomo sobre 1.1a `hankel` (GD), 1.1b `hankelmet` (GD), 1.1c `pca` (completo, GD reportado), 1.1d conv1d s0–s2 y 1.1e GRU s0–s2 (completo; omitidos si 2.4 los sacó), 1.1f y 1.1g solo Hankel (1.1g emparejado sobre su intersección, D3). Puntajes, diferencias emparejadas vs `hankel`, regla GRU (D9) y diagnósticos. Verificar: runs `…-e11a…e11g-…` y `output/dmd-operador-global/1.1.md`.
- [x] 3.3 `EXPERIMENT-LOG.md`: escalera, propuesta de ≤ 2 representaciones (D9), aporte de meteo (1.1b vs 1.1a), cruce entre estaciones (1.1f) y memoria (1.1g; si gana claro, se registra para decidir después, D12). Verificar: sin IDs.
- [x] 3.4 ⏸ **PAUSA** — el operador fija las ≤ 2 representaciones que siguen. Registrar.
- [x] 3.5 **[OPERADOR]** Publicar (como 2.5).

## 4. Picos (1.2)

- [x] 4.1 `estimadores.ponderado(segments, w)` con pesos de par `max_s w_s(target_s(t))` solo en `row_mask(…, "train")` (D5); `rollout.duan(…)` con `S_{s,k}` de rollouts de train con `t+24` en train (D7). Tests: `W = I` = exacto; con residuos log-normales sintéticos Duan corrige el sesgo de la media (±2 %); ningún `t` con `t+24` fuera de train aporta. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 4.2 `run.py --etapa 1.2`: sobre las representaciones de 3.4, base vs 1.2a vs 1.2b vs 1.2a+1.2b; error en episodios y sesgo por tramo destacados. Verificar: runs `…-e12-…` y `output/dmd-operador-global/1.2.md`.
- [x] 4.3 `EXPERIMENT-LOG.md` + ⏸ **PAUSA** — decisión global de picos. Registrar.
- [x] 4.4 **[OPERADOR]** Publicar.

## 5. Exógenas (1.3)

- [x] 5.1 `rollout.exogeno(…)` y ajuste `K_mm, K_ma` (con o sin pesos según 4.3), D6: AE/PCA con ventana aux del horizonte (meteo y máscaras por persistencia diurna, calendario real; AE `encode(x_mp, x_aux')[d_mp:]`, PCA proyecta); Hankel DMDc con entrada meteo persistida + calendario de cada paso, sin máscaras, GD sobre `[X; U]`. Tests: alterar meteo posterior a `t` no cambia el pronóstico; horas `τ > t` usan `τ−24` y calendario de `τ`; DMDc recupera `A, B` de un sintético. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 5.2 `run.py --etapa 1.3`: 1.3a autónoma vs 1.3b exógena sobre la configuración vigente; `K_ma` como artefacto. Verificar: runs `…-e13-…` y `output/dmd-operador-global/1.3.md`.
- [x] 5.3 `EXPERIMENT-LOG.md` + ⏸ **PAUSA** — decisión. Registrar.
- [x] 5.4 **[OPERADOR]** Publicar.

## 6. Horizonte (1.4)

- [x] 6.1 `estimadores.directo24(…)` (D5). Test: en un sistema lineal exacto `A₂₄ = K²⁴`. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 6.2 `run.py --etapa 1.4`: K horario (vigente) vs `A₂₄` (`tipo=referencia`). Multi-paso diferido a `deep-koopman` (anotarlo en el log). Verificar: runs `…-e14-…` y `output/dmd-operador-global/1.4.md`.
- [x] 6.3 `EXPERIMENT-LOG.md` + ⏸ **PAUSA** — decisión. Registrar.
- [x] 6.4 **[OPERADOR]** Publicar.

## 7. Estimador y estabilidad (1.5)

- [x] 7.1 `estimadores.recorte(K, eps=1e-3)` y `estimadores.fbdmd(segments, w=None, rank)` en numpy (D5: raíz más cercana a `eig(K_f)`, real, siempre con recorte); `modos.bootstrap_modos(…, dias=7, B=200)` con emparejamiento húngaro de D8. Tests: recorte → real y radio ≤ 1−ε; `fbdmd` en rango `r` = fórmula en coordenadas completas; con ruido de medición y de proceso sintético, fbDMD queda más cerca de los autovalores verdaderos que el exacto; bootstrap sobre un sistema sin ruido → 100 % de aparición; modo real comparado solo por vida media. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 7.2 `run.py --etapa 1.5`: exacto vs recorte vs fbDMD+recorte (con los pesos de 4.3 si aplican) + bootstrap por modo del elegido; espectro con intervalos y estabilidad entre semillas (AE). Verificar: runs `…-e15-…` con `bootstrap_modos.parquet` y `output/dmd-operador-global/1.5.md`.
- [x] 7.3 `EXPERIMENT-LOG.md` + ⏸ **PAUSA** — decisión. Registrar.
- [x] 7.4 **[OPERADOR]** Publicar.

## 8. Rango y modos (1.6)

- [x] 8.1 `modos.eliminacion_atras(…)`: techo GD, quita de a un modo (par conjugado junto) el de menor empeoramiento mientras no supere el intervalo bootstrap emparejado y se detiene en el primero significativo (D9); curva error ponderado 24 h vs nº de modos con intervalo. Test: sistema con 2 modos relevantes y 4 de ruido → elimina los 4 de ruido y conserva los 2. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 8.2 `run.py --etapa 1.6` sobre la configuración vigente (`d = 16` fijo). Verificar: run `…-e16-…` con `curva_modos.parquet` y `output/dmd-operador-global/1.6.md`.
- [x] 8.3 `EXPERIMENT-LOG.md` + ⏸ **PAUSA** — el operador fija rango, modos y configuración final. Registrar.
- [x] 8.4 **[OPERADOR]** Publicar.

## 9. Cierre: test, criterio e interpretabilidad (D11, D12)

- [x] 9.1 `run.py --etapa cierre`: reajuste hasta 2026-06-30 (Hankel/PCA: normalización, PCA y K; AE: solo K más pesos/Duan, encoder congelado), con `r` y modos de 1.6 fijados por emparejamiento húngaro (sin re-seleccionar); sin evaluar test. Test: el operador reajustado conserva el número de modos y los emparejados. Verificar: run `…-cierre-…` sin métricas `test_` y `.\.venv\Scripts\python -m pytest tests/operador -q`.
- [x] 9.2 `run.py --etapa test` con guardia de una sola evaluación (`--force-test` solo por orden del operador) y test que la ejercite. Referencias: `lightgbm-v1` holdout y persistencia, filas comunes. Runs por serie con contrato LightGBM (`split = test`, `evaluacion=test`). Veredicto único de D12 con intervalos. Verificar: `.\.venv\Scripts\python -m pytest tests/operador -q` y `--help`.
- [ ] 9.3 **[OPERADOR]** Confirmar y correr `--etapa test` una vez. Pegar tabla por serie y veredicto.
- [x] 9.4 Si el ganador es Hankel o PCA: rolling de 12 ventanas (`features.build.make_rolling_windows`) como dato adicional, sin elegir; si es AE, "no aplica" (Non-Goals). Verificar: `output/dmd-operador-global/rolling.md` o nota.
- [x] 9.5 `notebooks/dmd_operador_global.ipynb` (matplotlib, con outputs, sin credenciales) con los entregables de la spec: espectro con intervalos, formas de modo por estación/contaminante (fase), descomposición de episodios por modo, estabilidad semillas/bootstrap, `K_ma` si exógeno. Excepción `!notebooks/dmd_operador_global.ipynb` en `.gitignore`. Verificar: corre de punta a punta sobre `mlflow-op` y `git check-ignore` no lo ignora.
- [ ] 9.6 `EXPERIMENT-LOG.md`: entrada de cierre (configuración final, test, veredicto, interpretabilidad, tiempo de CPU) y `README.md` (estado de Fase 4, cómo reproducir). Verificar: sin IDs de cuenta ni ARNs.
- [ ] 9.7 **[OPERADOR]** Publicar (como 2.5).
- [ ] 9.8 **[OPERADOR]** (después de 9.7) Verificación de punta a punta: `.\.venv\Scripts\python -m pytest tests -q` en verde; `sel_summary.py --partition ae-dmd --version 2 --profile <AWS_PROFILE>` sigue funcionando con los runs nuevos; funciones del dashboard sin errores sobre el estado publicado (como 12.4 de `ae-dmd-prep`); `openspec validate dmd-operador-global --strict` válido.

## Workflow follow-up

- [ARQUITECTO] `project-decisions.md` §3.4/§9: criterio de éxito de D12 (mejor de LightGBM y persistencia, piso en las 6), §8 (7.1 completado), ganador y si entra `dmd-localizado` o `deep-koopman` (multi-paso diferido).
- [ARQUITECTO] `## Notas de este proyecto` en `modules/tracking-experimentos/blueprint.md` (estado local, `artifact_location` de `/opt/ml`, publicación) y `modules/compute/blueprint.md` (post-hoc en CPU local, tiempos); revisar `docs/ae-dmd-propuesta-experimentos.md` si alguna decisión de etapa lo contradice.
- `/opsx/sync` y `/opsx/archive dmd-operador-global` (implementador, por CLI).

## Notas de implementación

- 0.2: línea base 147 tests en verde.
- 0.3: `validated-op/` con `sinca`, `open_meteo`, `feriados` (286 objetos, ~10 MB).
- 1.6: los artefactos se escriben directo en `<state-dir>/mlruns/<run_id>/artifacts/`; el `artifact_uri` del run queda el derivado del experimento (`/opt/ml/...`, como los runs v2): no se reescribe en el sqlite. `analysis.load` lo resuelve por ruta.
- 1.6: `bootstrap.parquet` = tabla por serie y media con estimación puntual e intervalo 95 % (la distribución se recalcula desde `predictions.parquet` con semilla 0).
- 1.7: "runs nuevos" = runs con `stage=dmd-operador-global` no listados en `published_runs` de `.download.json` (lo escribe `apply`).
- 1.9: `eval_rows.parquet` va en `output/dmd-operador-global/` (no versionado), junto a los resúmenes.
- 1.11: base sobre S3 (solo lectura): 218 objetos, 55 runs (6 AE, 36 DMD, 6 LightGBM v2, 1 persistencia v2, 6 `lightgbm-v1`); `LastModified` del sqlite 2026-11-18T04:14:52+00:00.
- 1.12: continuidad exacta (n por serie: LE pm10/pm25 1776/1776, Ñielol pm10/pm25 1412/1407, PLC II pm10/pm25 1636/1635; base 1407):

  | AE v2 | K rel. | y_pred máx. rel. | RMSE pond. media |
  |---|---|---|---|
  | conv1d s0 / s1 / s2 | 0 / 0 / 0 | 0 / 0 / 0 | 39.392 / 39.508 / 37.401 |
  | gru s0 / s1 / s2 | 0 / 0 / 0 | 0 / 0 / 0 | 36.934 / 36.171 / 35.767 |

  Referencias en `eval_rows`: LightGBM 34.236, persistencia 30.938 (= `sel_summary` v2).
- 2.1: el oráculo AE usa el latente de `latent.parquet` en `t+24`; donde la ventana `[t+1, t+24]` no es elegible (24 de 1976 posiciones) se codifica la ventana real con `encoder.keras`.
- 2.2: el oráculo no tiene operador: `spectrum.parquet` vacío y sin `operator.npz`.
- 1.1: `logs/` agregado a `.gitignore` (logs locales de procesos). Los helpers sintéticos van en `tests/operador/sint_operador.py` y el e2e en `test_operador_e2e.py`: con dos `conftest.py` y dos `test_run_e2e.py`, pytest choca por nombre de módulo; `tests/operador/conftest.py` reexpone `TZ` y `STATION_NAMES` para los `from conftest import …` diferidos de `tests/ae`.
- 1.13: 181 tests en verde (147 + 34).
- 2.4 / 3.4: decisiones tomadas por regla D9 (delegadas por el operador); cifras en `EXPERIMENT-LOG.md` y `decisiones.json`.
- 3.2: selección D9 implementada como: ganador = la más simple (orden hankel < hankelmet < hankelest < hankel48 < pca < conv1d < gru) que ninguna otra supera de forma significativa; pasa además la de mejor puntaje del resto si no es significativamente peor. `hankelest` y `hankel48` son ablaciones (no candidatas). Lectura secuencial alternativa (reemplazo desde Hankel): GRU es la mejor; la segunda por puntaje (conv1d) es significativamente peor y la siguiente (PCA) no; se toma {PCA, GRU}, el mismo conjunto.
- 2.5 / 3.5: una sola publicación tras 3.4 (la de 2.5 no se hizo antes de correr 1.1): `plan` = 18 runs nuevos (7 `e10` + 11 `e11*`); `apply --yes` subió 121 objetos (sqlite + 119 artefactos); respaldo `_mlflow/backups/mlflow-20261119-2307.sqlite`; `LastModified` nuevo 2026-11-20T02:07:21+00:00 en `.download.json`. boto3 en Windows con `AWS_CA_BUNDLE` del proceso apuntando al bundle del profile (ruta Windows).
- 4.1: par `(t, t+1)` con `t` fuera de `row_mask(train)` (últimas 24 h de train) entra con peso 1 (no aporta peso derivado del target), para que `W = I` siga siendo el exacto.
- 4.2: `{exp}` por variante: `e12base`, `e12a` (pesos), `e12b` (Duan), `e12ab`; con `e12` solo, los 4 runs de una semilla compartirían nombre. Igual en 1.3 (`e13a`/`e13b`) y siguientes.
- 4.3: orden de simplicidad usado: base < Duan < pesos < pesos+Duan (Duan no cambia el operador; D9 "sin peso < con peso").
- 4.4: 16 runs `e12*` publicados (respaldo `mlflow-20261119-2312.sqlite`). Un segundo `apply` sin runs nuevos subió igual el sqlite y dejó otro respaldo (`…-2313`); desde entonces `apply` no escribe si no hay runs nuevos (test agregado).
- 5.1: Hankel DMDc implementado y testeado (`entrada_hankel`, `estimadores.dmdc` con GD sobre `[X; U]`), no corrido: Hankel salió en 3.4. Convención común: `z_{t+1} = A z_t + B u_t`, con `u_t` observado y `u_{t+k}` (`k ≥ 1`) persistido.
- 5.2: con Duan vigente, los factores `S_{s,k}` se reestiman con el rollout exógeno.
- 5.4: 8 runs `e13*` publicados, 62 objetos; respaldo `mlflow-20261119-2319.sqlite`.
- 6.1: test `A₂₄ = K²⁴` en `tests/operador/test_rollout.py`. Con Duan vigente, los factores se reestiman con el rollout de `A₂₄`.
- 6.4: 8 runs `e14*` publicados, 58 objetos; respaldo `mlflow-20261119-2324.sqlite`.
- 7.1: el test "fbDMD más cerca de la verdad con ruido de medición y de proceso" usa trayectorias transitorias (condición inicial por segmento, ruido de proceso chico): con forzamiento estacionario fbDMD tiende a `|λ| = 1` (adelante y atrás comparten sesgo), y el test no se cumple. El bootstrap sin ruido usa también segmentos transitorios (con una sola trayectoria que decae, los modos rápidos se pierden en bloques tardíos).
- 7.2: el bootstrap re-estima con pesos = conteos de los días remuestreados (LS ponderado ≡ pares repetidos).
- 7.4: 12 runs `e15*` publicados (con `bootstrap_modos.parquet`), 90 objetos; respaldo `mlflow-20261119-2328.sqlite`.
- 8.1: "empeoramiento significativo" = intervalo de `nuevo − actual` entero sobre 0 (con tolerancia numérica 1e-9). La eliminación termina en 1 modo si nunca hay empeoramiento significativo.
- 8.2: el techo GD se aplica también a PCA y AE (rango 6 y 3–4); con Duan vigente, los factores se reestiman para cada candidato. `{exp}` = `e16`; el resumen reporta además la diferencia final − rango completo.
- **8.3 (PAUSA, sin decisión)**: la regla D9 daría PCA rango 6 con 1 modo + Duan (36.32; razón 1.240 [0.913, 1.564]), pero (a) el criterio elimina el modo diario en las 4 variantes (el target de 24 h lo cancela), en tensión con §9 ("ciclo diario = modo de la dinámica"), y (b) la acumulación de pasos no significativos deja un operador peor en punto que GRU rango completo + Duan (29.73; razón 1.014 [0.968, 1.066]). Decide el operador, p. ej.: aceptar la regla; conservar el modo diario forzado; no aplicar techo GD en PCA/AE (eliminar desde rango 16); o fijar GRU rango completo (o su resultado de 1.6, 30.99) como configuración final. 8.4 (publicar 1.6) queda tras la decisión: los 4 runs `e16` están solo en el estado local.
- 8.3: **decisión del operador (2026-11-19)**: regla 1.6 corregida (rango completo en PCA/AE; eliminación contra la configuración inicial; par ≈ 24 h protegido, armónicos solo con aparición ≥ 80 % en bootstrap) y candidato oficial GRU, PCA secundario. Enmienda corta en `design.md` (D9/D11) y en la spec (requisito "Rango y modos sin grilla" + 2 escenarios). 1.6 se re-corrió como `version = 2` (D10): GRU 5/5/6 modos, 29.79 [razón 1.015 (0.955, 1.068)]; PCA 3 modos, 36.16. Los 4 runs `e16` v1 llevan `descartado=regla-vieja` y `publish_state.py` excluye runs con tag `descartado` (sus metadatos van igual en el sqlite).
- 8.3: tolerancia para identificar el par diario y los armónicos: ±5 % del período (con ±10 % se tomaba un modo de 10.8 h como armónico de 12 h).
- 8.4: 4 runs `e16` v2 publicados (36 artefactos + sqlite); respaldo `mlflow-20261120-0000.sqlite`. `openspec validate dmd-operador-global --strict`: válido (el requisito de modos se partió en dos para no exceder el largo).
- 9.1: `--etapa cierre` reajusta GRU (3 semillas; solo K + Duan con latentes train+es+sel del encoder congelado; pares solo dentro de cada tramo del latente) y PCA (normalización, PCA y K con train+es+sel como un bloque) a rango completo, y fija los modos de 1.6 v2 por emparejamiento húngaro: los 19 modos se emparejan con `aparece = True` (p. ej. diario GRU s0 24.17/313 h → 24.17/312 h). Duan: filas con `t+24 < 2026-07-01`. Runs `…-cierre-aedmd-v1-s{seed}` sin métricas `test_` (`operator.npz`, `spectrum`, `bootstrap_modos` B = 200, `emparejamiento`); índice en `output/dmd-operador-global/cierre.json`. Test sintético en `test_modos.py` y e2e completo (1.6 → cierre → test → guardia) en `test_operador_e2e.py`.
- 9.1: interpretabilidad (§9) calculada en el cierre, con todos los modos conservados como "dominantes": **GRU no cumple** (5 modos; solo el diario está en las 3 semillas y en ≥ 80 % del bootstrap, y sus vidas medias 288–361 h no solapan entre semillas); **PCA no cumple** (3 modos; los dos reales < 80 % de aparición).
- 9.2: `--etapa test` usa la config de la decisión 1.6; referencias `lightgbm-v1` (holdout, split test) y persistencia calculada en el proceso; filas comunes; t con `row_mask(test)`. Veredicto D12 sobre GRU (media de semillas); PCA solo reportado. "PLC II skill > 0" se aplica a las 2 series de PLC II (punto). Runs por serie con `tipo=operador`, tag `evaluacion=test`, `rol` oficial/secundario (6 × 3 GRU + 6 PCA = 24). Guardia: rechaza si ya hay runs `etapa=test` de la versión, salvo `--force-test`.
- 9.4: no aplica (ganador oficial = GRU, AE; Non-Goals).
- 9.5: notebook generado y ejecutado sobre `mlflow-op` (con outputs, sin rutas ni IDs) con `nbclient`; para eso se instalaron `ipykernel`, `nbclient` y `nbformat` en el `.venv` (no están en requirements). Se generó antes del test: no incluye cifras de test. Formas de modo y descomposición de episodios en GRU vía jacobiano del decoder en el latente medio (aproximadas); en PCA, exactas. `K_ma`: no aplica (rollout autónomo).
- 9.1: runs de cierre publicados antes del test (registro previo de la configuración): 4 runs, 18 objetos; respaldo `mlflow-20261120-0006.sqlite`.
- **9.3 (PAUSA)**: listo para la evaluación única en test, a la espera de la autorización del operador. Comando: `.\.venv\Scripts\python pipeline/operador/run.py --etapa test --state-dir .\mlflow-op --local-validated-dir .\validated-op --no-download`. Luego 9.6–9.8.
