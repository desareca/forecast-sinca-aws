# Design: `deep-koopman`

## Context

- Vinculante: `project-decisions.md` §9 (partición, presupuesto fijo, criterio D12, enmienda 1.6,
  resultado de 7.1, regla "todo el cómputo en AWS"). Razonamiento: `docs/ae-dmd-propuesta-experimentos.md`
  §7 (con la nota de 2026-11-20). Motivo: `proposal.md`.
- Referencia a superar: GRU+DMD+Duan de 7.1: `sel` 29.79 (razón 1.015 [0.955, 1.068]); test razón
  1.030 [0.975, 1.176]. Runs: `sinca-zona-dmdop-gru-e16-aedmd-v2-s{seed}` (sel),
  `…-e15a-aedmd-v1-s{seed}` (`operator.npz` de rango completo), test por serie
  `sinca-{slug}-{pollutant}-dmdop-gru-aedmd-v1-s{seed}`.
- Base de arranque: `sinca-zona-ae_gru-aedmd-v2-s{seed}` (`encoder.keras`: `enc_mp` = GRU(64) sobre
  6 canales MP + Dense → 12; `enc_aux` = GRU(64) sobre 34 canales → 4).
- Reutilizable: `ae/data.py` (grilla, normalizador, ventanas, `mp_to_ugm3`, `CALENDAR_CHANNELS`),
  `ae/models.py` (`linear_decoder` como patrón), `operador/{evaluacion,modos,rollout,registro}.py`
  (filas fijas, bootstrap, emparejamiento, Duan, eliminación), `train/{launch,mlflow_log}.py`,
  `infra/schedule.tf` (patrón EventBridge + rol acotado), `infra/iam-training.tf`.
- 7.1 midió: modos en `|λ| ≈ 0.998–0.9995`; el diario es el único identificable; Duan aporta
  −6.6 en `sel`; el promedio de 24 h cancela el modo diario.

## Goals / Non-Goals

**Goals:** saber si un Koopman entrenado da modos identificables entre semillas (criterio D11)
sin perder la precisión de 7.1; reportar D12 e interpretabilidad del §9.

**Non-Goals:** `A₂₄`, fbDMD y recorte; meteo exógena en el horizonte; K condicionado (LPV) y red de
autovalores (→ `dmd-localizado`); meteo futura; híbridos; grillas de `d`, `L`, mezcla de bloques o
λ (salvo B1); ensamble y encoder lineal (C1/C2: follow-up del arquitecto); cómputo local de experimentos;
cambios al dashboard más allá de no romperse.

## Decisions

### D1. Cómputo y orquestación

- SageMaker Training Job CPU `ml.m5.large`, imagen `sinca-train:tf-cpu-v2` (TF 2.20, `pydmd`,
  scipy), código por `source_dir = pipeline/`, entry point nuevo `koopman/run.py` (se agrega a
  `ENTRY_POINTS` de `train/launch.py`). Datos desde `sinca-data/validated/` (rol de training).
- Un job por (encoder, semilla, fase); las 3 variantes de K de F1 corren en serie dentro de un job
  por semilla. Orquestación en serie con Step Functions (D18); el humo se lanza a mano con
  `train/launch.py`. Experimento `sinca-baseline`; humo y ejecución seca en `sinca-smoke`.
- Profile: el de `.env` (`AWS_PROFILE`), nunca `default`. CLI vía WSL; boto3/SDK en Windows con
  `AWS_CA_BUNDLE` del proceso (nota de 7.1).
- Descarga selectiva (`--mlflow-runs`): sqlite + runs de arranque, referencias (36 DMD v2,
  LightGBM v2, persistencia v2, GRU 7.1) y el `parent_run_id`.

### D2. Publicación desde el job

- La lógica de `scripts/mlflow/publish_state.py` (chequeo de `LastModified`, respaldo
  `_mlflow/backups/mlflow-<AAAAMMDD-HHMM>.sqlite`, subida del sqlite y de `mlruns/<run_id>/` de runs
  nuevos) pasa a `train/mlflow_log.py::publish_new_runs` y el script la llama (sin cambiar su CLI).
- El job guarda el `LastModified` al descargar y llama a `publish_new_runs` al final. Si el remoto
  cambió: no sube, copia su `mlruns/<run_id>/` y un sqlite parcial a `/opt/ml/model` y falla; el
  operador lo publica después con el script.
- Solo tras publicar con éxito, el job escribe su **marcador de éxito**
  `model-artifacts/orquestacion/deep-koopman/<campaña>/jobs/<clave>.json` (`run_id`s, artefactos
  verificados, fecha). Sin marcador, el job cuenta como no hecho (D18).

### D3. Módulos (`pipeline/koopman/`)

`modelo.py` (encoders, decoders, capa `K` por bloques, `B`), `perdidas.py`, `entrenar.py` (bucle,
calentamiento, currículo), `inicio.py` (arranque en caliente, init de `K`), `identificabilidad.py`
(aportes, dominantes, emparejamiento, bootstrap, criterio), `evaluar.py` (rollout con `u`, decode,
Duan, arnés de `operador`), `seleccion.py` (reglas preregistradas → selección, D14), `run.py` (CLI
`--fase`, `--dry`), `decisiones.json` (registro), `README.md`. Tests en
`tests/koopman/` (sintéticos, pocas épocas).

### D4. Entradas y latente

- Encoder: ventana `L = 24` de MP (6) + meteo (12) + máscaras de disponibilidad (12), normalizadas
  como el AE (`ae.data`, estadísticas de `train`). Máscaras sí: los huecos largos quedan en 0 y sin
  máscara serían indistinguibles de un valor real.
- Calendario fuera del encoder; `u_{t+1}` = `dow_sin`, `dow_cos`, `is_holiday`, `gec_window` de
  `t+1` (sin Fourier diaria, que es el modo diario de `K`).
- Latente único `z` de `d = 10` (2 + 2 + 2 + 4).
- **No se reconstruye la meteo**: obligaría a `K` a pronosticarla de forma autónoma (H7/S2 del doc);
  la meteo entra solo como información de estado para el MP.

### D5. Encoders y arranque

- GRU: GRU(64) + Dropout 0.1 + Dense(`d`). Arranque desde cada semilla GRU v2 (3 semillas
  independientes): kernel recurrente, sesgo y filas de entrada de MP desde `enc_mp`; filas de meteo
  y máscaras en 0 (al inicio el estado oculto coincide con v2); Dense = `W_v2·V_d` (las `d`
  direcciones principales de la Dense 64→12 de v2).
- conv1d somero: Conv1D(64, k = 5, `same`, relu) + Flatten + Dropout 0.1 + Dense(`d`), desde cero.
  **Asimetría anotada**: GRU parte en caliente, conv1d no; la comparación de F1 la arrastra.

### D6. Decoders

- Lineal (principal): Dense `z` → 24×6 (variante `linear_decoder`). Inicio: mínimos cuadrados del
  latente inicial de train a las ventanas MP, seguido de un **reescalado por bloque**: para cada
  bloque `b` de K, `s_b = ‖W_b‖_F / √n_b` (`n_b` = columnas del bloque, 2 o 1); `W_b ← W_b / s_b`,
  filas del bloque en la Dense final del encoder y su sesgo `← s_b ·`, filas de `B` `← s_b ·`. K no
  cambia (un escalar dentro del bloque conmuta con `ρR(θ)` y con `λ`): reconstrucción y dinámica
  quedan idénticas y `L_norma` parte en 0.
- **`L_norma` = Σ_b (‖W_b‖_F² / n_b − 1)²** (enmienda del operador tras el humo 8.1, reemplaza a la
  penalización de ortogonalidad, que valía ~2·10⁴ al inicio, dominaba el gradiente y pedía una
  ortogonalidad entre modos innecesaria: con K en bloques la mezcla entre bloques ya está fijada y
  rotación o escala dentro de un bloque no cambian período, vida media ni aporte). Solo fija la
  escala, para evitar el colapso `z → 0` que vuelve trivial `L_lin`. Con decoder GRU no aplica.
- GRU no lineal (comparación preregistrada, solo F1 con encoder GRU): `RepeatVector` + GRU(64) +
  `TimeDistributed(Dense(6))`, como v2.

### D7. Operador

- `K = diag(R_diario, R_1, R_2, ρ_1…ρ_4)`, `R = ρ·[[cos θ, −sin θ], [sin θ, cos θ]]`.
  `ρ = exp(−(δ + softplus(a)))`, `δ = ln 2 / 1000` (vida media ≤ 1000 h; escala log porque los modos de
  7.1 están en `|λ| ≈ 0.998–0.9995`, donde una sigmoide satura). Diario: `θ = 2π/24` fijo, solo `a`.
  Libres: `θ = π·sigmoid(b)`. `B` (10×4) denso, inicio 0.
- Init GRU (por semilla): autovalores del `operator.npz` de `e15a` de 7.1. Diario ← el más cercano a
  24 h; libres ← los 2 complejos no diarios de mayor `|λ|`; reales ← los 4 reales (o `|arg λ| < 0.005`)
  de mayor `|λ|`; recortados al techo. Init conv1d (genérico): diario vida media 300 h; libres
  (168 h, 100 h) y (48 h, 48 h); reales 24, 72, 168 y 500 h.
- Variantes F1 (solo GRU): `kden` (K denso 10×10 libre, sin techo; control), `perlib` (θ diario
  libre), `sinbu` (sin `B·u`).

### D8. Pérdidas

- `L_rec` = MSE enmascarado de la ventana MP (`masked_channel_mse`). `L_lin` = `‖enc(x_{t+1}) −
  (K z_t + B u_{t+1})‖²`. `L_pred` = Σ_{k≤H} MSE enmascarado de `dec(z_{t+k|t})` vs ventana real en
  `t+k` (no codifica ventanas futuras). `L_tgt` = RMSE ponderado (`w` de la métrica) del promedio
  de 24 h en µg/m³ (`expm1` en el grafo) contra el target.
- Normalización: cada término se divide por su valor en una muestra fija de 2048 ventanas de train
  al inicio de la fase. λ_lin = 1, λ_pred = 1, λ_tgt = 1, fijos. `L_norma` no se normaliza (adimensional, parte
  en 0), λ_norma = 1. B2: λ_cons = 1. B3:
  λ_L1 = 0.1.
- F1 = rec + lin (+ norma). F2 = F1 + pred. F3 = F2 + tgt.
- **F3 variante `icap`** (decisión del operador, PA-9): `L_tgt` sobre ICAP_zona con máximo suave
  (log-sum-exp sobre las 6 series, τ = 5 puntos ICAP, fijo) y peso `1 + max(0, ICAP − 100)`, en el
  encoder principal (3 semillas). Se evalúa con D12 y las métricas ICAP de `evaluate`; ante empate
  queda F3 sin `icap` (más simple).

### D9. Entrenamiento

- Adam, lr 1e-3 con calentamiento lineal de 2 épocas, `clipnorm = 1.0`, batch 128.
- Presupuesto fijo: F1 60 épocas (GRU: 5 con encoder congelado; conv1d: sin congelar), F2 30
  (currículo H = 1, 6, 24 por 10 épocas), F3 20. F2 y F3 arrancan del run de la fase previa.
- Orden canónico al registrar: libres y reales por `ρ` descendente (permutando `z`, decoder y `B`);
  signo de cada real con el mayor peso del decoder positivo. La rotación dentro de un bloque 2×2 no
  se fija: las formas de modo se comparan en módulo y fase relativa.
- **Checkpoints por época**: `CheckpointConfig` con `S3Uri =
  s3://sinca-mlflow/model-artifacts/checkpoints/deep-koopman/<campaña>/<clave>/` y `LocalPath =
  /opt/ml/checkpoints`. Al final de cada época el entry point guarda (sobrescribe, un solo
  checkpoint por variante) pesos de encoder, decoder, `K` y `B`, estado del optimizador, época,
  escalas de normalización de pérdidas y estado de RNG (numpy, Python, TF y el del barajado). Al
  iniciar, si hay checkpoint, retoma en la época siguiente: calentamiento congelado, lr y currículo
  se derivan del número de época, así que no se reinician y el presupuesto total no cambia. En el
  job de variantes de F1, cada variante tiene su subcarpeta y una variante terminada no se repite.
  Determinismo: mismo presupuesto, mismo orden de datos y misma época de retoma; no bit a bit
  (TF en CPU). Sin borrados: el operador limpia el prefijo al cerrar.

### D10. Evaluación

- Rollout de 24 pasos con `u` de `t+1…t+24` → `dec(z_{t+24})` → promedio → µg/m³. Arnés de
  `operador.evaluacion` (filas fijas de 7.1, bootstrap de bloques de 7 días, B = 2000, emparejado),
  diagnósticos de 7.1 y veredicto D12. Duan (`operador.rollout`) con y sin, en F1 y F2; F3 sin Duan.
- Puntaje: media de 6 series; por encoder, media de 3 semillas. Diferencias emparejadas contra
  GRU+DMD+Duan de 7.1 (media de semillas) y entre variantes.

### D11. Identificabilidad (preregistro; criterio de selección y reporte, no compuerta)

- Contribución del modo `j` en `t`: `c_j(t)` = parte del promedio decodificado de 24 h (espacio log1p
  normalizado) que viene del bloque `j` de `z_{t+24|t}` (decoder lineal: exacta; GRU: jacobiano del
  decoder en `z_{t+24|t}`). `B·u` se reporta aparte.
- Aporte `A_j = Σ_t Σ_s w_s(t)·c_{j,s}(t)²` en `sel`. Dominantes = mínimo conjunto, en orden de `A_j`,
  con ≥ 80 % de `Σ_j A_j`. El diario no se exige como dominante (el promedio lo cancela); se reporta.
- Bootstrap: DMD exacto sobre el latente aprendido de train, bloques de 7 días, B = 200,
  emparejado con `K` (D8 de 7.1: húngaro sobre representantes, ±25 %, reales solo vida media).
- Identificable: cada dominante de cada semilla tiene pareja en las otras dos, aparece en ≥ 80 % del
  bootstrap de cada semilla y sus intervalos de período y vida media se solapan entre semillas. Se
  calcula para cada configuración al evaluarla; no ramifica la campaña.

### D12. Variantes de remedio (siempre)

- Se entrenan siempre, como variantes de F1 del encoder GRU (3 semillas c/u): B1 λ_lin ∈ {0.1, 10};
  B2 `L_cons = ‖enc([dec(z), meteo, máscaras]) − z‖²`; B3 `L1 = mean_t Σ_bloques ‖z_bloque‖₂`; B4 `d = 8`
  (diario + 2 libres + 2 reales). Se reporta la razón `L_lin`/`L_rec` normalizada de F1.
- C1 (encoder lineal Hankel/PCA con `K` y pérdidas entrenados) y C2 (ensamble de semillas) quedan
  como follow-up del arquitecto si nada es identificable; no están en la máquina.

### D13. Eliminación de modos

Regla 1.6 enmendada sobre `K` entrenado: quitar un bloque = anular su fila/columna en `K`, su fila
en `B` y su contribución al decode, sin reentrenar; Duan reestimado si aplica. Se aplica a cada
configuración al evaluarla; la selección (D14) toma los bloques de la elegida y el cierre los fija.

### D14. Selección preregistrada (job `--fase seleccion`, antes del cierre)

- Candidatas: configuraciones con decoder lineal que no son de control (`kden`, `perlib`, `sinbu`,
  `decgru` se reportan, no se eligen): F1 GRU, F1 conv1d, B1 (0.1, 10), B2, B3, B4, F2 GRU, F2 conv1d,
  F3 GRU, F3 conv1d, F3 `icap` GRU, F3 `icap` conv1d.
- Orden de simplicidad (fijo): F1 < B1 λ0.1 < B1 λ10 < B2 < B3 < B4 < F2 < F3 < F3 `icap`; dentro de
  cada uno, GRU < conv1d.
- Si alguna candidata es identificable (D11), se elige solo entre esas; si no, entre todas y la
  interpretabilidad queda "no cumple". Se parte de la más simple; una más compleja la reemplaza solo
  si la diferencia emparejada de RMSE ponderado (media de 6 series y de semillas) tiene intervalo
  < 0 (empate → la más simple).
- Decoder GRU significativamente mejor que el lineal: se reporta; el lineal sigue como principal.
- Salida: `model-artifacts/orquestacion/deep-koopman/<campaña>/seleccion.json` (configuración, run
  por semilla, bloques finales de D13, tabla de candidatas, orden aplicado) y un run de MLflow con el
  mismo contenido, ambos antes del cierre. El implementador lo copia a `koopman/decisiones.json`.

### D15. Runs

| Run | Nombre |
|---|---|
| Fase/variante | `sinca-zona-koop-{enc}-{fase}{var}-aedmd-v{n}-s{seed}`; `enc` ∈ `gru`, `conv1ds`; `fase` ∈ `f1`, `b1`, `b2`, `b3`, `b4`, `f2`, `f3`, `cierre`; `var` ∈ ``, `kden`, `perlib`, `sinbu`, `decgru`, `ll01`, `ll10`, `icap` |
| Selección | `sinca-zona-koop-seleccion-aedmd-v{n}` (tag `stage=deep-koopman`, artefacto `seleccion.json`) |
| Test por serie | `sinca-{slug}-{pollutant}-koop-{enc}-aedmd-v{n}-s{seed}` (contrato LightGBM, `evaluacion=test`) |

Artefactos: `encoder.keras`, `decoder.keras`, `koopman.npz` (parámetros de bloque, `K`, `B`),
`latent.parquet`, `predictions.parquet`, diagnósticos de 7.1, `modos.parquet` (aportes, dominantes),
`bootstrap_modos.parquet`, `coherencia.parquet`, curvas de pérdida. `version = 1`.

### D16. Cierre y test

El cierre lee `seleccion.json`: encoder y decoder congelados; `K`, `B` (y Duan si la seleccionada es
F1/B/F2) se reentrenan 20 épocas con su pérdida sobre train+es+sel (hasta 2026-06-30), con los
bloques fijados en D13. Luego el job de test (sin pausa) evalúa solo esa configuración, una vez
(guardia; `--force-test` solo por orden del operador); falla si no hay selección. Referencias:
`lightgbm-v1` holdout, persistencia y test de GRU+DMD+Duan de 7.1.

### D17. Jobs y costo (tiempos del re-humo 8.2)

Tiempos por job del re-humo (paso compilado; notas 8.2 de `tasks.md`), tarifa ~USD 0.115/h
(`ml.m5.large`, no verificada en consola). `MAX_RUN` vigentes en `preparar_ejecucion.py` (margen ≥ 2×:
1 h; variantes de K 2 h).

| Grupo | Jobs | h c/u | h | USD |
|---|---|---|---|---|
| F1 GRU / conv1d / decGRU / variantes K (3 por job) | 3 / 3 / 3 / 3 | 0.29 / 0.13 / 0.46 / 0.79 | 5.0 | 0.58 |
| B1 (λ 0.1, 10) / B2 / B3 / B4 | 6 / 3 / 3 / 3 | 0.29 / 0.39 / 0.29 / 0.29 | 4.7 | 0.54 |
| F2 GRU / conv1d | 3 / 3 | 0.20 / 0.11 | 0.9 | 0.11 |
| F3 y F3 `icap`, GRU / conv1d | 6 / 6 | 0.15 / 0.09 | 1.4 | 0.17 |
| Selección + cierre + test | 3 | 0.1 / 0.19 / 0.1 | 0.4 | 0.05 |
| **Campaña completa** | **48** | | **~12.4** | **~1.45** |
| Ejecución seca (D20: 1 semilla, con falla y reanudación) | 18 + 1 relanzado | ~0.035 | ~0.7 | ~0.08 |

Orquestación (Step Functions, Lambda, SNS, EventBridge) y checkpoints en S3: ≈ USD 0. Techo con
reintentos: < USD 3.

### D18. Orquestación (Step Functions, en serie, sin ramas ni pausas)

- Máquina Standard `sinca-deep-koopman`: aviso de inicio → F1 → B → F2 → F3 → selección → cierre →
  test → aviso final. Cada fase es un `Map` con `MaxConcurrency = 1` sobre la lista fija de jobs de
  la entrada, cada uno `arn:aws:states:::sagemaker:createTrainingJob.sync`, con aviso de fin de fase
  (jobs lanzados y saltados). Sin `Choice` de ramas ni `waitForTaskToken`: las reglas viven en
  `seleccion.py` y la máquina solo ejecuta.
- Entrada (la arma `scripts/koopman/preparar_ejecucion.py`): URI de la imagen por digest,
  `sourcedir.tar.gz` de `pipeline/` en `_mlflow/code/deep-koopman/<sha>/`, ARN del rol de training,
  `campaña` (id estable entre ejecuciones), la lista de D17 con `clave` estable
  `<fase>-<enc><var>-s<seed>`, nombre único `sinca-koop-<clave>-<id de ejecución corto>` (≤ 63),
  hiperparámetros (`sagemaker_program = koopman/run.py`, `sagemaker_submit_directory`), `max_run`,
  `CheckpointConfig` (D9) y `modo` (`real` | `seco`). F2/F3 resuelven su padre como el run de F1/F2
  con el mismo encoder y semilla en la misma campaña (marcador de D2).
- **Salto por job**: dentro del `Map`, antes de cada job, la Lambda (`estado_job`) hace `HeadObject`
  del marcador de D2 y un `Choice` de dos salidas (saltar o lanzar) decide; es el único `Choice`. El
  marcador se prefiere a consultar MLflow: la Lambda no necesita el sqlite (~19 MB) y el marcador
  solo existe si el run se publicó con sus artefactos.
- **Lambda reducida** a `estado_job` y `resumen` (lee un JSON del prefijo de orquestación y devuelve
  su texto para SNS; usado para selección, cierre y test); se quita el modo `rama`. Se mantiene en
  vez de usar `aws-sdk:s3` desde la máquina: separa 404 de 403 sin dar lectura S3 al rol de Step
  Functions.
- Avisos: tópico SNS `sinca-deep-koopman-avisos` con suscripción email (variable `operator_email`,
  `sensitive = true`: local en `infra/terraform.tfvars`, gitignored; en CI,
  `-var="operator_email=${{ secrets.OPERATOR_EMAIL }}"` en `infra-cd.yml`, enmascarado en los logs
  públicos de Actions; placeholder en `.example`). Mensajes: inicio, fin de cada fase, selección
  elegida (cifras clave), cierre y test (veredicto D12). Fallos: regla EventBridge sobre
  `Step Functions Execution Status Change` (`FAILED`, `TIMED_OUT`, `ABORTED`) → SNS.
- Reintentos: `Retry` solo en `SageMaker.AmazonSageMakerException` y
  `SageMaker.ResourceLimitExceededException` (2 intentos, 5 min, backoff 2). Un job `Failed`
  (`States.TaskFailed`, error de código) no se reintenta: la ejecución falla y avisa.
- Reanudar tras un fallo: nueva ejecución con la misma `campaña`; salta los jobs con marcador y el
  job interrumpido retoma desde su último checkpoint (D9).

### D19. Infra e IAM (Terraform)

- `infra/orquestacion-koopman.tf` (+ `infra/lambda/decision_rama/handler.py`, empaquetada con
  `archive_file` con `output_file_mode = "0666"` para un zip reproducible entre Windows/WSL y el
  Linux de CI; provider `hashicorp/archive`): state machine (ASL en
  `infra/stepfunctions/deep-koopman.asl.json` con `templatefile`), Lambda, SNS + suscripción y
  política del tópico (publica EventBridge), regla EventBridge, roles. Ya aplicado (7.6).
- **Próximo apply** (tras esta enmienda): actualiza en sitio la definición de la máquina y el código
  de la Lambda (nombre y recurso iguales, para no recrearla); sin recursos nuevos ni destruidos y
  sin cambios de IAM. Plan esperado: 2 cambios en sitio.
- Rol de Step Functions: `sagemaker:CreateTrainingJob`, `DescribeTrainingJob`, `StopTrainingJob`
  (y `AddTags` si se etiquetan) sobre `training-job/sinca-koop-*`; `iam:PassRole` solo a
  `sinca-training-role` con `iam:PassedToService = sagemaker.amazonaws.com`; `events:PutTargets`,
  `PutRule`, `DescribeRule` sobre `rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule`;
  `lambda:InvokeFunction` sobre la Lambda; `sns:Publish` sobre el tópico. Sin logging a
  CloudWatch de la máquina (basta el historial de ejecuciones).
- Rol de la Lambda: `s3:GetObject` en `sinca-mlflow/model-artifacts/orquestacion/deep-koopman/*`,
  `s3:ListBucket` sobre `sinca-mlflow` con condición `s3:prefix` = ese prefijo (para que un marcador
  ausente dé 404 y no 403) y logs de su log group (retención 30 días).
- Rol de training: sin cambios. Ya tiene `Get/PutObject` en `model-artifacts/*` (selección,
  marcadores y checkpoints) y `ListBucket` sobre el bucket, que pide la sincronización de
  checkpoints.
- `sinca-github-infra-role`: acciones explícitas de `states`, `lambda`, `sns`, `events` y las de
  IAM ya presentes; la condición `iam:PassedToService` de su `iam:PassRole` suma
  `states.amazonaws.com` y `lambda.amazonaws.com`.
- Flujo: `terraform plan` en el PR de la rama (infra-cd, con el secret `OPERATOR_EMAIL`); un solo
  `terraform plan` + `apply` local del operador revisando el plan. Tras la ejecución seca exitosa,
  **PR solo de infra** (`infra/**`, `infra-cd.yml`, Lambda y ASL) a `main`: su plan de CI debe
  mostrar 0 cambios y el apply de `main` queda no-op. El resto del change se mergea al cierre.

### D20. Ejecución seca

Una sola prueba, con 1 semilla (`preparar_ejecucion.py --modo seco` usa `seeds = [0]` por defecto):
la lista fija de la semilla 0 (F1 4, B 5, F2 2, F3 4) más selección, cierre y test, todos en modo
seco (`--dry`: sin MLflow ni entrenamiento; selección, cierre y test escriben JSON sintéticos), con
`--dry-ckpt` (4 épocas sintéticas con checkpoints reales) en F1 y `--fallar-en f1:3:2` (el job 3 de
F1 falla tras guardar la época 2). Primera ejecución: `FAILED` sin reintento y aviso de EventBridge.
Segunda ejecución con la misma `campaña`: salta los jobs 1–2 de F1 (marcador), el job 3 retoma en la
época 3 (`epochs_run = 4`) y la campaña sigue hasta el final. Se verifican los avisos de inicio, fin
de cada fase, selección, cierre, test y fallo. ~40 min, ~USD 0.1.

## Risks / Trade-offs

- [Entrenamiento inestable por rollout de 24 pasos] → techo de `ρ`, currículo, clipping, calentamiento.
- [Sin pausas, un error de diseño se detecta al final] → la campaña cuesta ~USD 1.5; el operador
  revisa los avisos de fin de fase y puede detener la ejecución (`stop-execution`) y reanudarla.
- [Selección sobre muchas candidatas con `sel` corto] → orden de simplicidad fijo y reemplazo solo
  con mejora significativa; test una sola vez sobre la elegida.
- [Arranque asimétrico GRU/conv1d] → anotado; ambos llegan a F3 y compiten en la selección.
- [Bootstrap por DMD sobre el latente ≠ incertidumbre del entrenamiento] → la estabilidad entre
  semillas es la prueba principal; el bootstrap es complemento.
- [Escritura concurrente del sqlite] → `Map` con `MaxConcurrency = 1` y D2.
- [Marcador escrito y run no publicado, o al revés] → el marcador se escribe solo después de
  publicar; un run publicado sin marcador se repite y deja un run duplicado, que se marca
  `descartado`.
- [IAM nuevo (roles de Step Functions y Lambda, ampliación del rol OIDC de infra)] → riesgo alto
  (`WORKFLOW.md`): mínimo privilegio por ARN/prefijo, apply del operador con el plan a la vista,
  relectura completa del change al cierre.
- [Destrucción por CI: con los recursos aplicados desde la rama, si otro PR con `infra/**` se
  mergea a `main` antes que el PR de infra, el apply automático de `main` los destruiría] → no
  mergear otros PR de `infra/**` entre el apply local y el PR solo de infra; ese PR va apenas pasa
  la ejecución seca.
- [Email del operador en logs públicos] → variable `sensitive` y secret de Actions.

## Migration Plan

Código y tests locales → infra (plan en CI, apply local del operador) → humo manual → ejecución
seca (D20) → PR solo de infra a `main` (0 cambios) → campaña real: F1 → B → F2 → F3 → selección →
cierre → test → aviso final. Rollback: runs con `stage=deep-koopman` se marcan `descartado` o se
borran con `scripts/mlflow/maintenance.py`; respaldos en `_mlflow/backups/`.
