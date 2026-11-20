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

**Goals:** saber si un Koopman entrenado da modos identificables entre semillas (compuerta tras F1)
sin perder la precisión de 7.1; reportar D12 e interpretabilidad del §9.

**Non-Goals:** `A₂₄`, fbDMD y recorte; meteo exógena en el horizonte; K condicionado (LPV) y red de
autovalores (→ `dmd-localizado`); meteo futura; híbridos; grillas de `d`, `L`, mezcla de bloques o
λ (salvo B1); ensamble y encoder lineal salvo decisión en la pausa C; cómputo local de experimentos;
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
(aportes, dominantes, emparejamiento, bootstrap, compuerta), `evaluar.py` (rollout con `u`, decode,
Duan, arnés de `operador`), `compuerta.py` (reglas → JSON de decisión, D14/D18), `run.py` (CLI
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
  latente inicial de train a las ventanas MP. Penalización `‖WᵀW − I‖²_F` (W 144×10).
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
  al inicio de la fase. λ_lin = 1, λ_pred = 1, λ_tgt = 1, λ_orth = 0.1, fijos. B2: λ_cons = 1. B3:
  λ_L1 = 0.1.
- F1 = rec + lin (+ orth). F2 = F1 + pred. F3 = F2 + tgt.
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

### D11. Identificabilidad (preregistro)

- Contribución del modo `j` en `t`: `c_j(t)` = parte del promedio decodificado de 24 h (espacio log1p
  normalizado) que viene del bloque `j` de `z_{t+24|t}` (decoder lineal: exacta; GRU: jacobiano del
  decoder en `z_{t+24|t}`). `B·u` se reporta aparte.
- Aporte `A_j = Σ_t Σ_s w_s(t)·c_{j,s}(t)²` en `sel`. Dominantes = mínimo conjunto, en orden de `A_j`,
  con ≥ 80 % de `Σ_j A_j`. El diario no se exige como dominante (el promedio lo cancela); se reporta.
- Bootstrap: DMD exacto sobre el latente aprendido de train, bloques de 7 días, B = 200,
  emparejado con `K` (D8 de 7.1: húngaro sobre representantes, ±25 %, reales solo vida media).
- Compuerta: cada dominante de cada semilla tiene pareja en las otras dos, aparece en ≥ 80 % del
  bootstrap de cada semilla y sus intervalos de período y vida media se solapan entre semillas.

### D12. Remedios

- **B1 (decisión del operador)**: barrido λ_lin ∈ {0.1, 10} en F1 del GRU (3 semillas × 2), solo si
  la compuerta falla y `L_lin` normalizado al final de F1 > 10× `L_rec`. Si B1 aprueba, se detiene;
  si no, B2–B4 siguen con λ_lin = 1. Si no hay desbalance, B1 no se corre.
- Orden: B2 (`L_cons = ‖enc([dec(z), meteo, máscaras]) − z‖²`), B3 (`L1 = mean_t Σ_bloques ‖z_bloque‖₂`),
  B4 (`d = 8`: diario + 2 libres + 2 reales). Cada uno sobre F1 del encoder principal, 3 semillas;
  se detiene en el primero que aprueba.
- Si B4 falla: pausa de la máquina (D18); el operador elige C1 (encoder lineal Hankel/PCA con `K` y pérdidas entrenados) o C2
  (ensamble de semillas para precisión; interpretabilidad solo de modos robustos). Cualquiera
  requiere tareas nuevas (vuelve al arquitecto).

### D13. Eliminación de modos

Regla 1.6 enmendada sobre `K` entrenado: quitar un bloque = anular su fila/columna en `K`, su fila
en `B` y su contribución al decode, sin reentrenar; Duan reestimado si aplica. Se reporta en cada
fase; los bloques finales se fijan tras F3 (o la última fase corrida).

### D14. Decisiones por fase

Un job de compuerta (`--fase compuerta --de <fase>`, solo evaluación) aplica las reglas (D9 de 7.1,
D11, D12) y escribe la decisión en S3 (D18): precisión = diferencia emparejada (cuenta solo si el
intervalo excluye 0); empate → la opción más simple (decoder lineal < GRU; estructurado < denso;
GRU < conv1d; sin `icap` < con `icap`). Rama `pausa` ante ambigüedad o cambio de diseño. El
implementador copia las decisiones a `koopman/decisiones.json` al cerrar (registro versionado).

### D15. Runs

| Run | Nombre |
|---|---|
| Fase/variante | `sinca-zona-koop-{enc}-{fase}{var}-aedmd-v{n}-s{seed}`; `enc` ∈ `gru`, `conv1ds`; `fase` ∈ `f1`, `b1`, `b2`, `b3`, `b4`, `f2`, `f3`, `cierre`; `var` ∈ ``, `kden`, `perlib`, `sinbu`, `decgru`, `ll01`, `ll10`, `icap` |
| Test por serie | `sinca-{slug}-{pollutant}-koop-{enc}-aedmd-v{n}-s{seed}` (contrato LightGBM, `evaluacion=test`) |

Artefactos: `encoder.keras`, `decoder.keras`, `koopman.npz` (parámetros de bloque, `K`, `B`),
`latent.parquet`, `predictions.parquet`, diagnósticos de 7.1, `modos.parquet` (aportes, dominantes),
`bootstrap_modos.parquet`, `coherencia.parquet`, curvas de pérdida. `version = 1`.

### D16. Cierre y test

Encoder y decoder congelados; `K`, `B` (y Duan si el final es F1/F2) se reentrenan 20 épocas con la
pérdida de la fase final sobre train+es+sel (hasta 2026-06-30), con los bloques fijados en D13. Un
job de test aparte, tras pausa del operador; guardia de una sola evaluación (`--force-test`).
Referencias: `lightgbm-v1` holdout, persistencia y test de GRU+DMD+Duan de 7.1.

### D17. Jobs y costo

Tarifa ~USD 0.115/h (`ml.m5.large`, no verificada en consola). Tiempos estimados desde v2 (GRU 60
épocas ~40 min) × ~2 por `L_lin` (dos pasadas de encoder).

| Grupo | Jobs | h c/u | h | USD |
|---|---|---|---|---|
| Humo | 1 | 0.3 | 0.3 | 0.03 |
| F1 GRU lineal / conv1d / GRU+decGRU | 3 / 3 / 3 | 1.3 / 0.5 / 1.7 | 10.5 | 1.21 |
| F1 variantes K (3 por job) | 3 | 4.0 | 12.0 | 1.38 |
| F2, F3 (GRU; ×2 si sigue conv1d) | 3–6, 3–6 | 0.9, 0.7 | 4.8–9.6 | 0.55–1.10 |
| F3 variante `icap` | 3 | 0.7 | 2.1 | 0.24 |
| Cierre + test | 2 | 0.5, 0.3 | 0.8 | 0.09 |
| Compuertas (F1, F2, F3; solo evaluación) | 3 | 0.2 | 0.6 | 0.07 |
| **Base** | **27–33** | | **31–36** | **~3.6–4.1** |
| Contingente B1 (λ_lin) + compuerta | 0–7 | 1.3 / 0.2 | 0–8.0 | 0–0.92 |
| Contingentes B2–B4 + compuertas | 0–12 | 1.3 / 0.2 | 0–12.3 | 0–1.42 |
| **Máximo** | **52** | | **~56** | **~6.5** |
| Ejecución seca (D20, 2 pruebas, aparte) | ~35 | 0.1 | ~3.5 | ~0.40 |

Orquestación (Step Functions Standard, Lambda, SNS, EventBridge) y checkpoints en S3 (KB a pocos MB
por job): ≈ USD 0. Los checkpoints abaratan los reintentos (no se repiten épocas hechas). Techo con reintentos: < USD 8. `max_run`: humo 1 h; F1 3 h (decGRU 4 h; variantes 6 h); F2/F3 3 h.

### D18. Orquestación (Step Functions, en serie)

- Máquina Standard `sinca-deep-koopman`: F1 → compuerta → `Choice` {pasa → F2; `b1` → B1 → compuerta;
  `b2` → B2 → compuerta → … B4; `pausa`; `c` → pausa C} → F2 → compuerta → F3 (+ `icap`) →
  compuerta → cierre → pausa antes de test → test → aviso final. Sin paralelismo: cada fase es un
  `Map` con `MaxConcurrency = 1` sobre la lista de jobs de la fase, cada uno
  `arn:aws:states:::sagemaker:createTrainingJob.sync`.
- Entrada de la ejecución (la arma `scripts/koopman/preparar_ejecucion.py`): URI de la imagen por
  digest, `sourcedir.tar.gz` de `pipeline/` subido a `_mlflow/code/deep-koopman/<sha>/`, ARN del rol de
  training, jobs por fase (D17) con hiperparámetros (`sagemaker_program = koopman/run.py`,
  `sagemaker_submit_directory`), `max_run`, `modo` (`real` | `seco`), `campaña` (id estable entre
  ejecuciones). Cada job tiene una `clave` estable `<fase>-<enc><var>-s<seed>` y un nombre único
  `sinca-koop-<clave>-<id de ejecución corto>` (≤ 63), más su `CheckpointConfig` (D9).
- Decisiones: cada job de compuerta escribe `model-artifacts/orquestacion/deep-koopman/<campaña>/<fase>.json`
  (`rama`, ganadores por semilla con `run_id`, cifras clave). Prefijo dentro de lo que el rol de
  training ya escribe: sin cambiar su IAM. Una Lambda chica (Python, ~20 líneas) lee el JSON,
  valida `rama` contra la lista permitida y la devuelve con el resumen; no aplica reglas. F2/F3
  resuelven su `parent_run_id` desde ese JSON.
- Pausas: `arn:aws:states:::sns:publish.waitForTaskToken` (sin Lambda) con el comando de aprobación
  `aws stepfunctions send-task-success --task-token <token> --task-output '{"rama": "..."}' --profile <AWS_PROFILE>`
  (o `send-task-failure` para abortar); `TimeoutSeconds` 7 días. En: rama `pausa` (ambigüedad o
  cambio de diseño), fallo tras B4 (opciones C1/C2: la ejecución termina con "requiere tareas
  nuevas") y siempre antes de test. Sin link de aprobación por Lambda URL: una URL pública con el
  token como única autenticación no es segura y con `AWS_IAM` no es simple.
- Avisos: tópico SNS `sinca-deep-koopman-avisos` con suscripción email (variable `operator_email`,
  `sensitive = true`: local en `infra/terraform.tfvars`, gitignored; en CI,
  `-var="operator_email=${{ secrets.OPERATOR_EMAIL }}"` en `infra-cd.yml`, enmascarado en los logs
  públicos de Actions; placeholder en `.example`). Mensajes: resultado de cada compuerta
  (rama, puntaje vs 7.1, compuerta de identificabilidad), cada pausa (con el comando), cierre y
  test (veredicto D12). Fallos: regla EventBridge sobre `Step Functions Execution Status Change`
  (`FAILED`, `TIMED_OUT`, `ABORTED`) de esta máquina → SNS.
- Reintentos: `Retry` solo en `SageMaker.AmazonSageMakerException` y
  `SageMaker.ResourceLimitExceededException` (2 intentos, 5 min, backoff 2). Un job `Failed`
  (`States.TaskFailed`, error de código) no se reintenta: la ejecución falla y avisa.
- **Salto por job** (en toda ejecución): dentro del `Map`, antes de cada job, la Lambda (modo
  `estado_job`) hace `HeadObject` del marcador de D2 y un `Choice` salta el job si existe. Se eligió
  el marcador sobre consultar MLflow: la Lambda no necesita el sqlite (~19 MB) ni MLflow, y el
  marcador solo existe si el run se publicó con sus artefactos. Las compuertas también tienen
  marcador (su JSON de decisión) y se saltan igual.
- Reanudar tras un fallo: nueva ejecución con la misma `campaña`; salta los jobs y compuertas con
  marcador y el job interrumpido retoma desde su último checkpoint (D9). `desde = <fase>` queda
  como atajo opcional.

### D19. Infra e IAM (Terraform)

- `infra/orquestacion-koopman.tf` (+ `infra/lambda/decision_rama/handler.py`, empaquetada con
  `archive_file` con `output_file_mode = "0666"` para un zip reproducible entre Windows/WSL y el
  Linux de CI, sin diffs perpetuos de `source_code_hash`; provider `hashicorp/archive` nuevo): state machine (ASL en
  `infra/stepfunctions/deep-koopman.asl.json` con `templatefile`), Lambda, SNS + suscripción y
  política del tópico (publica EventBridge), regla EventBridge, roles.
- Rol de Step Functions: `sagemaker:CreateTrainingJob`, `DescribeTrainingJob`, `StopTrainingJob`
  (y `AddTags` si se etiquetan) sobre `training-job/sinca-koop-*`; `iam:PassRole` solo a
  `sinca-training-role` con `iam:PassedToService = sagemaker.amazonaws.com`; `events:PutTargets`,
  `PutRule`, `DescribeRule` sobre `rule/StepFunctionsGetEventsForSageMakerTrainingJobsRule`;
  `lambda:InvokeFunction` sobre la Lambda; `sns:Publish` sobre el tópico. Sin logging a
  CloudWatch de la máquina (basta el historial de ejecuciones).
- Rol de la Lambda: `s3:GetObject` en `sinca-mlflow/model-artifacts/orquestacion/deep-koopman/*`,
  `s3:ListBucket` sobre `sinca-mlflow` con condición `s3:prefix` = ese prefijo (para que un marcador
  ausente dé 404 y no 403) y logs de su log group (retención 30 días).
- Rol de training: sin cambios. Ya tiene `Get/PutObject` en `model-artifacts/*` (decisiones,
  marcadores y checkpoints) y `ListBucket` sobre el bucket, que pide la sincronización de
  checkpoints.
- `sinca-github-infra-role`: acciones explícitas de `states`, `lambda`, `sns`, `events` y las de
  IAM ya presentes, para que el `plan` de CI refresque los recursos nuevos; la condición
  `iam:PassedToService` de su `iam:PassRole` suma `states.amazonaws.com` y `lambda.amazonaws.com`
  (hoy solo `ecs-tasks` y `sagemaker`).
- Flujo: `terraform plan` en el PR de la rama (infra-cd, con el secret `OPERATOR_EMAIL`); un solo
  `terraform plan` + `apply` local del operador (profile admin) revisando el plan; confirmar la
  suscripción del email. Tras la ejecución seca exitosa, **PR solo de infra** (`infra/**`,
  `infra-cd.yml`, Lambda y ASL) a `main`: su plan de CI debe mostrar 0 cambios y el apply de `main`
  queda no-op. El resto del change se mergea al cierre.

### D20. Ejecución seca

`modo = seco`: cada job corre `koopman/run.py --dry` (sin MLflow, sin entrenar; ~5 min de arranque)
y las compuertas escriben la rama forzada en la entrada (`ramas_forzadas`). Una ejecución recorre:
F1 → `b1` → pasa → F2 → `pausa` (aprobación por comando) → F3 → cierre → pausa de test (aprobación) →
test → aviso final. Una segunda prueba usa `--dry-ckpt` (entrenamiento sintético de 4 épocas con
checkpoints reales) en F1 y `--dry-fallar-en 8:2` (el job 8 de F1 falla tras guardar la época 2):
la ejecución falla sin reintento y avisa por EventBridge; una nueva ejecución con la misma
`campaña` salta los jobs 1–7 (sin `CreateTrainingJob` en el historial), el job 8 retoma en la
época 3 (log y `epochs_run = 4`) y la campaña sigue. Se verifican los emails de cada paso.

## Risks / Trade-offs

- [Entrenamiento inestable por rollout de 24 pasos] → techo de `ρ`, currículo, clipping, calentamiento.
- [Tiempos de job subestimados] → `max_run` holgado; humo mide el tiempo por época y se reestima
  el costo antes de F1.
- [Arranque asimétrico GRU/conv1d] → anotado; conv1d pasa solo si iguala o supera.
- [Bootstrap por DMD sobre el latente ≠ incertidumbre del entrenamiento] → la estabilidad entre
  semillas es la prueba principal; el bootstrap es complemento.
- [Escritura concurrente del sqlite] → `Map` con `MaxConcurrency = 1` y D2.
- [Marcador escrito y run no publicado, o al revés] → el marcador se escribe solo después de
  publicar; un run publicado sin marcador se repite y deja un run duplicado, que se marca
  `descartado` (aviso en el resumen de la compuerta).
- [IAM nuevo (roles de Step Functions y Lambda, ampliación del rol OIDC de infra)] → riesgo alto
  (`WORKFLOW.md`): mínimo privilegio por ARN/prefijo, apply del operador con el plan a la vista,
  relectura completa del change al cierre.
- [Destrucción por CI: con los recursos aplicados desde la rama, si otro PR con `infra/**` se
  mergea a `main` antes que el PR de infra, el apply automático de `main` los destruiría] → no
  mergear otros PR de `infra/**` entre el apply local y el PR solo de infra; ese PR va apenas pasa
  la ejecución seca.
- [Email del operador en logs públicos] → variable `sensitive` y secret de Actions.
- [Token de tarea en un email] → solo llega al operador; no hay URL pública de aprobación.

## Migration Plan

Código y tests locales → infra (plan en CI, apply local del operador) → humo manual → ejecución
seca (D20) → PR solo de infra a `main` (0 cambios) → ejecución real: F1 → compuerta (→ B1–B4) → F2 → F3 → cierre → ⏸ test → aviso final. Rollback:
runs con `stage=deep-koopman` se marcan `descartado` o se borran con `scripts/mlflow/maintenance.py`;
respaldos en `_mlflow/backups/`.
