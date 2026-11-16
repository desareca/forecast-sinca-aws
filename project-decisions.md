# Predicción Calidad del Aire (SINCA) — Plan de arquitectura (contexto para OpenSpec)

**Fecha:** 2026-09-21
**Estado:** Fase 4 en progreso — `ae-reconstruccion` cerrado; `ae-dmd` replanificado en 4 changes (ver §8 y §9, actualizado 2026-11-15)

Este documento es la fuente de verdad de arquitectura del proyecto. Las
decisiones acá cerradas no se re-discuten salvo que cambien explícitamente, y
`AGENTS.md` obliga a leer este documento antes de generar cualquier artefacto
de OpenSpec.

---

## 1. Contexto y objetivo

Predicción de calidad del aire (MP2.5/MP10) en la zona saturada de
Temuco/Padre Las Casas (La Araucanía), horizonte **24 horas** (mismo que el
sistema oficial del MMA, para comparar contra ese benchmark implícito).

Target derivado **ICAP_zona**: máximo entre las 3 estaciones representativas
(Ñielol, Las Encinas, Padre Las Casas II), calculado por MP10 y MP2.5 con la
fórmula oficial y tomando el peor contaminante. El pipeline produce **6 series
continuas** (MP2.5 y MP10 × 3 estaciones) y el `ICAP_zona` se deriva siempre en
post-proceso. Detalle completo en `prompt-sinca-mlops.md`.

## 2. Cuentas AWS y entorno de desarrollo

Ver `PREREQUISITES.md`. Cuenta personal (Account ID `AWS_ACCOUNT_ID`), profile
`AWS_PROFILE` (nunca `default` ni un profile ajeno), región **`us-east-1`**.
Dev environment remoto según `modules/dev-environment/blueprint.md`.

## 3. Blueprints de dominio seleccionados

| Blueprint | ¿Aplica? | Parámetros de este proyecto |
|---|---|---|
| `compute` | Sí | Entrenamiento en SageMaker Training Job (CPU `ml.m5.large` tabular / GPU `ml.g4dn.xlarge` AE+DMD). **Cómputo de training confirmado (Fase 3):** Training Job on-demand, CPU `ml.m5.large`, imagen ECR `sinca-train` (solo runtime; el código viaja por `source_dir`/`entry_point` desde `pipeline/`), disparo manual. Cómputo del scraper: **Fargate on-demand** (decidido en Fase 2; imagen ECR `sinca-scraper`, task Fargate sin service). |
| `dev-environment` | Sí | SageMaker Studio Space `ml.t3.large`, EFS persistente, SSH over SSM, imagen JupyterLab fijada en Terraform (`sagemaker-distribution-cpu` + alias `3.2.0`); lifecycle config reinstala aws-cli si falta y clona el repo (pasos de conveniencia no bloqueantes, con log en EFS). |
| `data-quality` | Sí | pandera; convención `raw/`/`validated/`/`quarantine/`; dimensiones y umbrales se definen en Fase 2. |
| `tracking-experimentos` | Sí | MLflow self-hosted (SQLite+S3), prefijo `sinca-mlflow/_mlflow/`; logging directo sin servidor como modo principal. |
| `costos` | Sí | Checklist "¿qué queda corriendo?" por servicio; estimación mensual < USD 5 (Space apagable). |
| `cicd` | Sí | Terraform como código, remote state S3+DynamoDB, feature branches por change, OIDC GitHub Actions. |

### Naturaleza del proyecto

- **Tipo**: MLOps completo (ingesta → validación → entrenamiento → evaluación → productivización).
- **Stack ML**: tabular (LightGBM, scikit-learn) + red neuronal (autoencoder + DMD/Koopman, PyTorch/TF). Enfoque loss detallado en `prompt-sinca-mlops.md` (RMSE ponderado por serie; loss conjunta solo como variante experimental de AE+DMD).

## 4. Storage y datos

- Bucket **`sinca-data`**: `raw/` → `validated/` → `quarantine/` (Parquet).
- Bucket **`sinca-mlflow`**: `_mlflow/` (tracking: `mlflow.sqlite` + `mlruns/`) y `model-artifacts/` (modelos versionados).
- Target del pipeline: 6 series horarias MP2.5/MP10 (una por estación). Partición train/valid/test a definir en Fase 2/3.

## 5. IAM — mínimo privilegio

| Rol | Permisos acotados |
|---|---|
| `sinca-dev-role` (dev env / Studio) | S3 `sinca-data/*` + `sinca-mlflow/_mlflow/*` (r/w), logs `/aws/sagemaker/*`, acciones Studio mínimas (presigned url, describe, apps) |
| `sinca-training-role` (training) | S3 read `sinca-data/validated/*`, r/w `sinca-mlflow/model-artifacts/*`, logs `/aws/sagemaker/*` |
| OIDC infra (GitHub → Terraform) | Pemitos `plan`/`apply` acotados a los recursos gestionados (Fase 1b) |
| OIDC pipeline (GitHub → modelo) | Actualizar definición de pipeline, escribir artefactos, registrar modelos (Fase 1b) |

Nunca policies `*FullAccess` ni `AdministratorAccess`. Roles OIDC en Fase 1b.

## 6. Monitoreo y observabilidad

- CloudWatch Alarms sobre fallos de Training Job / pipeline (Fases 3+).
- Métricas de modelo y de calidad de datos → MLflow self-hosted, por corrida.
- Alerta selectiva (solo condiciones que requieren acción humana).

## 7. Ejecución y schedule

Nada automático en Fase 1. A partir de Fase 2: pipeline de datos disparado por
EventBridge Schedule diario a las **~01:00 hora local (America/Santiago)**,
para capturar el día anterior completo; inferencia diaria a 24h en Fase 6.

## 8. Estructura OpenSpec

1. `setup-infra-base` — buckets, SageMaker Studio Space, IAM base, remote state. **(completado)**
2. `setup-mlflow-ci-cd` — MLflow self-hosted (Fargate), OIDC + 3 workflows. **(completado)**
3. `data-pipeline` (Fase 2) — scraper SINCA/Open-Meteo/feriados + ICAP + pandera. **(completado)**
4. `baseline` (Fase 3) — persistencia + modelo simple (LightGBM lags). **(completado)**
5. `analisis-modelos` (Fase 3.5) — dashboard Streamlit + persistencia de diagnósticos por run. **(completado)**
6. `ae-reconstruccion` (Fase 4, parte 1) — autoencoders sobre las 6 series de MP con
   auxiliares al latente, compuerta de reconstrucción y primer pase de DMD global
   sin optimizar. **(completado)** AE de referencia: conv1d `L = 24`, `d = 16`.
7. `ae-dmd` (Fase 4, parte 2) — **se divide en 4 changes con compuertas** (ver §9,
   "Plan de `ae-dmd`"). El borrador original (proposal/design con PA-1…PA-14) se movió a
   `docs/referencia/ae-dmd-borrador/` y queda **solo como referencia**; lo que vale es §9.
   - 7.0 `ae-dmd-prep` — mantención y nueva partición (reentrenar conv1d y GRU).
   - 7.1 `dmd-operador-global` — oráculo, escalera de representaciones, picos,
     exógenas, horizonte, estimador y selección de modos (post-hoc, CPU).
   - 7.2 `dmd-localizado` — medición de variabilidad de la dinámica; operador
     condicionado al régimen meteo del día solo si la medición lo justifica.
   - 7.3 `deep-koopman` — Koopman de extremo a extremo; solo si lo lineal no cumple
     el criterio de éxito.

Deuda técnica registrada: migración del SDK de SageMaker v2 → v3 (change
`sagemaker-sdk-v3`, a planificar **antes de `productivizacion`**; v2 termina soporte el
2027-07-06).
8. `evaluacion` (Fase 5) — comparar baseline + referencia pública.
9. `productivizacion` (Fase 6) — reentrenamiento semanal + inferencia diaria.

## 9. Decisiones cerradas

- Región `us-east-1`; cuenta `AWS_ACCOUNT_ID`; profile `AWS_PROFILE`.
- Repo `desareca/forecast-sinca-aws`; OIDC con `sub` acotado al repo:
  `sinca-github-infra-role` permite cualquier ref (el `plan` corre en PRs; el
  `apply` queda gateado por `if: github.ref == 'refs/heads/main'`), y
  `sinca-github-pipeline-role` restringido a `refs/heads/main`.
- Ramas: feature branches por change (PR + plan/apply).
- Tracking y Model Registry: MLflow self-hosted (SQLite+S3); reabrir Registry en Fase 6.
- Space `ml.t3.large`, EFS, SSH over SSM.
- 6 series target (MP2.5/MP10 × 3 estaciones); ICAP_zona derivado en post-proceso.
- No Secrets Manager en Fase 1 (fuentes públicas HTTP).
- **Cómputo de `ae-reconstruccion` (2026-11-03)**: todo el change corre en CPU
  `ml.m5.large` (imagen `tf-cpu-v1`), sin GPU. **Desviación** de la fila `compute` del
  §3, que anticipaba GPU `ml.g4dn.xlarge` para AE+DMD. Motivo: con `L ∈ {24, 72}` y el
  GRU solo a `L = 24` ningún job pasa de ~3 h (medido en el humo), y la GPU exigiría una
  imagen nueva sin necesidad. `L = 168` queda fuera de la búsqueda (memoria). La GPU se
  reevalúa en `ae-dmd` si hace falta el GRU a `L ≥ 72`. Cuota `ml.g4dn.xlarge` = 1.
- **AE de referencia para `ae-dmd` (2026-11-10)**: conv1d, `L = 24`, `d = 16`
  (`d_aux = 4`), hidden 64, dropout 0.1, lr 0.003, batch 128, `w_aux` 0.1, `aux_input`
  on. Única configuración que aprueba la compuerta en 3 semillas, en valid y test. gru
  `L = 24` es exploratorio. Sin corrección del sesgo de `log1p` en `ae-reconstruccion`
  (se reabre en `ae-dmd`). Detalle en `EXPERIMENT-LOG.md` y `tasks.md` del change.
- **Notebooks de exploración (2026-11-10)**: matplotlib (el dashboard sigue en Plotly) y
  versionados **con outputs**, sin credenciales ni IDs de cuenta.

### Plan de `ae-dmd` (cerrado con el operador, 2026-11-12)

Reemplaza al borrador original, movido a `docs/referencia/ae-dmd-borrador/`
(proposal/design y sus PA quedan solo como referencia). Razonamiento completo en `docs/ae-dmd-propuesta-experimentos.md` (v5);
este bloque es lo vinculante y prevalece si hay diferencia.

**Diagnóstico de partida.** DMD global sobre el latente: error a 24 pasos 86 % de la
escala de z, RMSE ponderado 33.28 (LightGBM 28.17); pierde en los picos. La
reconstrucción ya subestima los picos (−22 µg/m³ en el tramo ≥100). Estructura actual:
K horario sobre ventanas solapadas 23/24; K también hace evolucionar `z_aux` (meteo y
calendario se "pronostican" con K); la meteo solo actúa de forma aditiva (no puede
cambiar la vida media de los modos); el calendario está comprimido con la meteo en
`z_aux` y no es separable.

**Información permitida.**
- Prohibido: meteo futura, observada o pronosticada.
- Permitido: todo lo pasado y todo lo determinista derivable de t.
- Meteo del horizonte: **persistencia diurna**, meteo(t+k) = meteo(t+k−24). No se
  congela el valor de t.
- Régimen del día (para condicionar la dinámica): agregados de las últimas 24 h
  (p. ej. BLH mínimo, viento medio).
- **Ciclo diario = modo de la dinámica** (K horario es el principal; A₂₄ directo solo
  como referencia de precisión). **Semana, feriados y temporada = control conocido**
  (calendario real de t+k en el rollout).

**Criterio de éxito (por serie, ambas condiciones).**
- Precisión: RMSE ponderado ≤ 1.05 × LightGBM en ≥ 4/6 series, mismas filas; en
  PLC II además skill > 0 vs persistencia. Razones con intervalo por bootstrap de
  bloques de días; una serie no cumple solo si todo el intervalo queda sobre 1.05.
- Interpretabilidad: ≤ 10 modos; modos dominantes presentes en las 3 semillas y en
  ≥ 80 % de las re-estimaciones bootstrap, con intervalos de período y vida media que
  se solapan.

**Partición de `ae-dmd`** (nueva; la valid actual de un mes se usó en el early stopping).

| Tramo | Fechas | Uso |
|---|---|---|
| Train | 2021-01 → 2026-02 | Entrenar AE y ajustar K (una sola vez) |
| Marzo (`es`) | 2026-03 | Sin uso para entrenar ni detener; solo diagnóstico (ver "Presupuesto fijo") |
| Validación de selección | 2026-04 → 2026-06 | Elegir el ganador de cada etapa |
| Test held-out | 2026-07-01 → 2026-10-04 | Solo confirma al final de cada change |

- Todos (Hankel, PCA, AE) se seleccionan en esa validación con K ajustado sobre el
  mismo train. El ganador se reajusta con datos hasta 2026-06 y se evalúa una vez en
  test. Los números de test ya conocidos (33.28, 31.09) no se usan para elegir.
- Rolling de 12 ventanas solo para Hankel/PCA (dato adicional) y para el modelo final
  (reentrenado por ventana).
- **Presupuesto fijo, sin early stopping (2026-11-15, tras `ae-dmd-prep`)**: detener con marzo
  cortaba muy pronto (mes de baja concentración, ~500–670 filas): AE en época 3–24 y LightGBM en
  41–145 rondas, peor que persistencia en `sel`. Desde `ae-dmd-prep` v2: AE con 60 épocas fijas;
  LightGBM con `num_boost_round` = árboles del booster v1 de cada serie (fuga leve conocida: el
  early stopping de v1 usó junio de 2026, a favor de LightGBM). Ningún presupuesto se elige con `sel`.
- Al reentrenar con esta partición, las cifras de referencia de `ae-reconstruccion`
  cambian algo: anotarlo en `EXPERIMENT-LOG.md`, no es regresión.

**Método.**
- Escalera de representaciones, cada peldaño agrega una sola cosa: Hankel (retardos
  lineales) → PCA de la ventana (PCA(12) sobre MP + PCA(4) sobre auxiliares, misma
  estructura que el AE) → conv1d y GRU L=24. Mismas entradas para todos. Pasan a las
  etapas siguientes a lo más 2 representaciones. El GRU deja de ser solo exploratorio
  para la dinámica; se descarta si su pronóstico varía mucho entre semillas.
- Screening secuencial con arrastre del ganador; sin factoriales. Sin grillas de d,
  L ni lookback: rango por SVD + Gavish–Donoho, modos por eliminación hacia atrás
  según el error a 24 h. d = 16 salvo que el análisis justifique 8.
- 3 semillas, con la semilla en el nombre del run.
- `pydmd` permitido **a prueba** para estimadores estándar (cambia "solo numpy" del
  borrador). El DMD exacto actual en numpy queda como control y ambos deben coincidir.
  `pydmd` no entra al entrenamiento TF: trabaja sobre latentes como arrays.
- Cómputo: CPU `ml.m5.large` en todos los changes salvo que `deep-koopman` demuestre
  lo contrario.

**Changes y compuertas.**
- `ae-dmd-prep`: reentrenar conv1d y GRU (3 semillas c/u) con la nueva partición;
  `predictions.parquet` de persistencia y LightGBM sobre la validación de selección;
  descarga selectiva en `analysis/load.py::ensure_state`; semilla en el nombre del run;
  limpieza de runs lightgbm v99; pase DMD para todas las semillas y el GRU; probar
  `pydmd` en la imagen (plan B: entorno aparte sobre latentes guardados).
- `dmd-operador-global` (post-hoc, sin entrenar redes), en orden:
  1. **Oráculo** (primer experimento): codificar la ventana real que termina en t+24,
     decodificar y promediar = techo de la representación. Si queda cerca o sobre
     LightGBM, los AE actuales salen y Hankel pasa a línea principal.
  2. Escalera: Hankel MP; Hankel MP+meteo; PCA; conv1d; GRU; ablaciones Hankel por
     estación vs conjunto y h = 48 (si gana claro, habilita conv1d L=72).
  3. Picos: DMD por mínimos cuadrados ponderados (peso del par = máx de w entre las 6
     series) y corrección de retransformación de `log1p` (smearing de Duan).
  4. Exógenas: `z_aux` deja de ser estado; en cada paso se re-codifica la ventana
     auxiliar con persistencia diurna y calendario real (z_mp,t+1 = K_mm·z_mp,t +
     K_ma·z_aux,t).
  5. Horizonte: K horario (principal) vs A₂₄ directo (referencia).
  6. Estimador: DMD exacto, clip |λ| ≤ 1−ε, fbDMD; incertidumbre por modo con
     bootstrap por bloques.
  7. Rango y modos sin grilla.
- `dmd-localizado`: siempre se mide primero la variabilidad (DMD por mes, por tercil
  de agregados meteo diarios, espectrograma). Localizar solo si la variación de vidas
  medias supera el ruido bootstrap. Modelos en orden: K_mm por temporada → por régimen
  meteo del día → **LPV** K_mm(u) = K₀ + Σ uᵢ·Kᵢ (candidata principal; ridge; u
  acotado al rango observado y radio espectral verificado). Las variables de la LPV
  salen de esta medición. La localización temporal (ventana deslizante) es solo
  diagnóstico, no modelo.
- `deep-koopman`: entra si el mejor lineal no cumple la precisión, o si el oráculo
  muestra techo de representación y Hankel tampoco alcanza. K entrenable global (no
  por batch) en bloques 2×2 de rotación con radio ≤ 1, con el condicionamiento heredado;
  calendario fuera del encoder, solo como control B·u; decoder lineal como variante
  principal (descomposición exacta por modo; techo de reconstrucción = PCA); arranque
  en caliente desde el AE. Pérdidas incrementales: F1 rec+lin → F2 +multi-paso 24 h →
  F3 +RMSE ponderado del promedio (variante ICAP_zona = PA-9). Diagnóstico de
  coherencia: DMD sobre el latente vs K aprendido.

**Interpretabilidad (entregable en cada change).** Espectro con período y vida media
e intervalo; formas de modo por estación y contaminante (fase = desfase entre
estaciones); descomposición de episodios por modo; estabilidad entre semillas;
matrices de forzamiento y, si hay LPV, curvas autovalor vs variable meteo.

**Fuera de alcance.** Híbridos DMD+LightGBM; clustering de regímenes latentes
(SLDS/HMM/k-means); localización temporal como modelo; meteo futura; grillas de
tamaño; LightGBM v2 y Holt-Winters/SARIMA (PA-11).

Ver `WORKFLOW.md` para el protocolo de coordinación entre Arquitecto,
Implementador y operador humano.
