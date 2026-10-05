## Context

Fase 3 del proyecto `forecast-sinca-aws` (predicción ICAP zona saturada Temuco/PLC, horizonte 24h). Fases 1-2 cerradas: buckets `sinca-data`/`sinca-mlflow`, SageMaker Studio, MLflow self-hosted (Fargate+ECR), y pipeline de datos que produce las 6 series horarias validadas en `sinca-data/validated/`. Este change construye el primer modelo (baseline) sobre esos datos.

Decisiones ya cerradas en `project-decisions.md`: cómputo SageMaker Training Job (CPU `ml.m5.large` tabular), tracking MLflow self-hosted (logging directo como modo principal), 6 series target (MP2.5/MP10 × 3 estaciones), ICAP_zona derivado en post-proceso.

## Goals / Non-Goals

**Goals:**
- Persistencia baseline (naive) como piso de referencia.
- LightGBM: 6 modelos independientes, RMSE ponderado por serie.
- Feature engineering completo (lags + meteorología + Fourier + calendario).
- Evaluación con ventana deslizante de tamaño fijo + test final held-out.
- Logging a MLflow (directo) con las métricas del prompt §5.

**Non-Goals:**
- Autoencoder + DMD/Koopman (Fase 4).
- SageMaker Pipeline de entrenamiento (Fase 4+; `pipeline-cd.yml` sigue skeleton).
- Features cruzadas entre estaciones (hipótesis Fase 4).
- Inferencia productivizada / reentrenamiento automático (Fase 6).

## Decisions

### 1. Target y horizonte — promedio móvil 24h a t+24h (single-step)

Cada modelo predice `mean(MP(t+1 … t+24h))` de su serie. El ICAP se deriva en post-proceso de ese promedio (es lo que consume la fórmula oficial). Single-step: 6 targets, no 144.

- **Alternativa considerada (multi-step horario t+1…t+24h)**: más granular, pero 24× más targets, acumula error en 24 pasos, y LightGBM no hace multi-step nativo. Descartada para el baseline; la granularidad horaria queda para Fase 4+ (AE+DMD).
- **Alternativa considerada (horaria single-step a t+24h)**: no sirve para ICAP, porque el promedio 24h necesita t+1…t+23, que no se tienen al predecir 24h adelante. Descartada.

### 2. Features

- **Lags de la serie propia**: `t-1h`, `t-24h`, `t-48h`, `t-72h`, `t-168h`.
- **Meteorología** (misma estación, lag `t-24h`): temp, humedad, presión, viento (dir/vel), radiación.
- **Altura de capa límite** (Open-Meteo): valor + lag `t-24h`.
- **Feriados**: flag binario (nacional + regional La Araucanía).
- **Día de semana**: cyclic (sin/cos).
- **Ventana GEC**: flag binario (1 abr–15 sep).
- **Fourier** (sin/cos, para que LightGBM capture periodicidad que los árboles no ven nativamente): diaria (24h), semanal (168h), anual (365.25 días).

### 3. Partición — ventana deslizante de tamaño fijo

- **Test final held-out**: `2026-07-01 → 2026-10-04` (nunca se toca en tuning).
- **Rolling origin** sobre `2021-01-01 → 2026-06-30`: train 2 años, valid 1 mes, step 1 mes (~40 ventanas).
- **Métrica reportada**: promedio ± desviación sobre ventanas + métrica del test final (número "oficial").

Justificación: 2 años captura la estacionalidad anual sin arrastrar dinámicas obsoletas (las medidas de mitigación cambian la dinámica, y un histórico muy largo introduce más ruido que señal).

### 4. Loss — RMSE ponderado por serie

`RMSE_ponderado = sqrt(Σ w_i·(y_i−ŷ_i)² / Σ w_i)`, con peso continuo `w_i = 1 + α·max(0, y_i − umbral_regular)`.

- Default: `α=1`, `umbral_regular` = umbral ICAP 100 (MP10 150 µg/m³, MP2.5 50 µg/m³).
- Motivo: las categorías críticas (alerta/preemergencia/emergencia) están desbalanceadas; un MSE plano optimiza los días normales e ignora los episodios raros que son el objetivo del proyecto.

### 5. Métricas a loguear en MLflow

- `train_loss` (RMSE ponderado), `mse`/`rmse` por serie (sin ponderar).
- Bloque ICAP oficial (sobre `ICAP_zona` post-proceso): `recall_alerta`, `recall_preemergencia`, `recall_emergencia`.
- Bloque percentil propio: `recall_top10pct`, `recall_top5pct`, `pr_auc_top10pct`.

### 6. Tracking — MLflow logging directo (modo 1)

Sin servidor: el Training Job importa `mlflow`, loguea a SQLite local, sube `mlflow.sqlite` + `mlruns/` a `sinca-mlflow/_mlflow/` al terminar. Convención `sinca-{estacion}-{modelo}-v{n}`. La persistencia baseline se reporta como referencia (no es un run de entrenamiento).

### 7. Cómputo — SageMaker Training Job

Imagen ECR con runtime LightGBM (`lightgbm`, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `boto3`, `pyarrow`). Código vía `SourceDir`/`entry_point` desde `pipeline/` (patrón compute). CPU `ml.m5.large`. Disparo manual (no automatizado en Fase 3).

### 8. IAM — finalizar `sinca-training-role`

Hoy skeleton. Se completa: S3 read `sinca-data/validated/*`, r/w `sinca-mlflow/_mlflow/*` (logging) y `sinca-mlflow/model-artifacts/*` (modelos), logs `/aws/sagemaker/*`. Sin permisos IAM ni managed policies amplias.

### 9. Hiperparámetros LightGBM — random search + early stopping

**Estrategia**: random search (no grid ni Optuna) + early stopping sobre el set de validación. Budget ~20 trials por serie. Selección por RMSE ponderado (promedio sobre ventanas de validación).

**Búsqueda por serie** (6 búsquedas independientes), porque MP2.5 y MP10 tienen dinámicas distintas, y las 3 estaciones también.

**Espacio de búsqueda**:

| Parámetro | Valores a probar | Default |
|---|---|---|
| `n_estimators` | early stopping (cap 1000) | early stopping |
| `learning_rate` | `[0.01, 0.05, 0.1]` | 0.05 |
| `num_leaves` | `[31, 63, 127]` | 63 |
| `min_child_samples` | `[20, 50, 100]` | 50 |
| `subsample` | `[0.7, 0.8, 1.0]` | 0.8 |
| `colsample_bytree` | `[0.7, 0.8, 1.0]` | 0.8 |
| `reg_alpha` | `[0, 0.1, 1.0]` | 0.1 |
| `reg_lambda` | `[0, 0.1, 1.0]` | 0.1 |

Racional: `num_leaves` acotado + `min_child_samples` alto = regularización fuerte (dataset chico, riesgo de overfitting en series temporales). `subsample`/`colsample_bytree` < 1.0 = bagging/feature sampling estándar. `reg_alpha`/`reg_lambda` leves.

**Logging**: solo el mejor run por serie (`sinca-{estacion}-lightgbm-v1`); el detalle de la búsqueda va en `EXPERIMENT-LOG.md` (convención del prompt §6).

## Risks / Trade-offs

- **Pocos casos extremos en el histórico** → el bloque ICAP oficial puede ser inestable (pocos días de emergencia). Mitigación: el bloque percentil propio (top 5/10%) es el decisor principal; el ICAP oficial es reporte comparativo.
- **Ventana deslizante = ~40 retrains** → LightGBM en CPU es rápido (segundos a minutos por ventana); costo acotado. Si resulta lento, bajar step a 3 meses.
- **SQLite sin escritura concurrente** → una sola corrida de entrenamiento a la vez (serializar la subida final a S3).
- **Scraping frágil ya resuelto en Fase 2** → los datos validados son la entrada; si la calidad cae, la validación pandera ya lo detecta.

## Migration Plan

1. Escribir feature engineering + persistencia + entrenamiento + evaluación (`pipeline/`).
2. Escribir imagen de entrenamiento (`docker/`) y Terraform (ECR + IAM).
3. Build + push imagen; `terraform plan`/`apply`.
4. Correr Training Job manual (baseline completo) y verificar métricas en MLflow.
5. Rollback: `terraform destroy` de los recursos nuevos (no toca buckets ni Fases 1-2).

## Open Questions

- **α y umbral del RMSE ponderado**: default `α=1`, `umbral_regular` = ICAP 100 (150/50). El prompt lo deja "a evaluar, no cerrado" — se ajusta tras ver los primeros resultados.
- **Step de la ventana deslizante**: 1 mes (default) vs 3 meses (más liviano). Se arranca con 1 mes.
