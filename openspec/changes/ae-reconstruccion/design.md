## Context

Fase 4 (parte 1) de `forecast-sinca-aws`. Estado de partida:

- **Datos**: 6 series horarias validadas (MP2.5/MP10 × Ñielol, Las Encinas,
  Padre Las Casas II) + meteo SINCA por estación + BLH de Open-Meteo (un solo
  punto, Padre Las Casas II) + feriados. Histórico 2021-01-01 → 2026-10-04
  (~50 mil horas por serie). Dataset chico para estándares de redes neuronales.
- **Huecos medidos (2026-10-30)**: MP faltante 1.4-5.0% según estación; las 6
  series MP están presentes a la vez en el 90.2% de las horas. Las Encinas tiene
  cortes largos (~700 h). Meteo inutilizable: `glob` (todas), `rhum` (Las Encinas
  96%, PLC II 63%), `temp` de PLC II (63%). `rain` tiene buena cobertura y el
  baseline no la usa.
- **Baseline (Fase 3)**: persistencia + 6 LightGBM, target
  `mean(MP(t+1 … t+24h))`, RMSE ponderado, ventana deslizante 2a/1m/1m (42
  ventanas) + test held-out `2026-07-01 → 2026-10-04`. LightGBM pierde en PLC II.
- **Análisis (Fase 3.5)**: dashboard Streamlit que lee MLflow (SQLite+S3) sin
  servidor; agrupa por familia de modelo (`modelo`) y por versión, y por serie
  vía params `estacion`/`pollutant`.
- **Reutilizable**: `features/build.py` (carga de validados, partición,
  held-out), `evaluate/evaluate.py` (métricas, `zone_icap`), `train/mlflow_log.py`,
  `train/launch.py` (Estimator SDK v2), `train/train.py::_select_windows`.

Este change es el grupo 1 de la Fase 4. El grupo 2 (`ae-dmd`) trabaja la
dinámica (parámetros del DMD, localización, control, Koopman end-to-end) y se
planifica con los resultados de este.

## Goals / Non-Goals

**Goals:**
- Establecer, para varias arquitecturas de autoencoder, **qué tan bien
  reconstruyen las 6 series de MP** (en µg/m³, sin aplanar los picos), con
  métricas y una compuerta explícita.
- Que el latente incorpore información auxiliar (meteo, capa límite, calendario)
  con una loss de pesos separados, para dar contexto de dinámica sin obligar a la
  reconstrucción de MP a depender de ella.
- Entregar un **primer pase de DMD global estándar** (sin optimizar) sobre el
  latente congelado: mide linealizabilidad del latente y da un pronóstico
  informativo comparable con persistencia y LightGBM.
- Conservar la interpretabilidad como objetivo de primer orden: latente en
  bloques, decoder lineal como variante, ablación de `z_aux`.
- Mantener el patrón operativo: Training Job on-demand, disparo manual, logging
  directo a MLflow.

**Non-Goals:**
- Optimizar parámetros del DMD, DMD local/por regímenes (catálogo PA-14),
  DMD con control, Koopman end-to-end, loss conjunta ICAP (→ `ae-dmd`).
- Comparación contra la referencia pública MMA (Fase 5), orquestación (Fase 6).
- Nuevas fuentes de datos (pronósticos meteo históricos).
- LightGBM v2 y Holt-Winters/SARIMA.

## Decisiones heredadas (cerradas, no se re-discuten)

1. **Test held-out** `2026-07-01 → 2026-10-04`, nunca tocado en tuning.
2. **Target evaluado en el pase DMD**: `mean(MP(t+1 … t+24h))` por serie, con
   `ICAP_zona` en post-proceso (`evaluate.zone_icap`), mismas métricas que el
   baseline (`evaluate_predictions`, sin cambios) y evaluación sobre la
   intersección de filas con target válido de LightGBM.
3. **Cómputo**: SageMaker Training Job on-demand, disparo manual, código por
   `source_dir`/`entry_point` desde `pipeline/`, imagen ECR solo-runtime.
4. **Tracking**: MLflow logging directo (SQLite+S3, `sinca-mlflow/_mlflow/`),
   una corrida a la vez (SQLite sin concurrencia).
5. **IAM**: se reutiliza `sinca-training-role` (ya tiene pull acotado a
   `sinca-train`, S3 read `validated/*`, r/w `_mlflow/*` y `model-artifacts/*`).
6. **Región/profile**: `us-east-1`, `AWS_PROFILE` explícito.
7. **Framework**: TensorFlow + Keras (Keras 3 sobre backend TF).

## Decisions

### D1 — Datos de entrada: 6 series de MP como núcleo, auxiliares al latente

**Decisión**: el AE recibe y reconstruye únicamente `X_mp ∈ [L × 6]`
(pm25/pm10 × 3 estaciones, un único modelo conjunto). Los datos auxiliares
entran por una rama aparte y se mezclan en el latente:

| Grupo | Canales | Papel |
|---|---|---|
| MP (núcleo) | pm25, pm10 × 3 estaciones (6) | entra y se reconstruye |
| Meteo auxiliar | `pres`, `wspd`, `rain` × 3 estaciones + `temp` de Ñielol y Las Encinas (11) | entra a la rama aux; se reconstruye con peso `w_aux` |
| Capa límite | `blh` (1) | ídem meteo |
| Calendario | Fourier diaria/semanal/anual sin/cos, `dow_sin/cos`, `is_holiday`, `gec_window` (~10) | entra a la rama aux; **sin loss** |

Excluidos: `rhum` y `glob` (cobertura inútil), `temp` de PLC II (63% faltante),
gases (no usados). Open-Meteo es un solo punto para las 3 estaciones.

**Alternativas**: (i) apilar todo en una sola matriz con un encoder
(más simple, pero mezcla roles y no permite pesos de loss por rol); (ii)
reconstruir también la meteo en el mismo bloque (la meteo compite con MP por
capacidad del latente). Se eligió rama separada porque deja `w_aux` como una
perilla (0 = contexto sin obligación de representarla) y mantiene el MP como
objetivo. Control adicional: `aux_input` apagable (rama aux solo con calendario)
para medir el aporte de la meteo.

### D2 — Arquitectura: encoder por rol, latente en bloques, decoder MP ve todo

```
X_mp [L×6] ──► encoder_mp ──► z_mp ─┐
                                    ├─► z=[z_mp | z_aux] ─► decoder_mp  ─► X̂_mp [L×6]   (peso w_mp)
aux [L×(12+12 máscaras+~10 cal)]    │                  └─► decoder_aux ─► X̂_aux [L×12] (peso w_aux)
        ──► encoder_aux ──► z_aux ──┘
```

- **Latente `d`** ∈ {8, 16, 32}; `d_aux = d/4` por defecto (≥ 2), parámetro
  ajustable.
- **El decoder de MP ve todo el latente** (`z_mp` y `z_aux`): la meteo aporta
  contexto de dinámica a la reconstrucción de MP. Diagnóstico de aporte:
  poner `z_aux` en cero en evaluación y medir el empeoramiento del MP.
- **Variantes** (encoder/decoder): `mlp` (ventana aplanada), `conv1d`,
  `gru` (y `lstm` como variante interna de recurrente si hace falta) y
  `linear_decoder` (encoder no lineal, decoder lineal por rama: cada dimensión
  del latente tiene una forma fija sobre las series → interpretabilidad).
- Salida del decoder **lineal** (sin sigmoide/acotado), para no saturar picos.
- `L` ∈ {24, 72, 168} h como hiperparámetro de la búsqueda.

**Alternativas descartadas**: ramas fusionadas con capa densa opaca (pierde
atribución por bloque); decoder MP solo con `z_mp` (la meteo no daría contexto).

### D3 — Preprocesamiento

- **Grilla horaria completa** (como `build.py`), de modo que los huecos
  aparezcan como `NaN` y no se corran las series.
- **Huecos**: interpolación lineal de huecos de ≤ 6 h (MP y aux). Una ventana de
  MP con algún hueco > 6 h **se descarta** del entrenamiento y de la evaluación.
  En aux, hueco > 6 h → valor 0 (media en escala normalizada) + columna de
  máscara 0/1 como entrada. Se guarda siempre la máscara de **valores reales**:
  la loss y las métricas de reconstrucción se calculan **solo sobre valores
  reales** (no sobre interpolados).
- **Transformación**: `log1p` en `pm25`, `pm10`, `rain`, `blh`, `wspd`;
  z-score por canal con media/desviación calculadas **solo con el train** de
  cada partición. `temp`, `pres` solo z-score. Calendario sin transformar.
- **Recorte**: solo en aux, a ±5 desviaciones tras z-score. **MP no se recorta**
  (los picos son el objetivo).
- **Revisión de rangos**: antes de entrenar, tabla por canal con mínimo, máximo,
  percentiles y fracción recortada; confirma que ningún canal domina la loss por
  escala. Igualar escalas no iguala dificultad (p. ej. `rain` seguirá siendo
  difícil): por eso `w_aux` se barre.
- **Muestras**: ventana deslizante hora a hora, `X[t-L+1 … t]`. Una muestra
  pertenece a un conjunto solo si **toda su ventana** cae dentro de él
  (evita mezclar horas entre train/valid/test). Se espera ~17 mil muestras muy
  solapadas por ventana de 2 años: redes chicas, dropout, early stopping.
- **Medición previa**: cuántas ventanas sobreviven a la regla de descarte según
  `L` (primera tarea de ejecución). Si son muy pocas con `L = 168`, se revisa la
  política de huecos (alternativa: máscara de entrada en MP) y se registra como
  desviación.

### D4 — Loss con pesos separados

`loss = w_mp · E_mp + w_aux · E_aux`.

- `E_mp`: MSE enmascarado en espacio transformado, **promediado por canal** (no
  sumado) sobre los 6 canales de MP; `E_aux`: ídem sobre los 12 canales de
  meteo/blh. Promedio por canal para que 6 vs 12 canales no imponga un peso
  implícito.
- `w_mp = 1` fijo; `w_aux ∈ {0, 0.1, 0.5, 1}` como barrido sobre la mejor
  arquitectura (no en toda la búsqueda).
- El calendario no tiene término de loss (es determinista; reconstruirlo no
  aporta).
- Los pesos ponderados por magnitud del baseline (`w_i = 1 + α·max(0, y − 100)`)
  **no** se usan en la reconstrucción de este change; el sesgo hacia picos se
  controla con los diagnósticos de aplanamiento (D5). Se registra como pregunta
  abierta para `ae-dmd`.

### D5 — Compuerta de reconstrucción y diagnósticos de aplanamiento

La reconstrucción es una **condición necesaria**. Un AE solo es elegible para el
pase de DMD y para el ranking si pasa la compuerta (**dura sobre MP**; lo aux se
reporta, sin compuerta).

**Referencias** (corridas baratas, registradas como runs): predictor del
promedio por canal y **PCA con el mismo `d`** (el mejor AE lineal): un AE que no
supera a PCA(d) no justifica su no linealidad.

**Diagnósticos por serie de MP** (valores en µg/m³ tras deshacer la
transformación), calculados sobre valid (decisión) y test (reporte):
- razón de varianza `std(recon)/std(real)`;
- pendiente de la regresión real vs. reconstruido;
- espectro de potencia (atenuación de frecuencias altas);
- RMSE y sesgo por tramos de concentración (< 50, 50–100, > 100 µg/m³), captura
  de máximo y percentiles altos;
- RMSE de reconstrucción frente a las referencias.

**Diagnósticos del latente**: dimensión efectiva (razón de participación del PCA
de `z`), dimensiones muertas, y ablación de `z_aux`.

**Sesgo de `log1p`**: al deshacer la transformación, una red entrenada en espacio
log tiende a subestimar la media (desigualdad de Jensen), sobre todo en picos.
Se mide por separado del aplanamiento; si es relevante se evalúa una corrección
de sesgo al destransformar.

**Umbrales**: no se fijan a priori. Se calibran con la corrida de humo y los fija
el Arquitecto en el punto de control (ver tareas). Hasta entonces la compuerta
reporta métricas sin decidir.

### D6 — Protocolo y selección

- **Selección por arquitectura**: la búsqueda (random search sobre `d`, `L`,
  tamaño de capas, `lr`, dropout, early stopping) devuelve la **mejor
  configuración de cada arquitectura**, no un único ganador. Presupuesto de
  trials a calibrar tras el humo.
- **Ventanas**: pocas ventanas equiespaciadas del rolling (selección con
  `_select_windows`, ~4) para valid + el modelo final entrenado con todo lo
  anterior al test y evaluado una sola vez en el test held-out.
- **Semillas**: el modelo final de cada finalista se entrena con 3 semillas para
  reportar la varianza de la reconstrucción (media ± desviación); el pase DMD usa
  la semilla 0 en este change.
- Solo se optimizan parámetros del AE; **no** parámetros del DMD.

### D7 — Primer pase de DMD global sobre el latente congelado

Aplicado a las configuraciones finalistas que pasan la compuerta:

1. Se congela el AE y se extrae `z_t` (latente completo, ambos bloques) para
   cada hora de train.
2. DMD exacto global por mínimos cuadrados sobre pares `(z_t, z_{t+1})`, con
   `numpy`/`tf.linalg`, rango completo (y una versión truncada como segunda
   variante). Sin búsqueda de parámetros.
3. **Linealizabilidad**: error de predecir `z` a 1 y a 24 pasos con `K` y
   `K^24` sobre valid/test.
4. **Pronóstico informativo**: `z_{t+24} = K^24 z_t` → `decoder_mp` → ventana
   decodificada que **termina en `t+24`**; sus últimas 24 horas cubren
   exactamente `t+1 … t+24`, así que el promedio de esas 24 horas decodificadas
   (en µg/m³) es el `mean(MP(t+1 … t+24h))` comparable con el baseline sin
   necesitar una salida de 24 pasos. Requiere `L ≥ 24`.
5. Es **informativo**: no se afina nada para mejorarlo. La meteo futura no se
   conoce: el bloque `z_aux` evoluciona de forma autónoma bajo `K` (el
   tratamiento de la meteo futura y del control es el PA-2, en `ae-dmd`).

### D8 — Tracking y dashboard

- **Experimento MLflow**: mismo `sinca-baseline` (comparación directa en el
  dashboard; el nombre es engañoso pero cambiarlo rompería el flujo existente).
  Las corridas AE se distinguen por tags. Alternativa (`sinca-advanced`) queda
  como pregunta abierta.
- **Run AE (modelo conjunto)**: `sinca-zona-ae_{arq}-v{n}`; tags
  `modelo=ae-{arq}`, `tipo=entrenamiento`, `stage=ae-reconstruccion`,
  sin params `estacion`/`pollutant`. Params: `L`, `d`, `d_aux`, `w_mp`, `w_aux`,
  `aux_input`, `seed`, arquitectura, hiperparámetros. Métricas:
  `recon_rmse_*`, `recon_var_ratio_*`, `recon_slope_*` por serie,
  `gate_pass`, métricas de latente. Artefactos: `reconstruction.parquet`
  (timestamp, serie, y_true, y_recon, split), `latent.parquet` (latente `z` por
  hora y partición, columnas por bloque), `encoder.keras` y `decoder_mp.keras`
  (por separado) y tabla de diagnósticos. El pase DMD añade `dmd_operator.npz`
  (`K`, rango, autovalores).
- **Corridas de referencia**: `ref-mean` y `ref-pca-d{d}` con las mismas métricas.
- **Pase DMD**: por serie, **6 runs hijos** `modelo=ae-{arq}-dmd` con params
  `estacion`/`pollutant`/`version` y artefactos
  `predictions.parquet`/`zone_icap.parquet`/`rolling_scores.parquet` (mismo
  contrato que LightGBM → el dashboard los compara sin cambios), más tag
  `ae_run_id` apuntando al run AE. Métricas de linealizabilidad `dmd_latent_rmse_1`
  y `dmd_latent_rmse_24` en el run AE.
- **Dashboard**: tolerar runs sin `estacion`/`pollutant` y sin
  `feature_importance.parquet`; panel de reconstrucción (diagnósticos de
  aplanamiento, real vs reconstruido) y panel de linealizabilidad.

### D9 — Notebooks de exploración (además del dashboard)

El dashboard es la vista fija de resultados; los **notebooks** son para explorar
libremente. Tres plantillas en `notebooks/`: `exploracion_datos_ae` (huecos,
ventanas que sobreviven según `L`, rangos), `exploracion_reconstruccion`
(real vs reconstruido, aplanamiento, referencias, ablación) y
`exploracion_latente_dmd` (dimensiones activas, trayectoria de `z`, espectro de
`K`, linealizabilidad por época, pronóstico). Convención heredada de
`analisis_baseline.ipynb`: JupyterLab del Space con `sinca-dev-role` (sin
credenciales), lectura de `validated/` y `_mlflow/`, outputs limpios y versionado
con `git add -f`; son scratch, no producción ni fuente de verdad.

**Decisión**: exploración en **dos niveles**, según si hace falta el decoder.

- **Nivel 1, sin TensorFlow** (numpy/pandas/plotly sobre artefactos persistidos
  `reconstruction.parquet`, `latent.parquet`, `dmd_operator.npz`): datos de
  entrada, diagnósticos de reconstrucción, y casi todo el análisis del DMD sobre
  el latente — reajustar el DMD con otro rango, otra ventana temporal o
  regularización, espectro y períodos de `K`, linealizabilidad (error de `z` a
  1/24 pasos), variación de `K` entre épocas del año.
- **Nivel 2, con TensorFlow** (cargar `encoder.keras` y `decoder_mp.keras`):
  todo lo que **pasa por el decoder** — llevar un pronóstico del DMD a µg/m³
  tras cambiar el rango/ventana, recorrer una dimensión del latente y ver su
  efecto sobre las 6 series (travesías del latente, base de la
  interpretabilidad por modo), anular `z_aux` o un modo y ver el efecto, o
  perturbar la entrada.

`requirements-analysis.txt` (fuente única del contenedor del dashboard y del
Space) **no suma TF**. El nivel 2 usa un archivo aparte
(`requirements-notebooks-tf.txt`) que fija `tensorflow-cpu` en la **misma
versión** que `tf-cpu-v1` (evita incompatibilidades al cargar los `.keras`) y se
instala solo en el venv del Space cuando se necesita. Se verifica primero si la
imagen del Space ya lo incluye (no se asume). Los runs guardan `encoder.keras` y
`decoder_mp.keras` por separado (más livianos que cargar el modelo completo).
Alternativa descartada: reimplementar los decoders en numpy (inviable para
GRU/Conv; solo triviales para `linear_decoder`).

### D10 — Cómputo e imagen

- **Corrida de humo en CPU `ml.m5.large`** (1 ventana, pocas epochs), midiendo
  tiempo por epoch por arquitectura. Se sube a GPU solo si el conjunto de
  corridas supera ~2 h en CPU o el GRU lo vuelve inviable; si se queda en CPU, se
  registra la desviación respecto de `project-decisions.md` §3. Cuota
  `ml.g4dn.xlarge` = 1 ya disponible.
- **Imagen**: `docker/train-tf.Dockerfile` (solo runtime, `tensorflow-cpu`), en
  el repo ECR `sinca-train` con tags explícitos `lgbm-v1` y `tf-cpu-v1`; se
  deja de usar `:latest`. La imagen LightGBM existente se retaggea
  sin reconstruir.
- Iteración local en Docker con la misma imagen (TF en Windows nativo es solo CPU
  y sensible a la versión de Python).

## Risks / Trade-offs

- **Dataset chico / muestras muy solapadas → overfit** → redes chicas, dropout,
  early stopping, y las referencias (promedio, PCA(d)) como control.
- **Aplanamiento de picos** → compuerta dura, diagnósticos por nivel de
  concentración y espectro.
- **Sesgo de `log1p` confundido con aplanamiento** → se miden por separado;
  corrección de sesgo opcional.
- **Descarte de ventanas por huecos reduce mucho las muestras con `L` grande**
  → se mide primero (tarea de ejecución); alternativa de máscara de entrada.
- **GRU lento en CPU** → medición en el humo; GPU si hace falta (cuota ya
  disponible).
- **El pronóstico DMD del pase depende de que el decoder "invente" horas nuevas**
  (con `L = 168` solo las últimas 24 importan) → se reporta por `L`; es
  informativo, no un objetivo de este change.
- **Bloque `z_aux` autónomo en el rollout** (meteo futura desconocida) →
  limitación explícita; se aborda en `ae-dmd` (PA-2).
- **SQLite sin concurrencia** → una corrida por vez, como en Fases 3/3.5.
- **Pisar la imagen LightGBM** → tags explícitos, sin `:latest`.

## Migration Plan

1. Datos para red (huecos, normalización, ventanas, revisión de rangos) + tests
   con fixture.
2. Modelos, loss, métricas, compuerta y módulo DMD, probados localmente en
   Docker sobre un subset.
3. Imagen TF + push con tag explícito; retaggeo de la imagen LightGBM.
4. Medición de ventanas que sobreviven + corrida de humo en CPU → punto de
   control (Arquitecto fija umbrales de la compuerta y confirma CPU/GPU).
5. Referencias, búsqueda por arquitectura, control sin meteo, barrido de
   `w_aux`, pase DMD, evaluación final en test.
6. `EXPERIMENT-LOG.md` + verificación funcional (runs visibles en el dashboard).
7. **Rollback**: código aditivo; runs nuevos identificables por tag `modelo`
   y `stage`; la imagen LightGBM sigue disponible con su tag.

## Open Questions

- **Experimento MLflow**: ¿mismo `sinca-baseline` (decidido provisionalmente) o
  `sinca-advanced`? Confirmar antes de implementar el logging.
- **Umbrales de la compuerta** y presupuesto de trials: se definen tras el humo.
- **`d_aux`**: ¿proporción `d/4` o barrido propio?
- **Pesos por magnitud en la reconstrucción**: ¿hace falta ponderar los picos en
  la loss o alcanza con los diagnósticos? Decisión para `ae-dmd`.
- **Corrección del sesgo de `log1p`** al destransformar: decidir con los
  datos del humo.
