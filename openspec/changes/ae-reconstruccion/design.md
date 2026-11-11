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

- **Latente `d`** ∈ {8, 16, 32} es la dimensión **total**; `z_aux` tiene `d_aux`
  dimensiones (`max(2, d/4)`, **fijo**, no se barre: 4 con `d = 16`) y `z_mp`
  tiene `d − d_aux`. Así PCA(d) tiene el mismo tamaño total de latente y la
  comparación es justa. La búsqueda fija `d = 16`; `d ∈ {8, 32}` se barre solo
  sobre la mejor arquitectura (D6, tarea 11.2).
- **El decoder de MP ve todo el latente** (`z_mp` y `z_aux`): la meteo aporta
  contexto de dinámica a la reconstrucción de MP. Diagnóstico de aporte:
  poner `z_aux` en cero en evaluación y medir el empeoramiento del MP.
- **Variantes** (encoder/decoder): `mlp` (ventana aplanada), `conv1d`,
  `gru` (y `lstm` como variante interna de recurrente si hace falta) y
  `linear_decoder` (encoder no lineal, decoder lineal por rama: cada dimensión
  del latente tiene una forma fija sobre las series → interpretabilidad).
- Salida del decoder **lineal** (sin sigmoide/acotado), para no saturar picos.
- `L` ∈ {24, 72, 168} h soportado por el módulo de datos; **la búsqueda usa solo
  `L ∈ {24, 72}`** (`L = 168` queda fuera por memoria/tiempo, ver D10).

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

**Compuerta en dos capas** (decisión del Arquitecto tras la implementación;
ajustada en el punto de control 10.3). Se evalúa sobre valid, por serie de MP:
- **Capa relativa, siempre activa y sin parámetros**:
  1. `beats_mean`: `RMSE_AE < RMSE_promedio` en **todas** las series (estricto;
     en el humo el promedio da ~45 µg/m³ frente a ~16–21 de los AE, sin riesgo).
  2. `beats_pca`: en el **agregado**, la media de las 6 razones
     `RMSE_AE / RMSE_PCA(d)` es `< 1` (estricto). Se promedian razones, no RMSE,
     para que PM10 (mayor escala) no domine. Las razones por serie se siguen
     registrando.
  `gate_relative = 1 ∧ 2`. Sin umbrales numéricos, `gate_pass = gate_relative`.
- **Capa numérica, por serie, con valores por defecto fijos en el código** (se
  registran en el param `gate_thresholds`):
  - `max_rmse_vs_pca = 1.02`: en cada serie `RMSE_AE ≤ 1.02 × RMSE_PCA(d)`.
  - `min_var_ratio = 0.75` y `slope_tol = 0.30` (`|pendiente − 1| ≤ 0.30`):
    **fijados en el punto de control 10.5** (en 10.3 eran 0.80 y 0.20, a priori).
    La calibración 10.4 mostró que PCA(16), la referencia, queda por fuera de los
    valores a priori en valid: `var_ratio` mínimo 0.800 (`nielol_pm25`) y pendiente
    0.740 (`nielol_pm25`) y 0.797 (`las_encinas_pm10`), es decir `|pendiente − 1|`
    de 0.26 y 0.20. Un umbral más estricto que PCA(16) rechaza al propio
    estimador lineal óptimo y deja de medir "aplana más que PCA". Regla aplicada:
    **el umbral es el nivel de PCA(16) más un margen de ~0.05** (0.80 → 0.75;
    0.26 → 0.30). Se fijan por PCA y no por conv1d; que conv1d pase en sus 3 semillas
    es consecuencia (sus peores casos fueron pendiente 0.744, 0.792 y `var_ratio`
    0.794, todos dentro). Un AE solo falla esta capa si aplana claramente más que
    PCA(16).
  `gate_pass = gate_relative ∧ capa numérica`.

  **Limitación de umbrales absolutos**: valen igual en todas las series, así que en
  una serie donde PCA(16) aplana poco (p. ej. pendiente 0.95) permiten a un AE
  aplanar más que PCA sin fallar. La forma más precisa sería relativa por serie
  (`var_AE ≥ var_PCA − δ`, `|pend_AE − 1| ≤ |pend_PCA − 1| + δ`); no se adopta
  ahora porque no hay tabla por serie de PCA(16) en este punto de control y el ruido
  entre semillas de la pendiente (0.744–0.792 en conv1d) es del orden del margen.
  Compensación: 11.5 compara por serie `var_ratio` y pendiente de cada finalista
  contra `ref_pca_*` en valid y test, y anota cualquier serie donde el AE aplane
  más que PCA(16) en más de 0.05.

**Por qué tolerancia y no el criterio estricto por serie**: el humo mostró
`conv1d` empatado con PCA(16) en `las_encinas_pm10` (21.96 vs 21.96), de modo que
el veredicto dependía del tercer decimal, es decir, de ruido. Pero `≤ 1.00×`
tampoco resuelve eso (un empate por redondeo seguiría decidiéndose al azar). Se
separa la pregunta en dos: (i) *¿la no linealidad paga?* → estricto en el agregado;
(ii) *¿se sacrifica alguna serie?* → tolerancia de 2 % por serie (~0.4 µg/m³ sobre
21.96, del orden del ruido esperado entre semillas). La tarea 10.4 mide esa
dispersión con 3 semillas; si la desviación entre semillas de la razón supera
2 %, la tolerancia se reabre en 10.5. **Resultado de 10.4**: desviación entre
semillas de la razón media de conv1d 1.64 % (mlp 0.63 %, linear_decoder 0.71 %), bajo
el 2 %, y ningún fallo de conv1d fue por RMSE por serie (todos por pendiente o
`var_ratio`), así que **se mantiene 1.02**. Es una estimación con 3 semillas y sobre
el agregado (por serie el ruido es mayor): las 3 semillas de cada finalista de 11.3
se revisan de nuevo en 11.5.

**Limitación conocida**: en el entrenamiento final, `valid` es el último bloque del
rolling y queda **dentro** del train (se usa también para early stopping), así que
la compuerta se decide sobre datos que el AE y PCA vieron al ajustarse. No se
cambia en este change (sesgo ya aceptado, igual que el baseline), pero 11.5 compara
siempre `valid` contra `test` (`ref_pca_test_*` ya se registra) y anota cualquier
discrepancia en `EXPERIMENT-LOG.md`.

**Señal de alerta tras 10.4/11.1**: la búsqueda puntúa en un `valid` fuera del train
y **ningún trial le gana a PCA(16)** (mejor: conv1d 1.02 a `L = 72`), mientras que la
calibración, con `valid` dentro del train, da razón 0.895 en conv1d a `L = 24`. Los
regímenes difieren (`L`, 15 vs 60 epochs, 2 vs ~4 años de train), así que no es una
contradicción demostrada, pero el `valid` en-muestra favorece al AE (más parámetros
que PCA). Por eso, en 11.3, el Implementador reporta **la razón media AE/PCA en
test** de cada finalista antes de seguir; una compuerta aprobada en valid con razón
de test ≥ 1 se anota como "aprobada solo en valid" y se lleva al Arquitecto (el pase
DMD encadenado es informativo y no se detiene por esto, pero la conclusión del change
no puede apoyarse en ese modelo). La evidencia fuera de muestra disponible a favor de
conv1d a `L = 24`: en el humo de 3 epochs ganó a PCA(16) en test en las 6 series.

### D6 — Protocolo y selección

- **Selección por arquitectura**: la búsqueda (random search sobre `L`, tamaño de
  capas, `lr`, dropout, batch, con early stopping) devuelve la **mejor
  configuración de cada arquitectura**, no un único ganador.
- **Score de selección** (cambio respecto de la implementación inicial, que usaba
  el RMSE absoluto): en cada ventana de valid, `score = media sobre las 6 series
  de RMSE_AE / RMSE_PCA(d, L)`, promediado sobre las ventanas; menor es mejor y
  `< 1` significa ganarle a PCA. El RMSE absoluto no es comparable entre `L`
  (PCA(16) pasa de 17.1 a 22.8 µg/m³ al ir de `L = 24` a `L = 72`) ni entre `d`, y
  premiaría siempre `L = 24` y el `d` más grande.
- **Espacio de búsqueda por arquitectura** (`d = 16` y `d_aux = 4` fijos,
  `w_mp = 1`, `w_aux = 0.1`, `aux_input = true`, ver D2/D4):

  | Arquitectura | `L` | `hidden` | `dropout` | `lr` | `batch` | Trials |
  |---|---|---|---|---|---|---|
  | `linear_decoder` | {24, 72} | {32, 64, 128} | {0, 0.1, 0.2} | {1e-3, 3e-3} | {64, 128} | 12 |
  | `mlp` | {24, 72} | {32, 64, 128} | {0, 0.1, 0.2} | {1e-3, 3e-3} | {64, 128} | 12 |
  | `conv1d` | {24, 72} | {32, 64, 128} | {0, 0.1, 0.2} | {1e-3, 3e-3} | {128} | 6 |
  | `gru` | **{24}** | {32, 64} | {0, 0.1, 0.2} | {1e-3, 3e-3} | {128} | 4 |

  Los trials se reparten según el costo medido: las arquitecturas baratas
  (`linear_decoder`, `mlp`) admiten más muestras del espacio; `conv1d` y sobre todo
  `gru` se acotan (el GRU solo corre a `L = 24`, ver D10).
- **Presupuesto**: 4 ventanas equiespaciadas del rolling (`_select_windows`) ×
  15 epochs de búsqueda × patience 8. Estimado por extrapolación lineal en `L` de
  los s/epoch del humo, con ~17.5 mil muestras por ventana de 2 años: ~20 min
  (`linear_decoder`), ~30 min (`mlp`), ~100 min (`conv1d`), ~70 min (`gru`).
  Cada búsqueda tiene un tope de tiempo (`--search-time-budget`, 10 800 s): al
  excederlo no lanza más trials y escribe lo que lleva.
- **Ejecución**: el modo `search` no toca MLflow (escribe `ae_search.json` en el
  output del job), así que las **4 búsquedas corren en paralelo**, un Training Job
  por arquitectura (`ml.m5.large`, cuota 20), con `--max-run 14400`. La regla
  "una corrida a la vez" (SQLite sin concurrencia) sigue vigente para todos los
  modos que escriben en MLflow (`final`, `reference`, `dmd`).
- **Ventanas del final**: el modelo final se entrena con todo lo anterior al test
  y se evalúa una sola vez en el test held-out.
- **Epochs del final: 60 con patience 8 (confirmado en 10.5).** Regla aplicada: la
  loss de train mejoró 1.5–1.8 % entre las epochs 40 y 60 en conv1d y 3.6 % en el GRU
  (umbral 1 %); mlp y linear_decoder paran antes por early stopping. El GRU seguía
  mejorando a las 60: su veredicto vale con ese tope (ver Open Questions).
- **Semillas**: el modelo final de cada finalista se entrena con 3 semillas para
  reportar la varianza de la reconstrucción (media ± desviación); el pase DMD usa
  la semilla 0 en este change.
- **Finalistas (11.3)**: la mejor configuración de cada arquitectura según la
  búsqueda (conv1d, linear_decoder y mlp a `L = 72`; gru a `L = 24`) **más conv1d a
  `L = 24`** con los mismos hiperparámetros del ganador (hidden 64, dropout 0.1, lr
  **3e-3** (corregido en el cierre: `ae_search.json` dice 0.003; el texto decía 1e-3 por error de transcripción y 11.2 usó 1e-3), batch 128), como control de `L`. Motivo: la búsqueda (valid fuera del train,
  15 epochs, 2 años) no encontró ningún trial que le gane a PCA(16), pero conv1d a
  `L = 24` es la única configuración con evidencia a favor (60 epochs, razón 0.895
  en valid en-muestra; 6/6 series en test en el humo de 3 epochs). Sin este control no
  se podría separar el efecto de `L` del efecto de la arquitectura.
- **Barridos (11.2)**: sobre la arquitectura de menor score de búsqueda (conv1d),
  **a `L = 24`** y con los hiperparámetros del ganador (semilla 0, 60 epochs): control
  `aux_input` apagado, `w_aux ∈ {0, 0.1, 0.5, 1}` y `d ∈ {8, 32}` (con sus referencias
  PCA(8) y PCA(32) a `L = 24`). `d_aux` no se barre. Se usa `L = 24` y no el `L = 72`
  del ganador porque los barridos preguntan por el aporte de la meteo y de `d` en un
  modelo que pueda pasar la compuerta, y porque cuestan 16 min por corrida en vez de
  ~48 (7 corridas: ~2 h en vez de ~5.6 h).
- Solo se optimizan parámetros del AE; **no** parámetros del DMD.

### D7 — Primer pase de DMD global sobre el latente congelado

Aplicado a las configuraciones finalistas que pasan la compuerta:

1. Se congela el AE y se extrae `z_t` (latente completo, ambos bloques) para
   cada hora de train.
2. DMD exacto global por mínimos cuadrados sobre pares `(z_t, z_{t+1})`, con
   `numpy`/`tf.linalg`, rango completo (y una versión truncada como segunda
   variante). Sin búsqueda de parámetros. El pronóstico informativo (runs por
   serie `ae-{arq}-dmd`, con `rolling_scores.parquet` vacío por no haber
   búsqueda) usa solo el rango completo; la variante truncada se evalúa en
   linealizabilidad (`dmd_trunc_latent_rmse_*`) y se persiste en
   `dmd_operator_trunc.npz`.
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
6. **Cómo el pase accede al AE**: camino principal = **encadenado en el mismo
   job** del entrenamiento final, solo si pasa la compuerta (no hay descarga de
   artefactos ni escritura concurrente a SQLite). Respaldo = modo `dmd`
   independiente con `--ae-run-id`, que descarga `encoder.keras` y
   `decoder_mp.keras` del run desde `sinca-mlflow/_mlflow/` con los helpers de
   `analysis/load.py` (el `sinca-training-role` ya tiene lectura sobre ese
   prefijo); `--ae-dir` (directorio local) se mantiene para pruebas.

### D8 — Tracking y dashboard

- **Experimento MLflow (CERRADO)**: mismo `sinca-baseline`. El dashboard lee un
  solo experimento (`load.EXPERIMENT_NAME`); uno separado exigiría soporte
  multi-experimento fuera de alcance. Las corridas AE se distinguen por tags. El
  nombre es engañoso pero cambiarlo rompería el flujo existente.
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
credenciales), lectura de `validated/` y `_mlflow/` y versionado con `git add -f` (solo
notebooks nuevos); son scratch, no producción ni fuente de verdad. **Cierre
(2026-11-10):** (i) los gráficos usan **matplotlib** (commit `d77d812`): Plotly
tardaba demasiado en el Space; el dashboard sigue en Plotly. (ii) Los notebooks de
exploración se versionan **con outputs** de la última corrida con datos reales (antes:
outputs limpios); sin credenciales, IDs de cuenta ni ARNs.

**Decisión**: exploración en **dos niveles**, según si hace falta el decoder.

- **Nivel 1, sin TensorFlow** (numpy/pandas/matplotlib sobre artefactos persistidos
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
  tiempo por epoch por arquitectura (hecha en 10.1/10.2; ver tabla abajo).
- **Decisión de cómputo (punto de control 10.3): todo el change en CPU
  `ml.m5.large`, sin GPU.** Con `L ≤ 72` y el GRU solo a `L = 24`, ningún job pasa
  de ~3 h y el costo total queda en pocos USD (`ml.m5.large` ≈ USD 0.12/h). La GPU
  (`ml.g4dn.xlarge`, cuota 1) exigiría una imagen nueva (`tf-gpu-v1`, TensorFlow +
  CUDA compatible con el driver de la instancia), con riesgo de build y
  verificación, solo para habilitar el GRU a `L = 72`, que no es necesario para la
  compuerta. **Desviación registrada** respecto de `project-decisions.md` §3, que
  anticipaba GPU para AE+DMD (anotada también en §9). La GPU se reevalúa en
  `ae-dmd` si hace falta el GRU a `L ≥ 72`.
- **`L = 168` fuera de la búsqueda** (opción (c) de las cuatro evaluadas). El humo
  mostró que el pico de memoria no depende de la arquitectura sino del tamaño de
  los arrays de ventanas (1.8 GB a `L = 24` en las 4 arquitecturas; 3.5 GB a
  `L = 72`): ≈ 35 MB por hora de `L`, lo que extrapola a ~6.9 GB a `L = 168` antes
  de predecir y evaluar, coherente con el OOM en 8 GB (extrapolación de dos
  puntos, no medida). Por eso (a) batch más chico no ayuda (reduce activaciones, no
  los arrays) y (b) `ml.m5.xlarge` + (d) medir solo conv1d/mlp a `L = 168` agregan
  tiempo (lineal en `L`; el GRU a `L = 168` sería ~240 s/epoch) y un `L` cuyo
  RMSE ni siquiera es comparable con los demás. Además el pronóstico del pase DMD
  solo usa las últimas 24 h de la ventana decodificada. El módulo de datos sigue
  soportando `L = 168` (33.669 muestras de train medidas en 9.5). Si más
  adelante se quiere, se hace en `ae-dmd` con `ml.m5.xlarge` (16 GB).
- **GRU solo a `L = 24`** (opción "L más corto"): a `L = 72` el GRU tarda ~121
  s/epoch (≈ 2 h por semilla a 60 epochs, sobre el `max_run` de 2 h) y a `L = 24`
  ~43 s/epoch (≈ 43 min por semilla). Se descartó "más epochs con early stopping"
  (el tiempo por epoch es el problema, y el early stopping del final se mide sobre
  un `valid` que está dentro del train, ver D5) e "instancia más rápida" (agrega
  una cuota por verificar para un beneficio que `L = 24` ya cubre). `hidden` del
  GRU se acota a {32, 64} y `batch = 128` por costo.
- **`--max-run`**: los jobs de búsqueda y de entrenamiento final se lanzan con
  `--max-run 14400` (4 h; el default de 2 h del lanzador queda para el humo). Es una
  guarda, no una previsión: los tiempos esperados están en D6.
- **Mediciones del humo (3 epochs, `L = 24`, semilla 0; `version = 0`, entrenamiento sobre
  ~43 mil muestras)**:

  | Arquitectura | s/epoch | RSS pico | RMSE test medio (µg/m³) |
  |---|---|---|---|
  | `linear_decoder` | 2.2 | 1.8 GB | 19.4 |
  | `mlp` | 3.0 | 1.8 GB | 19.7 |
  | `conv1d` | 16.0 | 1.8 GB | 16.2 |
  | `gru` | 42.7 | 1.8 GB | 21.3 |
  | `gru` (`L = 72`) | 121.3 | 3.5 GB | 35.8 |
  | `gru` (`L = 168`) | — | OOM > 8 GB | — |

  Referencias de test: promedio 44.8 (`L = 24`) / 45.6 (`L = 72`); PCA(16) 17.1 /
  22.8. Ningún AE es de calidad (3 epochs): sirven para tiempos, memoria y para
  confirmar que el pipeline corre de punta a punta.
- **Imagen**: `docker/train-tf.Dockerfile` (solo runtime, `tensorflow-cpu`), en
  el repo ECR `sinca-train` con tags explícitos `lgbm-v1` y `tf-cpu-v1`; se
  deja de usar `:latest`. La imagen LightGBM existente se retaggea
  sin reconstruir.
- Iteración local en Docker con la misma imagen (TF en Windows nativo es solo CPU
  y sensible a la versión de Python).
- **`requirements.txt` fuera de `source_dir`**: el toolkit `sagemaker-training`
  instala el `requirements.txt` que encuentra en `source_dir` (`pipeline/`),
  arrastrando dependencias del scraper (`pandera`, `beautifulsoup4`, …) ajenas al
  training. Verificado que nadie lo usa (`scraper.Dockerfile` instala sus
  dependencias inline; ni el CI ni el README lo referencian): se mueve a la raíz
  como `requirements-scraper.txt`. En el humo, el job loguea las versiones de
  `tensorflow`, `numpy` y `pandas` al arrancar para confirmar que no cambian.

## Risks / Trade-offs

- **Dataset chico / muestras muy solapadas → overfit** → redes chicas, dropout,
  early stopping, y las referencias (promedio, PCA(d)) como control.
- **Aplanamiento de picos** → compuerta dura, diagnósticos por nivel de
  concentración y espectro.
- **Sesgo de `log1p` confundido con aplanamiento** → se miden por separado;
  corrección de sesgo opcional.
- **Descarte de ventanas por huecos reduce mucho las muestras con `L` grande**
  → se mide primero (tarea de ejecución); alternativa de máscara de entrada.
- **GRU lento en CPU** → medido en el humo; el GRU se limita a `L = 24`, `hidden ≤ 64`
  y 4 trials; la GPU queda diferida a `ae-dmd` (D10).
- **Umbrales de aplanamiento absolutos** (`min_var_ratio = 0.75`, `slope_tol = 0.30`,
  fijados en 10.5 a nivel PCA(16) + margen) → son laxos en series donde PCA aplana
  poco; 11.5 compara por serie contra PCA(16) y anota excesos > 0.05 (D5).
- **Compuerta decidida sobre un `valid` que está dentro del train final** → sesgo
  conocido; ahora con señal de alerta (búsqueda sin trials que ganen a PCA vs razón
  0.895 en-muestra): 11.3 reporta la razón en test de cada finalista y 11.5 compara
  `valid` vs `test` (D5).
- **Búsqueda y finales en regímenes distintos** (15 vs 60 epochs, 2 vs ~4 años de
  train) → el ganador de la búsqueda puede no ser el mejor del final; por eso conv1d
  a `L = 24` entra como quinto finalista (D6).
- **`L = 168` no explorado** → los resultados del change valen para `L ∈ {24, 72}`;
  la conclusión sobre contexto largo queda para `ae-dmd`.
- **Búsqueda por arquitectura con pocos trials** (4–12 sobre un espacio de
  72–216 combinaciones) → es una búsqueda exploratoria, no un óptimo; el score
  y las 4 configuraciones ganadoras se registran con su historial completo.
- **Un job de búsqueda excede su tiempo** → tope `--search-time-budget` y
  `ae_search.json` incremental: se pierde un trial, no la búsqueda entera.
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
   control 10.3 (hecho: CPU, `L = 168` fuera, GRU a `L = 24`, compuerta con
   tolerancia, presupuesto de búsqueda y `d_aux` fijados).
4b. Ajustes de código derivados del 10.3 (compuerta agregada, curva de
   aprendizaje, búsqueda por arquitectura con score vs PCA y tope de tiempo) →
   calibración 10.4 (60 epochs, `L = 24`) → punto de control corto 10.5
   (umbrales numéricos y epochs del final). La búsqueda 11.1 puede lanzarse en
   paralelo con 10.4 (no comparten MLflow); 10.5 bloquea solo a 11.3.
5. Referencias, búsqueda por arquitectura, control sin meteo, barrido de
   `w_aux` y de `d`, pase DMD, evaluación final en test.
6. `EXPERIMENT-LOG.md` + verificación funcional (runs visibles en el dashboard).
7. **Rollback**: código aditivo; runs nuevos identificables por tag `modelo`
   y `stage`; la imagen LightGBM sigue disponible con su tag.

## Open Questions

**Resueltas en el punto de control 10.3:**

- ~~Umbrales numéricos y presupuesto de trials~~ → compuerta relativa con
  tolerancia y umbrales por defecto fijos (D5); `min_var_ratio` y `slope_tol`
  quedan provisionales hasta 10.5. Presupuesto de búsqueda en D6.
- ~~`d_aux`~~ → fijo en `max(2, d/4)`; no se barre (los controles `aux_input` y
  `w_aux` ya miden el aporte auxiliar; la ablación de `z_aux` del humo fue
  0.35–1.21 µg/m³ en `linear_decoder`/`conv1d`/`mlp`, pequeña frente a un RMSE de
  16–21, y 5.10 en el GRU, que se revisa en 11.3).
- ~~CPU frente a GPU~~, ~~`L = 168`~~, ~~tiempo del GRU~~ → D10.

**Resueltas en el punto de control 10.5 (tras la calibración 10.4):**

- ~~`min_var_ratio` y `slope_tol` provisionales~~ → 0.75 y 0.30 (PCA(16) + margen);
  `max_rmse_vs_pca = 1.02` se mantiene (desviación entre semillas 1.64 % < 2 %). D5.
- ~~Epochs del final~~ → 60 (loss de train mejora 1.5–3.6 % entre las epochs 40 y 60).

**Siguen abiertas:**

- **GRU limitado a 60 epochs**: su loss de train aún mejoraba 3.6 % entre las epochs 40
  y 60 y su razón media vs PCA(16) fue 1.024 (1 semilla). Su veredicto vale con ese
  tope; si se quiere darle más epochs (≈ 43 s/epoch a `L = 24`) o `L = 72`, es decisión
  para `ae-dmd`.
- **Sesgo del early stopping del modelo final**: usa la última ventana `valid`
  del rolling, que queda dentro del train final (igual que el baseline); sesgo
  menor, se anota en `EXPERIMENT-LOG.md` al comparar (ver D5).
- **Pesos por magnitud en la reconstrucción**: ¿hace falta ponderar los picos en
  la loss o alcanza con los diagnósticos? Decisión para `ae-dmd`.
- **Corrección del sesgo de `log1p`** al destransformar: **cerrada en el cierre** (2026-11-10):
  no se corrige en este change; el sesgo por tramo es igual en AE y PCA(16) y no se
  separa del aplanamiento. Se reabre en `ae-dmd` junto con los pesos por magnitud.
- **GRU a `L ≥ 72` y `L = 168`** (GPU o `ml.m5.xlarge`): decisión para `ae-dmd`, solo
  si los resultados de `L ∈ {24, 72}` muestran que el contexto largo aporta.
