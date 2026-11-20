# EXPERIMENT-LOG

Registro narrativo de los experimentos de modelado de este proyecto
(convención de `prompt-sinca-mlops.md` §6). MLflow guarda las métricas y los
parámetros por corrida; acá va el **por qué**: qué se probó, con qué
hipótesis, qué resultado dio, y qué se decidió después.

Una entrada por experimento o familia de experimentos:

```
## <fecha> — <change / serie o grupo de series>

- **Hipótesis**: qué se esperaba y por qué.
- **Qué se probó**: datos, features, hiperparámetros, partición, loss.
- **Resultado**: métricas clave (las de MLflow) y observaciones.
- **Decisión / siguiente paso**: qué se hace con esto.
```

Reglas rápidas:

- Las corridas de MLflow se nombran `sinca-{estacion}-{modelo}-v{n}` (ver
  `pipeline/train/mlflow_log.py`); acá se referencian por ese nombre.
- Métricas decisoras: **RMSE ponderado** por serie y el **bloque percentil
  propio** (`recall_top10pct`/`recall_top5pct`/`pr_auc_top10pct`). El bloque
  ICAP oficial (`recall_alerta`/`recall_preemergencia`/`recall_emergencia`) es
  reporte comparativo: el histórico tiene pocos casos extremos y puede ser
  inestable.
- Cambios de alcance o de diseño no se anotan acá como hechos consumados: se
  manejan como propuesta de change OpenSpec (mismo §6 del prompt).

---

## 2026-10-19 — baseline (Fase 3): persistencia + LightGBM

- **Hipótesis**: con lags + meteorología + Fourier, un LightGBM por serie debe
  batir claramente al baseline de persistencia (piso trivial = último promedio
  móvil 24h). Las features de calendario (feriados, ventana GEC) y las
  armónicas diaria/semanal/anual deben capturar parte de la estacionalidad que
  los árboles no ven nativamente.
- **Qué se probó**:
  - **Serie objetivo**: promedio móvil de 24h a `t+24h` (`mean(MP(t+1 … t+24h))`
    por serie; 6 modelos independientes MP2.5/MP10 × 3 estaciones).
  - **Loss**: RMSE ponderado, `w_i = 1 + α·max(0, y_i − umbral_regular)` con
    `α = 1` y `umbral_regular` = ancla ICAP 100 (MP10 150 / MP2.5 50).
  - **Partición**: ventana deslizante (train 2 años / valid 1 mes / step 1 mes,
    42 ventanas sobre 2021-01-01 → 2026-06-30) + test final held-out
    `2026-07-01 → 2026-10-04`.
  - **Hiperparámetros**: random search + early stopping, ~20 trials por serie
    (espacio del `design.md` §9), evaluados promediando el RMSE ponderado sobre
    una muestra de ventanas de validación.
- **Resultado**: _(pendiente — se completa tras correr el Training Job
  completo, tareas 8.4/8.5/9.1)._ Se espera loguear en MLflow, por serie:
  `train_loss`, `rolling_weighted_rmse_mean/std`, `test_loss` y el bloque
  completo de métricas; más el run de referencia `sinca-persistence-baseline`.
- **Decisión / siguiente paso**: si el percentil propio (`recall_top10pct`)
  apenas mejora el de persistencia, revisar features (p. ej. lags de
  meteorología más cortos o viento direccional). Si `α`/`umbral_regular`
  resultan mal calibrados (muchos o ningún caso extremo), ajustar y re-correr.
  Con los resultados del baseline se fija la vara para la Fase 4 (autoencoder +
  DMD/Koopman).

---

## Plantilla - autoencoders (`ae-reconstruccion`)

Una entrada por corrida de experimento (o por familia de corridas). Los campos se
llenan con los **valores reales** de MLflow (`sinca-baseline`, runs `sinca-zona-ae_{arq}-v{n}`);
no se anotan resultados antes de tenerlos. Los umbrales de la compuerta numérica se fijan
en el punto de control 10.3, no acá.

```
## <fecha> - ae-<arquitectura> L=<L> d=<d>

- **Arquitectura y configuración**: arquitectura (`mlp`/`conv1d`/`gru`/`lstm`/`linear_decoder`),
  `L`, `d`, `d_aux`, `hidden`, `dropout`, `lr`, `batch_size`, `aux_input`.
- **Pesos de la loss**: `w_mp` (fijo en 1) y `w_aux` (barrido `{0, 0.1, 0.5, 1}`).
- **Semillas**: entrenamiento final con 3 semillas (`--seeds 0,1,2`); el DMD usa la semilla 0.
- **Hipótesis**: qué se esperaba de esta configuración y por qué.
- **Compuerta (capa relativa)**: `gate_pass`, `gate_beats_pca`, `gate_beats_mean` (por serie
  en `beats_*_by_series` del `config.json`). Capa numérica: `gate_numeric` (solo con umbrales).
- **Reconstrucción (test, µg/m³)**: RMSE medio y por serie (`recon_rmse_*`), razón de varianza,
  pendiente real vs reconstruido, espectro, error por tramo de concentración (< 50, 50–100, > 100).
- **Latente**: dimensión efectiva (`latent_eff_dim_pr`), dimensiones muertas (`latent_dead_dims`).
- **Aporte de `z_aux`**: `ablation_zaux_delta_rmse_test` (empeoramiento del MP al anularlo).
- **Linealizabilidad (DMD, semilla 0)**: `dmd_latent_rmse_1`, `dmd_latent_rmse_24` (y la
  variante truncada `dmd_trunc_latent_rmse_*`).
- **Pronóstico informativo vs persistencia y LightGBM**: métricas de los runs `ae-<arq>-dmd`
  frente a `sinca-persistence-baseline` y `sinca-<estacion>-lightgbm-v1` (mismas filas).
- **Hallazgos**: qué pasó, qué no esperábamos, problemas de datos o de cómputo (tiempo por
  epoch, memoria pico en `ml.m5.large`).
- **Decisión / siguiente paso**: qué se hace con esto (descartar, ajustar, pasar a `ae-dmd`).
```

---

## 2026-11-02 — ae-reconstruccion 9.4: entorno TensorFlow del Space

- **Hipótesis**: el Space (`sagemaker-distribution-cpu`) puede traer TensorFlow en una versión distinta de `tf-cpu-v1` (2.20.0), lo que impediría cargar los `.keras` de la imagen de training en el nivel 2 de los notebooks.
- **Qué se verificó (salida real)**: la imagen trae `tensorflow 2.18.0`, `keras 3.10.0`, `tf_keras 2.18.0`; disco `/home` 5.0 GB (4.9 GB libres); RAM 7.6 GiB. **No coincide** con `tf-cpu-v1`.
- **Qué se probó**:
  1. `pip install` directo sobre el Python base (`/opt/conda`): instaló TF 2.20 y `numpy` 2.5.4, rompió dependencias de la imagen (`sagemaker`, `autogluon`, `numba`, `catboost`, `scipy`) y **no persistió** tras apagar y encender el Space (volvió `tf 2.18.0`). Descartado.
  2. `pip install --user` en `~/.local`: TF importó, pero el `keras` 3.10 de la base quedó mezclado y el import falló. Descartado.
  3. **Venv aislado** `/home/sagemaker-user/venv-tf` con `requirements-notebooks-tf.txt`: `tf 2.20.0`, `keras 3.15.1`, `numpy 2.5.4`. Queda dentro de `/home`, que es el disco que persiste.
- **Decisión**: el nivel 2 de los notebooks usa `/home/sagemaker-user/venv-tf/bin/python`, no el Python base. El Python base sigue con TF 2.18 y **no** debe usarse para cargar `encoder.keras`/`decoder_mp.keras`.
- **Pendiente (para 12.2)**:
  - Persistencia del venv tras apagar y encender el Space: **verificada** (`tf 2.20.0`, `keras 3.15.1`).
  - Registrar el venv como kernel de JupyterLab (`ipykernel`), que todavía no está instalado.
  - Verificar que `analysis/load.py` y los notebooks de nivel 1 siguen funcionando en el Python base.

---

## 2026-11-02 — ae-reconstruccion 9.5: ventanas por L y rangos por canal (datos reales)

- **Hipótesis**: con la regla de descarte (ventana de MP sin huecos > 6 h) sobreviven ventanas suficientes también para `L = 168`. Si no, se para y se vuelve al Arquitecto.
- **Qué se corrió**: grilla horaria de 50.471 h (2021-01-01 → 2026-10-04), particiones held-out (train = posiciones [0, 48168), test = [48168, 50471)), normalización ajustada solo con train. Lectura de `sinca-data/validated/` y cómputo local; no se escribió nada en AWS. Cuenta personal verificada antes de correr.
- **Ventanas que sobreviven** (`survival_counts`):

  | L | train | test |
  |---|---|---|
  | 24 | 43.051 | 1.999 |
  | 72 | 39.414 | 1.761 |
  | 168 | 33.669 | 1.377 |

- **Rangos por canal** (`range_report`, espacio z, normalizado con train):
  - **MP**: sin recorte (fracción recortada 0). Rango z entre −3.7 y 4.6 (`pm10_nielol` máx. 4.56).
  - **Aux con colas**: `pres_nielol` mín. −20.7 z; `pres_las_encinas` mín. −10.2 z; `rain_las_encinas` máx. 62.1 z; `rain_nielol` máx. 23.9 z; `rain_padre_las_casas_ii` máx. 19.7 z (1,1 % recortado). Con ±5 de recorte en aux, la fracción recortada es baja (≤ 1,1 %).
  - **`rain`**: la mediana y los percentiles 1 y 99 de `rain_las_encinas` son iguales (≈ −0,026 z): casi todas las horas están en cero. Es el canal que se espera difícil (ver design D3).
- **Lectura**: `L = 168` tiene ~34 mil ventanas de train, más del doble de las ~17 mil que estimaba `design.md`. El criterio de parar no se cumple.
- **Decisión / siguiente paso**: continuar con 10.1 (humo). Como hay más muestras que lo estimado, la memoria pico de `L = 168` en `ml.m5.large` (8 GB) hay que medirla de verdad.

### Autoencoders — calibración de compuerta (10.4) y búsqueda (11.1), 2026-11-03

**Búsqueda (11.1, `mode=search`, 4 ventanas, 15 epochs, patience 8, score = razón RMSE_AE/RMSE_PCA(16) en valid):**

| Arquitectura | Trials | Score mejor | Config ganadora |
|---|---|---|---|
| conv1d | 6 | 1.02 | L=72, hidden 64, dropout 0.1, lr **3e-3** (corregido el 2026-11-10 según `ae_search.json`; antes decía 1e-3), batch 128 |
| linear_decoder | 12 | 1.04 | L=72, hidden 32, dropout 0, lr 1e-3, batch 64 |
| mlp | 12 | 1.05 | L=72, hidden 32, dropout 0, lr 1e-3, batch 64 |
| gru | 4 | 1.16 | L=24, hidden 64, dropout 0.1, lr 1e-3, batch 128 |

Ningún trial gana a PCA(16) en valid. Arquitectura para 11.2: **conv1d** (menor score).

**Calibración (10.4, L=24, d=16, hidden 64, dropout 0.1, lr 1e-3, batch 128, w_aux 0.1, 60 epochs, patience 8):**

| Arq | Semillas | Razón media AE/PCA (valid) | Desv. entre semillas | Compuerta (semilla) | Mejora loss ep40→60 |
|---|---|---|---|---|---|
| conv1d | 0, 1, 2 | 0.880 / 0.915 / 0.890 (media 0.895) | 1.64 % | pasa en 0; falla en 1 y 2 (numérica) | 1.53 % / — / 1.84 % |
| mlp | 0, 1, 2 | 1.091 / 1.075 / 1.078 | 0.63 % | no pasa | — (early stop) |
| linear_decoder | 0, 1, 2 | 1.087 / 1.105 / 1.098 | 0.71 % | no pasa | — (early stop) |
| gru | 0 | 1.024 | — | no pasa | 3.56 % |

Notas para 10.5:
- Pendiente de PCA(16) por debajo de `slope_tol = 0.20` (valor vigente en la calibración) en algunas series (nielol_pm25 0.740; las_encinas_pm10 0.797): el umbral era más estricto que PCA. Desde 10.5 `slope_tol = 0.30` y `min_var_ratio = 0.75` (tarea 3.9).
- Fallos numéricos de conv1d: pendiente en las_encinas_pm10 (semillas 1 y 2) y var_ratio en las_encinas_pm10 (0.794, semilla 1).
- DMD (conv1d semilla 0, único que pasó la compuerta): error de z a 1 paso 0.18 y a 24 pasos 0.97 en test.
- Ablación de `z_aux` (empeoramiento de RMSE MP, µg/m³): conv1d 0.94–1.87, mlp 0.29–0.36, linear_decoder 0.31–0.33, gru 3.46.
- Pico de RSS: conv1d ~1.8-2.2 GB, gru ~1.8 GB con L=24.

---

## 2026-11-07 - ae-reconstruccion 11.5: resumen de experimentos (finalistas, compuerta valid/test, aplanamiento, linealidad, pronóstico)

- **Qué se revisó (solo lectura)**: copia local de `_mlflow/` (`sinca-mlflow-state`), alineada con S3 (`head-object` de `_mlflow/mlflow.sqlite`, `LastModified` 2026-11-07 20:06 UTC). Artefactos `reconstruction.parquet`, `diagnostics.parquet`, `latent.parquet` y `predictions.parquet`. Las métricas se recalcularon desde `reconstruction.parquet` con las definiciones del pipeline (`pipeline/ae/metrics.py`): `var_ratio = std(recon)/std(real)`; `slope = cov(real, recon)/var(real)`; tramos por `y_true`: `< 50`, `50 ≤ y < 100`, `≥ 100`. Se verificó contra `diagnostics.parquet` (var_ratio, slope, rmse y bias_lt50 coinciden; bias_gt100 solo coincide con el tramo `≥ 100`).
- **Control de la compuerta**: la compuerta recalculada en `valid` coincide con `gate_pass` registrado en las 15 corridas finales (0 discrepancias). Umbrales: `min_var_ratio 0.75`, `slope_tol 0.30`, `max_rmse_vs_pca 1.02`.

### 1. Compuerta en valid frente a test (finalistas de 11.3, `version = 1`)

| Finalista | L | Semillas 0/1/2 | Compuerta valid (registrada) | Compuerta test (recalculada) | Razón AE/PCA(16) valid | Razón AE/PCA(16) test |
|---|---|---|---|---|---|---|
| conv1d | 24 | 0 / 1 / 2 | 1 / 1 / 1 | 1 / 1 / 1 | 0.904 (sd 0.004) | 0.916 (sd 0.012) |
| gru | 24 | 0 / 1 / 2 | 1 / 1 / 0 | 0 / 0 / 1 | 0.953 (sd 0.007) | 0.976 (sd 0.003) |
| conv1d | 72 | 0 / 1 / 2 | 0 / 0 / 0 | 0 / 0 / 1 | 0.854 (sd 0.009) | 0.945 (sd 0.010) |
| mlp | 72 | 0 / 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 | 1.015 (sd 0.008) | 1.025 (sd 0.005) |
| linear_decoder | 72 | 0 / 1 / 2 | 0 / 0 / 0 | 0 / 0 / 0 | 1.062 (sd 0.004) | 1.056 (sd 0.009) |

- **conv1d L=24 es el único finalista que aprueba en las 3 semillas, tanto en valid como en test.**
- **gru L=24 es inestable entre semillas.** En test, las semillas 0 y 1 fallan la capa numérica. La semilla 0 es la del pase DMD encadenado. Su razón media en test es 0.976 (< 1), así que no entra en la regla de "aprobada solo en valid" de D5.
- **conv1d L=72** cumple la alerta de D5: brecha valid–test de 0.09 (0.854 frente a 0.945). Aprueba 0/3 en valid, pero en test aprueba 1/3 (semilla 2) porque la semilla 2 es la única que pasa la capa numérica ahí.
- Ningún finalista queda en "aprobada solo en valid" con razón de test ≥ 1.

### 2. Aplanamiento por serie frente a PCA(16) del mismo L

Media de semillas. Δvar = `var_PCA − var_AE` y Δpendiente = `|pend_AE − 1| − |pend_PCA − 1|`. **FLAG** si el AE aplana más que PCA(16) en más de 0.05 en alguna de las dos. Valores negativos significan que el AE aplana menos.

| Finalista | Split | Series con FLAG (Δvar / Δpend) |
|---|---|---|
| conv1d L=24 | valid | PLC II pm10 (0.053 / 0.027), PLC II pm25 (0.088 / 0.069) |
| conv1d L=24 | **test** | **ninguna** (máx. PLC II pm25: 0.046 / 0.025) |
| gru L=24 | valid | LE pm10 (0.056 / 0.047), PLC II pm25 (0.105 / 0.086) |
| gru L=24 | test | PLC II pm25 (0.082 / 0.064) |
| conv1d L=72 | valid | LE pm10 (0.075 / 0.048) |
| conv1d L=72 | test | ninguna |
| mlp L=72 | test | Nielol pm10 (0.071 / 0.072) |
| linear_decoder L=72 | test | Nielol pm10 (0.083 / 0.090) |

- Las pendientes de todos los modelos, AE y PCA, quedan entre 0.44 y 0.90. **El aplanamiento es común a AE y PCA(16)**, y la diferencia con PCA depende de la serie.
- En valid, conv1d L=24 aplana más que PCA(16) en PLC II (pm10 y pm25); en test esa diferencia baja de 0.05 en las dos series, así que no hay FLAG en test.

### 3. Dispersión entre semillas

- Desviación estándar de la razón entre semillas: conv1d L=24 0.004 (valid) y 0.012 (test); gru 0.007 y 0.003; conv1d L=72 0.009 y 0.010; mlp 0.008 y 0.005; linear_decoder 0.004 y 0.009.
- Coherente con la medición de 10.4 (1.64 % entre semillas de conv1d). El límite de 2 % de la tolerancia por serie (D5) sigue siendo razonable para conv1d L=24. Para gru, la inestabilidad entre semillas se da en la capa numérica por serie, no en el agregado.

### 4. Sesgo y RMSE por tramo de concentración (test, media de las 6 series, µg/m³)

| Modelo | Sesgo `< 50` | Sesgo `50–100` | Sesgo `≥ 100` | RMSE `≥ 100` |
|---|---|---|---|---|
| conv1d L=24 | +0.45 | −4.71 | −21.75 | 49.15 |
| gru L=24 | +0.44 | −6.96 | −27.42 | 52.79 |
| PCA(16) L=24 | +0.42 | −5.63 | −25.35 | 56.00 |
| conv1d L=72 | +1.03 | −9.02 | −38.75 | 68.56 |
| PCA(16) L=72 | +0.88 | −10.38 | −44.83 | 74.80 |
| referencia promedio L=24 | +0.22 | −52.02 | −144.82 | 156.71 |

- El patrón (sobreestima en tramo bajo, subestima en tramo alto) aparece igual en el AE y en PCA(16). Como PCA también trabaja en el espacio `log1p`, **el sesgo no se puede atribuir al `log1p` por separado de la contracción general hacia la media.** En tramo alto el conv1d L=24 subestima menos que PCA(16).
- Esto no justifica por sí solo una corrección de sesgo en este change (ver Decisión).

### 5. Aporte de `z_aux` (ablación: Δ RMSE de MP en test, µg/m³) y latente

| Finalista | L | Semillas 0/1/2 | Dim. efectiva (PR de z, de 16) | Dimensiones muertas |
|---|---|---|---|---|
| conv1d | 24 | 1.24 / 1.54 / 0.52 | 8.2–8.6 | 0 |
| gru | 24 | 3.47 / 2.57 / 2.77 | 7.1–8.0 | 0 |
| conv1d | 72 | 2.13 / 1.69 / 1.86 | 10.9–12.4 | 0 |
| mlp | 72 | 4.48 / 1.53 / 0.73 | 8.6–9.0 | 0 |
| linear_decoder | 72 | 1.28 / 1.47 / 0.09 | 7.1–9.6 | 0 |

- `z_aux` aporta algo en todos los modelos, pero el efecto es pequeño frente a un RMSE de 16–21 µg/m³. El aporte es mayor en gru, aunque su compuerta es inestable.

### 6. Linealizabilidad del latente (pase DMD, semilla 0)

| Modelo | Escala `z` (rms train) | Error a 1 paso | Error a 24 pasos | Truncado (rango 8): 1 paso / 24 pasos |
|---|---|---|---|---|
| conv1d L=24 | 0.90 | 0.147 (16 %) | 0.779 (86 %) | 0.480 / 0.813 |
| gru L=24 | 0.46 | 0.057 (12 %) | 0.372 (81 %) | 0.192 / 0.386 |

- A 1 paso el operador lineal es razonable. **A 24 pasos el error es del mismo orden que la escala del latente**: `K^24` no predice bien el latente en el horizonte de 24 h. El rango truncado es peor en ambos casos.
- Desviación estándar de cada bloque del latente en train (contexto de escala): `z_mp` 0.85 y `z_aux` 1.03 en conv1d; `z_mp` 0.40 y `z_aux` 0.61 en gru.

### 7. Pronóstico informativo frente a LightGBM v1 (test, mismas filas)

Las filas comunes son la intersección de DMD, LightGBM v1 y `timestamp` UTC por serie. Los `n` del DMD (≈1 650–1 850 por serie) son menores que los del LightGBM registrado (≈1 800–2 000) porque el DMD solo predice donde la ventana decodificada está completa. Por eso la comparación se hizo sobre las filas comunes y no sobre los números reportados.

| Modelo | RMSE media (6 series) | RMSE ponderado media (6 series) |
|---|---|---|
| LightGBM v1 | 21.25 | 28.17 |
| DMD conv1d L=24 (semilla 0) | 20.97 | 33.28 |
| DMD gru L=24 (semilla 0) | 20.69 | 31.09 |

- Por serie, el DMD gana en RMSE en PLC II (pm10 y pm25) y pierde en las otras cuatro. En RMSE ponderado, pierde en 5 de 6 series. Pierde en los picos: el error de alta concentración es el que dispara el ponderado.
- **Persistencia no entra en esta comparación.** Sus runs no guardaron `predictions.parquet`, así que no es posible reconstruir las mismas filas. Sus métricas por serie de MLflow están sobre `n = 10 319` (no comparables fila a fila).
- Los runs `lightgbm` `version = 99` son de humo (9.2) y no se usan aquí. Están pendientes de limpieza antes de 12.1.

### 8. Barrido de 11.2 (conv1d, L=24, `version = 2`, semilla 0)

Nota: el barrido usó `lr = 0.001`, pero el ganador de 11.1 tiene `lr = 0.003`. Por eso estos runs no son idénticos al conv1d L=24 de 11.3 (ver discrepancia en 11.3). Razón AE/PCA con el PCA del mismo `d`:

| Variante | `aux_input` | `w_aux` | `d` | Compuerta (valid) | Razón valid | Razón test | Δ ablación `z_aux` |
|---|---|---|---|---|---|---|---|
| control sin meteo | off | 0.1 | 16 | 0 | 0.950 | 0.932 | 2.17 |
| base | on | 0.1 | 16 | 1 | 0.901 | 0.909 | 1.28 |
| `w_aux = 0` | on | 0.0 | 16 | 1 | 0.909 | 0.935 | 1.82 |
| `w_aux = 0.5` | on | 0.5 | 16 | 0 | 0.944 | 0.931 | 0.50 |
| `w_aux = 1` | on | 1.0 | 16 | 0 | 0.967 | 0.961 | 0.44 |
| `d = 8` | on | 0.1 | 8 | 0 | 0.893 | 0.906 | 1.51 |
| `d = 32` | on | 0.1 | 32 | 0 | 0.994 | 1.028 | 1.04 |

- Sin meteo (control) la compuerta falla y el MP empeora: la meteo aporta en el barrido. `w_aux = 0.1` es el mejor equilibrio. `w_aux` alto (0.5, 1) hace que el AE se parezca menos a PCA.
- `d = 8` tiene la razón de valid más baja, pero falla la capa numérica. `d = 32` no gana a PCA(32) en test.
- Este barrido es de una sola semilla, con `lr` distinto del finalista. Sirve para ordenar variantes, no para concluir sobre el final.

### Observaciones para el Arquitecto (no decididas aquí)

1. **conv1d L=24 es el AE de referencia del change**: única configuración que pasa la compuerta en las 3 semillas y en valid y test. Su pase DMD (semilla 0) es el resultado principal de linealizabilidad.
2. **gru L=24 no es estable**: el pase DMD encadenado (semilla 0) corresponde a una semilla que falla la capa numérica en test. Su resultado debe leerse como exploratorio.
3. **Sesgo de `log1p`**: no se puede separar del aplanamiento con este diseño (ver §4). Propuesta: no aplicar corrección en `ae-reconstruccion`; reabrirla en `ae-dmd`, junto con los pesos por magnitud (pregunta abierta de `design.md`).
4. **Pronóstico DMD**: en RMSE está al nivel de LightGBM, con el mismo orden de magnitud; en RMSE ponderado es peor. No es evidencia de mejora sobre LightGBM.
5. **Discrepancia de `lr` (11.2 frente a 11.3)** sin resolver: el barrido de 11.2 y el finalista de 11.3 no se pueden comparar directo.
6. **Nombres de run duplicados**: `sinca-zona-ae_conv1d-v2` (11.2) y `ref-mean` repetido se distinguen solo por params. La consulta de este resumen los separó por `d`, `aux_input`, `w_aux` y `seed`.

### Decisión / siguiente paso

- 11.5 registrada con las cifras anteriores. Se entrega al Arquitecto para decidir: (a) confirmar conv1d L=24 como AE de referencia, (b) tratar gru L=24 como exploratorio, (c) el criterio de corrección de `log1p` (propuesta: no corregir aquí), (d) cómo registrar la discrepancia de `lr`.
- No se modificó ningún run de MLflow ni se lanzó ningún Training Job en esta tarea.

### Decisión del Arquitecto (cierre, 2026-11-10)

- **(a)** conv1d L=24 es el **AE de referencia** y el punto de partida de `ae-dmd`: d=16 (d_aux=4), hidden 64, dropout 0.1, **lr 0.003**, batch 128, `w_aux` 0.1, `aux_input` on, 60 epochs, patience 8 (`version = 1`, semillas 0/1/2).
- **(b)** gru L=24 queda como **exploratorio**.
- **(c)** **No se corrige** el sesgo de `log1p` en este change; se reabre en `ae-dmd` con los pesos por magnitud.
- **(d)** El `lr` correcto del ganador de 11.1 es **0.003** (la fila de la tabla de 11.1 estaba mal transcripta). La referencia es el finalista de 11.3; los barridos de 11.2 (lr 0.001, semilla 0) solo ordenan variantes. La base de 11.2 (0.901/0.909) y el finalista (0.904/0.916) difieren poco; no se re-corren.
- **Matices para `ae-dmd`:** ganancia modesta sobre PCA(16) (~8 %); aplanamiento común a AE y PCA; PLC II pm25 justo bajo el umbral de 0.05 en test; el DMD global no mejora a LightGBM en RMSE ponderado (33.28 frente a 28.17), lo que motiva `ae-dmd`; `d = 8` como palanca a evaluar.


---

## 2026-11-15 - ae-dmd-prep: partición ae-dmd, reentrenamiento y referencias en sel

- **Hipótesis**: con una partición que no usa test para entrenar ni para detener (train hasta 2026-02, early stopping en marzo, selección en abril-junio), los AE, el pase DMD y las referencias (LightGBM y persistencia) se pueden comparar en `sel` sobre las mismas filas. Eso habilita la elección de `dmd-operador-global` sin mirar el test.
- **Qué se probó**:
  - Partición `ae-dmd` (features/partition.py): train 2021-01-01 a 2026-03-01, `es` 2026-03, `sel` 2026-04-01 a 2026-07-01, test 2026-07-01 a 2026-10-05 (solo latentes). Embargo de 24 h para filas con target.
  - AE conv1d y GRU, L = 24, d = 16, hidden 64, dropout 0.1, lr 0.003, batch 128, w_aux 0.1, aux_input on, 60 épocas, patience 8. Tres semillas por arquitectura: 6 Training Jobs, con pase DMD encadenado a cada semilla, sin mirar la compuerta.
  - LightGBM y persistencia reajustados en `sel` con el mismo corte, usando los hiperparámetros de los runs `lightgbm-v1` (un run fuente por serie, sin búsqueda nueva). Sesgo conocido: esas búsquedas usaron ventanas de validación hasta 2026-06, que incluye `sel`; favorece a LightGBM en esta comparación.
  - Nombres de run con semilla y contaminante (`sinca-{estacion}-{pollutant}-ae-{arq}-dmd-aedmd-v1-s{seed}`), para que pm10 y pm25 de una estación no compartan nombre.
- **Resultado** (`output/sel_summary_ae_dmd.md`, medias de las 6 series sobre filas comunes, n mínimo por serie = 1407):

  | modelo | RMSE | RMSE ponderado |
  |---|---|---|
  | DMD conv1d s0 / s1 / s2 | 23.08 / 24.13 / 23.98 | 38.83 / 40.92 / 39.54 |
  | DMD GRU s0 / s1 / s2 | 23.87 / 24.73 / 24.40 | 38.07 / 41.49 / 41.05 |
  | LightGBM (reajustado) | 31.04 | 32.13 |
  | Persistencia | 25.23 | 30.94 |

  - Dispersión entre semillas: conv1d ~1.05 de RMSE (rango); GRU ~0.86. Hay variación entre semillas en la misma magnitud que la diferencia con persistencia en RMSE simple.
  - Compuerta por semilla: `gate_pass = 0` en las 6 semillas (conv1d y GRU). El pase DMD corrió igual, como estaba definido. RMSE medio de reconstrucción en `sel`: conv1d 18.70 / 18.41 / 18.17; GRU 18.07 / 18.96 / 18.35 µg/m³.
  - Contraste con `pydmd` 2025.8.1: `dmd_pydmd_ok = 1` en los 6 AE (rango completo y truncado); diferencias de autovalores ~1e-15 en el contraste sintético dentro de la imagen `tf-cpu-v2`.
  - Linealidad del latente en `sel` (RMSE a 1 y 24 pasos): conv1d 0.15-0.18 y 0.81-0.93; GRU 0.06-0.07 y 0.39-0.40.
  - Costo real (facturable, 11.6): 7 jobs de la serie = 4637 s (1.29 h). Por job: lgbm-v1 89 s; conv1d s0/s1/s2 269/333/326 s; gru s0/s1/s2 1328/1149/1143 s. Con los dos humos, 4940 s (1.37 h). A la tarifa de `design.md` (~USD 0.115/h, no verificada en consola), ~USD 0.16 contra ~USD 0.46 estimados.
- **Las cifras de referencia cambian por la nueva partición; no es regresión.** Los números 33.28 / 31.09 / 28.17 eran RMSE de test con train hasta junio. Ahora son RMSE en `sel` (abril-junio) con train hasta febrero. No son comparables entre sí.
- **Decisión / siguiente paso**: estas filas `sel` son la base para `dmd-operador-global` (7.1): elegir representación y operador con `sel`, sin test. La compuerta en `ae-dmd` es informativa (`gate_pass = 0`), y el DMD global sigue siendo la fila de continuidad de 33.28 hasta que 7.1 decida. Anotado en las notas de `tasks.md`.

### v2 (presupuesto fijo)

- **Motivo**: diagnóstico de v1. El early stopping con marzo (`es`, mes de baja concentración) cortaba muy pronto: AE en épocas 3–24 y LightGBM en 41–145 rondas. No hay bug de máscaras ni de fuga.
- **Cambio (D11)**: AE con 60 épocas fijas, sin early stopping (`early_stopping=off`). LightGBM refit con `num_boost_round` = árboles del booster baseline `sinca-<estacion>-lightgbm-v1` de cada serie: LightGBM v2 usa las mismas rondas que el baseline v1 (1000 por serie; el `.txt` tiene 1000 árboles). El resto, igual que v1. Serie v2: 7 jobs `Completed`, `-Version 2`.
- Resumen: `output/sel_summary_ae_dmd_v2.md`. Filas comunes, n = 1407 en las 8 etiquetas.

**a) AE v2 (`sel`, 3 semillas por arquitectura)**

| run (`sinca-zona-ae_{arq}-aedmd-v2-s{seed}`) | epochs_run | gate_pass | sel_recon_rmse_mean | dmd_sel_latent_rmse_1 | dmd_sel_latent_rmse_24 | dmd_pydmd_ok |
|---|---|---|---|---|---|---|
| conv1d s0 | 60 | 1 | 16.820 | 0.137 | 0.698 | 1 |
| conv1d s1 | 60 | 1 | 16.659 | 0.128 | 0.650 | 1 |
| conv1d s2 | 60 | 1 | 16.549 | 0.137 | 0.703 | 1 |
| gru s0 | 60 | 1 | 17.159 | 0.060 | 0.351 | 1 |
| gru s1 | 60 | 0 | 17.732 | 0.055 | 0.340 | 1 |
| gru s2 | 60 | 0 | 17.904 | 0.057 | 0.327 | 1 |

**b) y d) Media de 6 series en `sel` (RMSE / RMSE ponderado), v1 frente a v2**

| modelo | RMSE v1 | RMSE v2 | Pond. v1 | Pond. v2 |
|---|---|---|---|---|
| DMD conv1d s0 | 23.075 | 24.065 | 38.828 | 39.392 |
| DMD conv1d s1 | 24.131 | 24.205 | 40.918 | 39.508 |
| DMD conv1d s2 | 23.980 | 23.087 | 39.544 | 37.401 |
| DMD GRU s0 | 23.867 | 24.009 | 38.068 | 36.934 |
| DMD GRU s1 | 24.725 | 23.654 | 41.493 | 36.171 |
| DMD GRU s2 | 24.402 | 22.723 | 41.054 | 35.767 |
| LightGBM | 31.038 | 28.283 | 32.130 | 34.236 |
| Persistencia | 25.226 | 25.226 | 30.938 | 30.938 |

LightGBM cambia de v1 a v2 porque v1 usó las rondas del early stopping (41–145) y v2 usa 1000 (las del baseline).

**c) Por serie, RMSE ponderado (v2)**

| serie | Persist. | LightGBM | Límite 1,05×LGBM | DMD conv1d (media) | DMD GRU (media) | Mejor DMD | Cumple |
|---|---|---|---|---|---|---|---|
| Las Encinas pm10 | 33.338 | 32.236 | 33.848 | 37.313 | 35.255 | 34.918 (GRU s1) | No (+3,2 %) |
| Las Encinas pm25 | 38.394 | 29.397 | 30.867 | 49.479 | 45.030 | 44.501 (GRU s1) | No (+44 %) |
| Nielol pm10 | 21.837 | 22.044 | 23.146 | 20.853 | 20.114 | 19.758 (GRU s2) | Sí |
| Nielol pm25 | 27.901 | 36.900 | 38.745 | 43.809 | 40.489 | 40.403 (GRU s1) | No (+4,3 %) |
| Padre Las Casas II pm10 | 33.566 | 50.889 | 53.433 | 41.390 | 37.449 | 36.428 (GRU s2) | Sí |
| Padre Las Casas II pm25 | 30.592 | 33.951 | 35.649 | 39.758 | 39.408 | 37.267 (GRU s2) | No (+4,5 %) |

RMSE simple por serie: `output/sel_summary_ae_dmd_v2.md`.

**e) Costo (`BillableTimeInSeconds`)**

| job | v1 (s) | v2 (s) |
|---|---|---|
| lgbm | 89 | 164 |
| conv1d s0 / s1 / s2 | 269 / 333 / 326 | 893 / 918 / 994 |
| gru s0 / s1 / s2 | 1328 / 1149 / 1143 | 2429 / 2364 / 2388 |
| **Total serie** | **4637 (1,29 h)** | **10150 (2,82 h)** |

v2 ≈ USD 0.32 a la tarifa de `design.md` (no verificada en consola). Sin los humos.

**Lectura**

1. Con 60 épocas, conv1d aprueba la compuerta 3/3 (igual que `ae-reconstruccion`). GRU 1/3, inestable entre semillas.
2. En `sel`, persistencia es la referencia más fuerte en RMSE ponderado (30,94). LightGBM entrenado una vez hasta 2026-02 queda peor (34,24); el control con reentrenamiento mensual (diagnóstico de v1) le ganaba. Se atribuye al desfase de 1 a 4 meses al inicio de temporada, no a un error.
3. El DMD tiene el mejor RMSE simple (GRU s2: 22,72; persistencia 25,23) y el peor RMSE ponderado (35,8–39,5, frente a 34,2 de LightGBM y 30,9 de persistencia): subestima los picos (a verificar con sesgo en picos). Es el objetivo de la etapa "Picos" de 7.1.
4. Criterio de §9 (ponderado ≤ 1,05 × LightGBM, por serie; mejor corrida DMD por serie, punto estimado): cumple en 2 de 6 (Nielol pm10, Padre Las Casas II pm10). Necesita ≥ 4/6: no cumple. Los fallos van de +3 % a +44 %. El intervalo bootstrap no está calculado, así que el veredicto por serie puede cambiar. Dispersión entre semillas (RMSE simple, rango de 3): conv1d 1,1; GRU 1,3. El mejor DMD se elige entre 6 corridas, así que es optimista.

**Decisión**: v1 queda superada; la base de 7.1 es v2. Pendiente para 7.1: revisar si la referencia del criterio debe incluir persistencia en las 6 series (en 5 de 6 es mejor que el mejor DMD; no en Nielol pm10).

## 2026-11-19 - dmd-operador-global 1.0: oráculo de la representación

- **Hipótesis**: si decodificar el estado real en `t+24` (operador perfecto) queda cerca o sobre la mejor referencia, el techo es la representación y los AE salen (D9: techo si el intervalo de la razón alcanza 0.95).
- **Qué se probó**: oráculo de conv1d y GRU v2 (3 semillas) y PCA(12)+PCA(4) en `sel`, sobre `eval_rows` (n mínimo 1407 por serie). Bootstrap de bloques de 7 días, B = 2000, semilla 0. Mejor referencia por serie = menor RMSE ponderado entre LightGBM v2 y persistencia v2 (persistencia en 4/6; LightGBM en Las Encinas pm10/pm25). Runs `sinca-zona-dmdop-{rep}-e10-aedmd-v1-s{seed}` (`tipo=diagnostico`). Continuidad con el DMD v2: K y `y_pred` idénticos (rel. 0).
- **Resultado** (media de 6 series; razón vs mejor referencia con intervalo 95 %):

  | representación | RMSE pond. | razón vs mejor ref. |
  |---|---|---|
  | conv1d (media 3 semillas) | 7.22 | 0.244 [0.178, 0.308] |
  | GRU (media 3 semillas) | 6.45 | 0.218 [0.168, 0.257] |
  | PCA | 6.76 | 0.232 [0.202, 0.259] |
  | Referencias | LightGBM 34.24, persistencia 30.94 | |

  Peor serie: PLC II pm25 conv1d 0.370 [0.226, 0.450]. Ninguna serie ni familia se acerca a 0.95.
- **Lectura**: el techo de reconstrucción no limita el pronóstico a 24 h (el promedio de 24 h borra casi todo el error de reconstrucción). El problema es la dinámica.
- **Decisión** (decidido por regla D9): los AE siguen; Hankel no pasa a línea principal. Sigue la escalera 1.1 con conv1d, GRU, PCA y Hankel.

## 2026-11-19 - dmd-operador-global 1.1: escalera de representaciones

- **Hipótesis**: con el mismo DMD exacto autónomo, la representación que mejor linealiza la dinámica a 24 h gana; Hankel (retardos) es el piso.
- **Qué se probó**: DMD exacto autónomo en `sel` (`eval_rows`, n mín. 1407). Hankel con rango Gavish–Donoho (GD); PCA y AE con rango completo (GD reportado). Runs `…-e11a…e11g-aedmd-v1-s{seed}`.
- **Resultado** (media de 6 series; AE = media de 3 semillas; diferencia emparejada vs `hankel` con intervalo 95 %):

  | rep. | rango (GD) | RMSE pond. | razón vs mejor ref. | dif. vs hankel |
  |---|---|---|---|---|
  | 1.1a hankel | 38 (38) | 41.09 | 1.394 [1.077, 1.635] | — |
  | 1.1b hankelmet | 133 (133) | 41.62 | 1.412 [1.104, 1.652] | +0.53 [+0.07, +1.19] |
  | 1.1c pca | 16 (6) | 39.82 | 1.349 [1.069, 1.553] | −1.27 [−2.18, +0.24] |
  | 1.1d conv1d | 16 (4) | 38.77 | 1.316 [1.079, 1.478] | −2.32 [−4.46, +0.50] |
  | 1.1e gru | 16 (3–4) | 36.29 | 1.234 [1.061, 1.346] | −4.80 [−8.38, −0.36] |
  | 1.1f hankelest | 45 (bloques) | 39.62 | 1.348 [1.122, 1.527] | −1.47 [−3.55, +1.37] |
  | 1.1g hankel48 (n mín. 1376) | — | — | — | −1.25 [−2.20, +0.01] |

  - Pares entre candidatos: GRU supera significativamente a hankel, hankelmet y conv1d (−2.48 [−4.14, −0.30]); no a PCA (−3.53 [−6.69, +0.58]).
  - Regla GRU (D9): rango entre semillas 1.17 < semiancho del intervalo 7.85: no se descarta (conv1d: 2.11 < 9.69).
  - Meteo con retardos (1.1b) empeora poco pero de forma significativa; Hankel por estación (1.1f) y h = 48 (1.1g) no difieren de forma significativa: no hay señal clara de cruce entre estaciones ni de memoria (conv1d L = 72 no se habilita).
  - Todas las variantes siguen sobre la mejor referencia (intervalos de la razón > 1); mejor serie: Ñielol pm10 (GRU 0.921 [0.810, 1.063]).
- **Decisión** (decidido por regla D9): pasan **PCA** (la más simple que ninguna otra supera de forma significativa) y **GRU** (mejor puntaje; no significativamente peor que PCA). conv1d sale (GRU lo supera de forma significativa); Hankel sale.

## 2026-11-19 - dmd-operador-global 1.2: picos (pesos y Duan)

- **Hipótesis**: el DMD pierde en picos por regresión a la media (H2) y por Jensen al deshacer `log1p` (H6). 1.2a pondera los pares con `w` (máx. de las 6 series, solo `t` con `t+24` en train); 1.2b corrige la retransformación con smearing de Duan (`S_{s,k}` de rollouts de train).
- **Qué se probó**: sobre PCA y GRU (3 semillas), DMD exacto autónomo: base, Duan, pesos, pesos+Duan. Runs `…-e12base/e12b/e12a/e12ab-aedmd-v1-s{seed}`.
- **Resultado** (media de 6 series; GRU = media de semillas; sesgo en episodios ≥ ICAP100, media de series):

  | rep. | variante | RMSE pond. | razón vs mejor ref. | sesgo episodios |
  |---|---|---|---|---|
  | PCA | base | 39.82 | 1.349 [1.069, 1.553] | −58.1 |
  | PCA | Duan | 32.68 | 1.109 [0.953, 1.236] | −41.1 |
  | PCA | pesos | 29.48 | 1.003 [0.907, 1.099] | −33.1 |
  | PCA | pesos+Duan | 39.98 | 1.363 [1.127, 1.638] | +3.5 |
  | GRU | base | 36.29 | 1.234 [1.061, 1.346] | −47.9 |
  | GRU | Duan | 29.73 | 1.014 [0.968, 1.066] | −27.1 |
  | GRU | pesos | 29.56 | 1.010 [0.945, 1.130] | −22.6 |
  | GRU | pesos+Duan | 37.96 | 1.297 [1.147, 1.589] | +8.1 |

  - Duan y pesos mejoran de forma significativa sobre la base en ambas (PCA −7.14 [−9.97, −2.38] y −10.34 [−14.94, −2.93]; GRU −6.56 [−9.75, −1.81] y −6.73 [−11.04, −0.09]).
  - Pesos frente a Duan: no significativo (PCA −3.20 [−5.19, +0.10]; GRU −0.17 [−1.42, +1.83]).
  - Combinadas sobrecorrigen (sesgo positivo en picos y peor RMSE ponderado global).
  - Primera vez que el operador queda en el nivel de la mejor referencia (razón ≈ 1.0, intervalos que cruzan 1).
- **Decisión** (decidido por regla D9, "sin peso < con peso"): **Duan (1.2b)** como decisión global, sin pesos, para PCA y GRU. Nota: en PCA pesos queda en el borde de la significancia.

## 2026-11-19 - dmd-operador-global 1.3: exógenas en el rollout

- **Hipótesis**: si `z_aux` deja de ser estado y en cada paso se recodifica la ventana auxiliar con persistencia diurna de la meteo y calendario real (`z_mp,t+1 = K_mm·z_mp,t + K_ma·z_aux,t`), el rollout deja de "pronosticar" meteo con K (S2–S4) y mejora.
- **Qué se probó**: PCA y GRU (3 semillas), DMD exacto + Duan (decisión 1.2): 1.3a autónoma vs 1.3b exógena (`K_mm`, `K_ma` por LS en train; Duan reestimado con el rollout exógeno). Sin meteo posterior a `t` (test de no-fuga). `K_ma` en `k_ma.parquet` y `operator.npz`. Runs `…-e13a/e13b-aedmd-v1-s{seed}`.
- **Resultado** (media de 6 series; GRU = media de semillas):

  | rep. | variante | RMSE pond. | razón vs mejor ref. | dif. exógena − autónoma |
  |---|---|---|---|---|
  | PCA | autónoma | 32.68 | 1.109 [0.953, 1.236] | |
  | PCA | exógena | 32.06 | 1.089 [0.957, 1.189] | −0.62 [−1.46, +0.34] |
  | GRU | autónoma | 29.73 | 1.014 [0.968, 1.066] | |
  | GRU | exógena | 29.20 | 0.997 [0.951, 1.057] | −0.53 [−1.14, +0.18] |

  La exógena mejora poco y en la dirección esperada (menos sesgo en picos: PCA −41.1 → −38.9; GRU −27.1 → −22.9), sin significancia.
- **Decisión** (decidido por regla D9, "autónomo < exógeno"): queda el rollout **autónomo** en PCA y GRU.

## 2026-11-19 - dmd-operador-global 1.4: horizonte (K horario vs A₂₄)

- **Hipótesis**: el operador directo a 24 h (`z_t → z_{t+24}`) evita el error compuesto de `K²⁴`; se mide solo como referencia de precisión (el ciclo diario debe quedar como modo de K horario, §9).
- **Qué se probó**: PCA y GRU (3 semillas), exacto autónomo + Duan: K horario (`e14a`) vs `A₂₄` (`e14b`, `tipo=referencia`; Duan reestimado con `A₂₄`).
- **Resultado** (media de 6 series):

  | rep. | variante | RMSE pond. | razón vs mejor ref. | dif. A₂₄ − horario |
  |---|---|---|---|---|
  | PCA | horario | 32.68 | 1.109 [0.953, 1.236] | |
  | PCA | A₂₄ | 33.09 | 1.126 [0.928, 1.296] | +0.41 [−1.42, +1.61] |
  | GRU | horario | 29.73 | 1.014 [0.968, 1.066] | |
  | GRU | A₂₄ | 30.78 | 1.046 [0.923, 1.133] | +1.04 [−1.82, +3.21] |

  `A₂₄` no mejora la precisión (peor en punto, sin significancia): el error compuesto del rollout horario no es la limitante. K horario multi-paso (1 + 24 h) queda diferido a `deep-koopman`.
- **Decisión** (decidido por regla D9; `A₂₄` no es candidato): queda **K horario**.

## 2026-11-19 - dmd-operador-global 1.5: estimador y estabilidad

- **Hipótesis**: el recorte `|λ| ≤ 1−ε` evita explosiones del rollout y fbDMD corrige el sesgo de autovalores hacia adentro (H2).
- **Qué se probó**: PCA y GRU (3 semillas), autónomo + Duan: exacto (`e15a`), recorte ε = 1e-3 (`e15b`), fbDMD + recorte (`e15c`). Bootstrap por modo del elegido: bloques de 7 días de train, B = 200, emparejamiento húngaro en `(log|λ|, arg λ)`, aparición si período y vida media difieren ≤ 25 % (`bootstrap_modos.parquet` en los runs `e15a`).
- **Resultado** (media de 6 series):

  | rep. | exacto | recorte | fbDMD+recorte | fbDMD − exacto |
  |---|---|---|---|---|
  | PCA | 32.68 | 32.68 | 36.66 | +3.98 [−0.67, +9.49] |
  | GRU | 29.73 | 29.73 | 34.02 | +4.28 [+2.75, +6.55] |

  - El recorte casi no actúa (radio espectral ≈ 0.999 < 1−ε en casi todos los operadores).
  - fbDMD empuja los modos hacia el círculo unidad (vidas medias más largas): reduce el sesgo en picos pero empeora el RMSE ponderado. Esperable: con forzamiento estocástico estacionario las regresiones hacia adelante y hacia atrás tienen el mismo sesgo y su cociente tiende a `|λ| = 1`; fbDMD supone ruido solo de medición.
  - Bootstrap por modo (exacto): el modo diario (período 24.0–24.3 h) aparece en ≥ 94.5 % de las re-estimaciones y en las 3 semillas GRU (vida media 290–440 h). Los modos de 10–33 h aparecen en 100 % del bootstrap de cada semilla, pero entre semillas GRU solo coinciden algunos; los casi reales de vida media > 100 h son inestables (aparición 30–70 %).
- **Decisión** (decidido por regla D9, "exacto < fbDMD"): queda el **DMD exacto** (sin recorte) en PCA y GRU.

## 2026-11-19 - dmd-operador-global 1.6: rango y modos (pendiente de decisión del operador)

- **Hipótesis**: con techo de rango Gavish–Donoho y eliminación hacia atrás por el error ponderado a 24 h quedan pocos modos interpretables sin perder precisión.
- **Qué se probó**: configuración vigente (exacto autónomo + Duan) en PCA y GRU (3 semillas), `d = 16` fijo. Techo GD sobre `Z₀` de train; se quita de a un modo (par conjugado junto) el de menor empeoramiento mientras el intervalo de `nuevo − actual` no quede entero sobre 0; Duan reestimado en cada paso. Runs `…-e16-aedmd-v1-s{seed}` con `curva_modos.parquet`.
- **Resultado**:

  | variante | rango GD | modos techo → final | RMSE pond. rango completo | RMSE pond. final | final − completo |
  |---|---|---|---|---|---|
  | PCA | 6 | 3 → 1 | 32.68 | 36.32 | +3.64 [−2.21, +8.40] |
  | GRU s0 | 4 | 2 → 1 | 29.78 | 31.54 | +1.76 [−1.17, +4.59] |
  | GRU s1 | 3 | 2 → 1 | 29.93 | 31.31 | +1.38 [−0.81, +2.97] |
  | GRU s2 | 4 | 2 → 1 | 29.48 | 30.11 | +0.63 [−1.64, +2.58] |

  - Finales (media de 6 series): PCA 36.32, razón vs mejor ref. 1.240 [0.913, 1.564]; GRU 30.99, 1.057 [0.989, 1.136]. GRU − PCA −5.33 [−14.72, +4.95] (no significativa).
  - La eliminación no se detiene por empeoramiento significativo en ninguna variante: queda **un solo modo**, casi real y lento (vida media 230–550 h; período > 1000 h o real): el "nivel" que decae.
  - **El modo diario (período ≈ 24 h) se elimina en todas**: no afecta al target (promedio de 24 h, que lo cancela). En GRU s0 y s2 quitarlo incluso mejora (−1.37 [−2.16, −0.39]; −0.80 [−1.67, +0.18]). En PCA cae en el primer paso (+0.20 [−0.01, +0.44]).
  - El techo GD (rango 3–6 de 16) ya reduce el operador a 2–3 modos.
- **Lectura por regla D9**: PCA, rango 6, 1 modo, Duan (la más simple; GRU no la supera de forma significativa). La cadena de "no significativo → la más simple" deja un operador peor en punto que los mejores evaluados (GRU completo + Duan 29.73; razón 1.014 [0.968, 1.066]).
- **Decisión**: **pendiente del operador** (no se aplicó la regla). El criterio de selección (error a 24 h del promedio) elimina el ciclo diario, que §9 fija como modo de la dinámica, y reduce el operador a persistencia del nivel. Opciones en `tasks.md` (8.3).
