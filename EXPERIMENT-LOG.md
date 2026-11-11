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

