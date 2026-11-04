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
- **Qué se corrió**: grilla horaria de 50.471 h (2021-01-01 → 2026-10-04), particiones held-out (train = posiciones [0, 48168), test = [48168, 50471)), normalización ajustada solo con train. Lectura de `sinca-data/validated/` y cómputo local; no se escribió nada en AWS. Cuenta verificada antes de correr: `549024266383`.
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
| conv1d | 6 | 1.02 | L=72, hidden 64, dropout 0.1, lr 1e-3, batch 128 |
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
- Pico de RSS: conv1d ~1.8–2.2 GB, gru ~1.8 GB con L=24.

