> Orden: primero las tareas de **escritura de código/archivos** (grupos 1-8, no
> requieren AWS); al final las de **ejecución real** (grupos 9-12), con el
> operador corriendo los comandos y devolviendo el output real. La última tarea
> es una verificación funcional explícita.

## 1. Módulo de datos para red neuronal (`pipeline/ae/data.py`)

- [x] 1.1 Cargar los datos validados con `features/build.load_validated` y armar la tabla horaria de las 6 series de MP sobre una grilla horaria completa
- [x] 1.2 Armar los canales auxiliares: `pres`, `wspd`, `rain` × 3 estaciones, `temp` de Ñielol y Las Encinas, y `blh` (sin `rhum`, `glob` ni `temp` de Padre Las Casas II)
- [x] 1.3 Armar los canales de calendario (Fourier diaria/semanal/anual sin/cos, `dow_sin/cos`, `is_holiday`, `gec_window`), reutilizando las constantes de `features/build.py`
- [x] 1.4 Interpolar linealmente huecos de ≤ 6 h; conservar la máscara de valores reales; en aux dejar huecos > 6 h en 0 con columna de máscara de entrada
- [x] 1.5 Cortar ventanas `[L × 6]` (`L ∈ {24, 72, 168}`), descartando las de MP con huecos > 6 h, y asignar cada muestra a train/valid/test solo si toda su ventana cae dentro del conjunto (reutilizar `make_rolling_windows`, `final_holdout_split`, `_select_windows`)
- [x] 1.6 Transformación: `log1p` en `pm25`, `pm10`, `rain`, `blh`, `wspd`; z-score por canal con estadísticas solo del train de la partición; recorte ±5 solo en aux (nunca en MP); invertir la transformación para reportar en µg/m³
- [x] 1.7 Reporte de rangos por canal (mínimo, máximo, percentiles, fracción recortada)
- [x] 1.8 Tests unitarios con fixture sintético: interpolación, descarte de ventanas, máscaras, asignación por partición, estadísticas solo de train, no recorte de MP

## 2. Modelos AE (`pipeline/ae/models.py`) y loss

- [x] 2.1 Implementar el AE en Keras: encoder de MP, rama auxiliar, latente `z=[z_mp | z_aux]` (`d ∈ {8,16,32}`, `d_aux = d/4` por defecto, ≥ 2), decoder de MP que recibe el latente completo y decoder auxiliar; salida lineal
- [x] 2.2 Variantes: `mlp`, `conv1d`, `gru` (con opción `lstm`) y `linear_decoder`
- [x] 2.3 Interruptor `aux_input` (apagado = rama aux solo con calendario)
- [x] 2.4 Loss `w_mp · E_mp + w_aux · E_aux` con MSE enmascarado y promedio por canal (calendario sin loss); `train_step`/`compile` según corresponda a Keras 3
- [x] 2.5 Tests: formas de entrada/salida por variante, enmascarado de valores no reales, `w_aux = 0` anula el término aux

## 3. Referencias, métricas y compuerta (`pipeline/ae/metrics.py`)

- [x] 3.1 Referencia predictor del promedio y referencia PCA con el mismo `d`, con las mismas métricas de reconstrucción
- [x] 3.2 Métricas de reconstrucción por serie en µg/m³: RMSE, razón de varianza, pendiente real vs reconstruido, espectro de potencia, RMSE y sesgo por tramos de concentración (< 50, 50–100, > 100), captura de máximo y percentiles altos
- [x] 3.3 Diagnósticos del latente: dimensión efectiva (razón de participación del PCA de `z`) y dimensiones muertas
- [x] 3.4 Ablación de `z_aux` (poner el bloque en cero y medir el empeoramiento del MP)
- [x] 3.5 Compuerta de reconstrucción dura sobre MP, con umbrales **parametrizables** (se fijan tras el humo; por defecto solo reporta sin decidir)
- [x] 3.6 Tests con datos sintéticos conocidos (señal suavizada → razón de varianza < 1; reconstrucción perfecta → pendiente 1)
- [x] 3.7 Compuerta en dos capas (ver `design.md` D5): `gate_pass` se calcula y se registra **siempre** con la capa relativa (el AE supera al predictor del promedio y a PCA(d) en la reconstrucción de MP; calcular `gate_beats_mean` además de `gate_beats_pca`); la capa numérica (`--gate-thresholds`) se suma al `gate_pass` cuando se fije en el punto de control. Actualizar `gate_decision` y los tests (con y sin umbrales)

## 4. Módulo DMD (`pipeline/ae/dmd.py`)

- [x] 4.1 Extraer `z_t` del AE congelado para cada hora de la partición
- [x] 4.2 DMD exacto global por mínimos cuadrados sobre pares `(z_t, z_{t+1})`, rango completo y variante truncada, con `numpy`/`tf.linalg`
- [x] 4.3 Linealizabilidad: error de `z` a 1 y a 24 pasos (`K`, `K^24`) en valid/test
- [x] 4.4 Pronóstico informativo: `K^24 z_t` → `decoder_mp` → promedio de las últimas 24 horas de la ventana decodificada, en µg/m³, alineado con el target de `features/build.py`
- [x] 4.5 Tests con sistema lineal sintético (K recuperado, linealizabilidad ≈ 0) y verificación de que el promedio coincide con `mean(MP(t+1 … t+24h))` en un caso trivial

## 5. Entry point de entrenamiento y tracking (`pipeline/ae/run.py`)

- [x] 5.1 Entry point que acepte arquitectura, `L`, `d`, `w_mp`, `w_aux`, `aux_input`, semilla y modo (búsqueda, final, referencia, DMD)
- [x] 5.2 Búsqueda aleatoria por arquitectura sobre pocas ventanas equiespaciadas (`_select_windows`), con early stopping; devolver la mejor configuración de cada arquitectura
- [x] 5.3 Entrenamiento final con todo lo anterior al test, 3 semillas, evaluación única en el test held-out
- [x] 5.4 Logging a MLflow: runs `sinca-zona-ae_{arq}-v{n}` con tags `modelo`, `tipo`, `stage` y sin `estacion`/`pollutant`; params, métricas `recon_*` y `gate_pass`; artefactos `reconstruction.parquet`, `latent.parquet` (latente `z` por hora y partición, con columnas por bloque `z_mp`/`z_aux`) y tabla de diagnósticos, más `encoder.keras` y `decoder_mp.keras` (por separado) como artefactos del run; sync de estado con `train/mlflow_log.py`
- [x] 5.5 Runs de referencia (`ref-mean`, `ref-pca-d{d}`) y runs por serie del pase DMD (`ae-{arq}-dmd`) con `predictions.parquet`/`zone_icap.parquet`/`rolling_scores.parquet` (mismo contrato que LightGBM), tag `ae_run_id` y métricas `dmd_latent_rmse_1`/`dmd_latent_rmse_24` en el run AE; persistir el operador ajustado (`dmd_operator.npz`: `K`, rango, autovalores) como artefacto para explorarlo sin TensorFlow
- [x] 5.6 Test del entry point con fixture local (corre de punta a punta sin AWS)
- [x] 5.7 Acceso del pase DMD al AE (ver `design.md` D7.6): (a) camino principal: encadenar el pase de DMD en el mismo job del entrenamiento final, solo si `gate_pass` es verdadero (flag `--dmd-after-final`, activo por defecto); (b) respaldo: en el modo `dmd` independiente, `--ae-run-id` descarga `encoder.keras` y `decoder_mp.keras` desde `sinca-mlflow/_mlflow/` reutilizando `analysis/load.py` (`ensure_state`/`artifact_path`), manteniendo `--ae-dir` para pruebas. Test con fixture local para ambos caminos

## 6. Imagen y lanzador

- [x] 6.1 Crear `docker/train-tf.Dockerfile` (solo runtime, `tensorflow-cpu`, patrón de `train.Dockerfile`; sin código ni `requirements.txt` ajeno en `source_dir`)
- [x] 6.2 Ajustar `train/launch.py` para elegir imagen (tag explícito), tipo de instancia y entry point; sin `:latest`
- [x] 6.3 Documentar en `docker/` el flujo de build/push con los tags `lgbm-v1` y `tf-cpu-v1`
- [x] 6.4 Mover `pipeline/requirements.txt` fuera de `source_dir` (`git mv pipeline/requirements.txt requirements-scraper.txt`), de modo que el toolkit de SageMaker no instale deps del scraper en los Training Jobs; confirmar que ningún Dockerfile, workflow ni doc lo referencia y actualizar lo que corresponda (ver `design.md` D10)

## 7. Dashboard

- [x] 7.1 `analysis/load.py`: tolerar runs sin `estacion`/`pollutant` y sin `feature_importance.parquet`; cargar `reconstruction.parquet` y las métricas `recon_*`/`dmd_*`. El filtro actual de runs de entrenamiento lee el **param** `tipo` (`!= "persistencia"`): excluir de las vistas por serie los runs de AE y de referencia (`ref-*`), y registrar `tipo` también como param en el logging de la tarea 5.4/5.5
- [x] 7.2 `analysis/streamlit_app.py` y `analysis/plots.py`: el flujo actual elige primero la serie y luego sus runs, y los runs de AE no tienen serie → agregar una **sección aparte** (selector de runs de AE) con el panel de reconstrucción (real vs reconstruido, razón de varianza, pendiente, espectro, error por nivel, estado de la compuerta) y el panel de linealizabilidad; los runs `ae-{arq}-dmd` entran por el flujo por serie existente
- [x] 7.3 Ampliar el fixture (`make_fixture.py`) con runs de AE y runs `ae-{arq}-dmd`; verificar degradación sin errores

## 8. Notebooks de exploración y documentación del experimento

Los notebooks complementan al dashboard: el dashboard es la vista fija de
resultados; los notebooks son para explorar libremente (scratch, no producción ni
fuente de verdad). Siguen la convención de `notebooks/analisis_baseline.ipynb`:
corren en JupyterLab del Space con `sinca-dev-role` (sin credenciales), leen los
artefactos persistidos de `sinca-data/validated/` y `sinca-mlflow/_mlflow/`.
Exploración en **dos niveles**: nivel 1 sin TensorFlow (parquet, numpy, plotly)
y nivel 2 con TensorFlow para lo que pasa por el decoder (ver `design.md` D9).
`requirements-analysis.txt` no suma TF; el nivel 2 usa `requirements-notebooks-tf.txt`.

- [x] 8.1 Crear `EXPERIMENT-LOG.md` con la plantilla de entrada por experimento (arquitectura, `L`, `d`, `w_aux`, métricas de compuerta, hallazgos)
- [x] 8.2 Registrar en `README.md` de `pipeline/ae/` cómo correr un experimento local en Docker y en Training Job
- [x] 8.3 `notebooks/exploracion_datos_ae.ipynb` (plantilla): explorar los datos de entrada del AE: cobertura y huecos por serie y canal, ventanas que sobreviven según `L`, rangos tras la transformación, efecto de la interpolación de ≤ 6 h y del recorte de aux
- [x] 8.4 `notebooks/exploracion_reconstruccion.ipynb` (plantilla): elegir runs de AE y explorar `reconstruction.parquet` — real vs reconstruido con zoom a episodios, diagnósticos de aplanamiento (varianza, pendiente, espectro, error por nivel), comparación contra `ref-mean` y `ref-pca-d*`, efecto de la ablación de `z_aux`
- [x] 8.5 `notebooks/exploracion_latente_dmd.ipynb` (plantilla), con dos secciones. **Nivel 1 (sin TF)**: `latent.parquet` y `dmd_operator.npz` — dimensiones activas, trayectoria de `z_mp` y `z_aux`, reajustar el DMD con otro rango, otra ventana temporal o regularización (numpy), espectro de `K` (autovalores → período y decaimiento), linealizabilidad por época del año. **Nivel 2 (con TF)**: cargar `encoder.keras`/`decoder_mp.keras` para llevar un pronóstico del DMD reajustado a µg/m³, recorrer una dimensión del latente y ver su efecto sobre las 6 series, y anular `z_aux` o un modo. Sirve de insumo exploratorio para `ae-dmd`
- [x] 8.6 Crear `requirements-notebooks-tf.txt` con `tensorflow-cpu` fijado en la **misma versión** que `docker/train-tf.Dockerfile`, separado de `requirements-analysis.txt`; documentar en el README cómo instalarlo en el venv del Space solo cuando se use el nivel 2
- [x] 8.7 Reutilizar `analysis/load.py` para leer el estado de MLflow desde los notebooks (sin duplicar la carga) y dejar los notebooks sin outputs ni execution counts (`jupyter nbconvert --clear-output --inplace`); documentar en `README.md` que se versionan con `git add -f` y el flujo encender/trabajar/apagar del Space

## 9. Ejecución real — preparación (con el operador)

- [x] 9.1 Verificar identidad AWS (`aws sts get-caller-identity --profile $AWS_PROFILE`), cuota `ml.g4dn.xlarge`/`ml.m5.large` for training job usage y existencia del repo ECR `sinca-train`
- [x] 9.2 Retaggear la imagen LightGBM existente como `lgbm-v1` (sin reconstruir) y ajustar la referencia en el lanzador; confirmar con una corrida corta de LightGBM que sigue funcionando
- [x] 9.3 Construir y subir la imagen `tf-cpu-v1` a `sinca-train`; confirmar el digest y que `lgbm-v1` sigue disponible
- [x] 9.4 En el Space (imagen `sagemaker-distribution-cpu`), verificar con output real si TensorFlow ya viene instalado y en qué versión (`pip list | grep -i -E "tensorflow|keras"`), y revisar disco/RAM disponibles; si no está o la versión difiere de `tf-cpu-v1`, instalar `requirements-notebooks-tf.txt` en el venv del Space y confirmar `import tensorflow`. Registrar en `EXPERIMENT-LOG.md`
- [ ] 9.5 Medir con datos reales cuántas ventanas sobreviven por `L ∈ {24, 72, 168}` y generar el reporte de rangos por canal; registrar en `EXPERIMENT-LOG.md` (si con `L = 168` sobreviven muy pocas, parar y volver al Arquitecto)

## 10. Ejecución real — humo y punto de control

- [ ] 10.1 Corrida de humo en CPU `ml.m5.large`: 1 ventana, pocas epochs, las 4 arquitecturas; medir tiempo por epoch y registrar métricas de reconstrucción y diagnósticos. Loguear al arrancar las versiones de `tensorflow`, `numpy` y `pandas` (confirmar que el toolkit no las cambia) y medir la memoria pico (`L = 168`, ~17 mil muestras; `ml.m5.large` tiene 8 GB)
- [ ] 10.2 Corridas de referencia (`ref-mean`, `ref-pca-d{d}`) sobre las mismas ventanas
- [ ] 10.3 **PUNTO DE CONTROL — vuelve al Arquitecto**: con los resultados del humo se fijan los umbrales numéricos de la compuerta, el presupuesto de trials, `d_aux` y el criterio CPU/GPU (el experimento MLflow ya está cerrado: `sinca-baseline`). No continuar al grupo 11 sin actualizar `design.md`/`tasks.md`

## 11. Ejecución real — experimentos

- [ ] 11.1 Búsqueda de hiperparámetros por arquitectura (una corrida por vez), con los parámetros fijados en el punto de control
- [ ] 11.2 Control sin meteo (`aux_input` apagado) y barrido de `w_aux ∈ {0, 0.1, 0.5, 1}` sobre la mejor arquitectura
- [ ] 11.3 Entrenamiento final de cada finalista (3 semillas), evaluación única en el test held-out y aplicación de la compuerta
- [ ] 11.4 Pase de DMD global (rango completo y truncado) sobre los AE aprobados, encadenado al entrenamiento final de 11.3 (tarea 5.7); si hace falta repetirlo sin reentrenar, usar el modo `dmd` con `--ae-run-id`. Registrar linealizabilidad y los runs `ae-{arq}-dmd` por serie
- [ ] 11.5 Registrar un resumen por experimento en `EXPERIMENT-LOG.md` (reconstrucción, aplanamiento, aporte de `z_aux`, linealizabilidad, pronóstico informativo vs persistencia/LightGBM)

## 12. Verificación funcional

- [ ] 12.1 Abrir el dashboard sobre el estado real de MLflow y confirmar con output real que: los runs de AE aparecen con compuerta, diagnósticos de aplanamiento y métricas `dmd_`; los runs `ae-{arq}-dmd` se comparan por serie contra persistencia y LightGBM; y los runs sin pronóstico no rompen el dashboard (adjuntar la salida/captura)
- [ ] 12.2 Abrir los notebooks de la tarea 8 en JupyterLab del Space con datos reales y correr `Run All`, confirmando con output real que cargan los artefactos (`reconstruction.parquet`, `latent.parquet`, `dmd_operator.npz`) y generan sus figuras, y que las celdas de nivel 2 cargan `encoder.keras`/`decoder_mp.keras` de un run real, decodifican y recorren una dimensión del latente; limpiar outputs y **apagar el Space** al terminar (Spaces → `sinca-dev` → Stop)

## Notas de implementación

<!-- El implementador registra acá, por tarea: qué encontró y qué hizo, cuando algo no calce con el blueprint o con lo planificado y no sea un bloqueo. -->

### Implementación (grupos 1–8 de código; ejecución real pendiente)

- **Alcance:** se implementó solo el código (grupos 1–8 de escritura). Quedan **sin marcar** las tareas de documentación `6.3`, `8.1`, `8.2` y la parte README de `8.7` (no son código), y todo el grupo 9–12 (AWS, Training Jobs, humo, punto de control, verificación en el Space). El punto de control `10.3` sigue siendo el corte para el Arquitecto.
- **Interpretación de `d`** (tarea 2.1, D2): `d` es la dimensión **total** del latente; `z_mp` tiene `d − d_aux` dimensiones y `z_aux` tiene `d_aux` (default `max(2, d/4)`). Así la comparación contra PCA(d) es justa. Confirmar con el Arquitecto.
- **Experimento MLflow** (5.4, Open Question): se usa `sinca-baseline` por defecto, configurable con `--experiment`. Pendiente de confirmación, como indica `design.md`.
- **Compuerta** (3.5): sin umbrales (`--gate-thresholds` vacío) la compuerta solo **reporta**: `gate_pass` no se loguea y sí `gate_beats_pca`. El escenario del spec "lo marca como no aprobado" solo puede aplicarse cuando se fijen umbrales en el punto de control. Pide decisión.
- **Nombres de archivo:** los encabezados de sección mencionan `ae/reference.py` y `ae/train.py`. Implementé las referencias y la compuerta en `ae/metrics.py`, y el entry point en `ae/run.py` (evita confundir con el paquete `train/`). Arquitecto: decidir si se renombra o se actualizan los encabezados.
- **Partición held-out** (1.5): se usa `TEST_START`/`TEST_END` y `make_rolling_windows`/`_select_windows` de `features.build`, pero no `final_holdout_split` (opera sobre el DataFrame de features; el corte es el mismo).
- **Early stopping del modelo final** (5.3): igual que el baseline, se usa la última ventana `valid` del rolling, que queda **dentro** del train del modelo final. Es un sesgo menor que conviene anotar al comparar.
- **DMD truncado** (5.5): los runs por serie `ae-{arq}-dmd` usan solo el rango completo. La variante truncada se evalúa en linealizabilidad (`dmd_trunc_latent_rmse_*`) y se persiste en `dmd_operator_trunc.npz`. Los `rolling_scores.parquet` de esos runs van vacíos (el pase DMD no tiene búsqueda).
- **Modo `dmd` y artefactos** (5.5, riesgo de diseño): el modo `dmd` lee `encoder.keras`/`decoder_mp.keras` desde un **directorio local** (`--ae-dir`). Los artefactos sí quedan en MLflow, pero un Training Job nuevo no los tiene en disco. Antes de 11.4 hay que decidir cómo se descargan (`load.artifact_path` o `mlflow.artifacts`).
- **Memoria** (1.5/2.x): una muestra con `L = 168` y 34 canales auxiliares pesa ~22 KB en float32; con ~17 mil muestras son ~390 MB, más copias al indexar. Verificar en el humo (`ml.m5.large`, 8 GB).
- **Dependencias en `source_dir`** (6.1): existe `pipeline/requirements.txt`. El toolkit de `sagemaker-training` lo instala dentro del contenedor al arrancar, y la imagen TF no lo necesita. Esto contradice la intención de 6.1 ("sin requirements.txt ajeno en source_dir") y ya afecta a la imagen LightGBM. No lo modifiqué: decidir si se mueve fuera de `pipeline/`.
- **Tests:** el repo no tenía suite. Creé `tests/ae/` en la raíz (fuera de `source_dir`). Instalé `pytest` y `tensorflow-cpu==2.20.0` en el `.venv` local. **52 tests pasan** en Windows (Python 3.13, TF 2.20.0): datos (1.8), modelos (2.5), métricas/DMD (3.6, 4.5) y punta a punta sin AWS (5.6, incluye `search`, `final`, `reference` y `dmd`).
- **Docker** (6.1): `docker/train-tf.Dockerfile` escrito, pero **no construido** (Docker no se ejecutó en esta sesión). Requiere verificación al hacer el build de `tf-cpu-v1` (9.3).
- **Lanzador** (6.2): `train/launch.py` acepta `--entry-point` (`train/train.py` o `ae/run.py`) y `--hp CLAVE=VALOR`, y **rechaza** imágenes sin tag o con `:latest`. El comportamiento de LightGBM no cambia.
- **Dashboard** (7.1–7.3): `load.is_series_run` excluye persistencia, referencias y runs conjuntos de AE de las vistas por serie. Se agregó la vista "Autoencoders (reconstrucción)" en `streamlit_app.py` y las figuras en `analysis/plots_ae.py` (módulo nuevo, `plots.py` intacto). El fixture se regeneró con 27 runs. Verificado con `AppTest` en ambas vistas, sin excepciones.
- **Notebooks** (8.3–8.5): plantillas JSON sin outputs ni contadores, leen vía `analysis/load.py`. Las celdas de nivel 1 de 8.4 y 8.5 se ejecutaron contra el fixture. 8.3 necesita datos validados reales (S3), no ejecutado.

### Decisiones del Arquitecto (respuesta a las notas anteriores)

- **`d` total** (2.1): aprobado; `z_mp = d − d_aux`, `z_aux = d_aux` (`max(2, d/4)`). Reflejado en `design.md` D2.
- **Experimento MLflow** (5.4): cerrado en `sinca-baseline`. Sin cambios de código.
- **Compuerta** (3.5): `gate_pass` se registra siempre con la capa relativa (supera al promedio y a PCA(d)); la capa numérica se suma tras el punto de control 10.3. **Tarea nueva 3.7.**
- **Modo `dmd` / artefactos** (5.5): camino principal encadenado en el mismo job tras el entrenamiento final si pasa la compuerta; respaldo con `--ae-run-id` descargando de `_mlflow/`. **Tarea nueva 5.7.**
- **`pipeline/requirements.txt`** (6.1): se mueve fuera de `source_dir` a `requirements-scraper.txt` (verificado que nada lo referencia). **Tarea nueva 6.4.** En el humo (10.1) se loguean versiones de `tensorflow`/`numpy`/`pandas` y la memoria pico.
- **Nombres de archivo**: aprobados `ae/metrics.py` (referencias y compuerta) y `ae/run.py` (entry point); encabezados de los grupos 3 y 5 actualizados.
- **Sin acción**: early stopping con la última ventana `valid` dentro del train final (sesgo menor, igual que el baseline; anotar en `EXPERIMENT-LOG.md`); DMD truncado solo en linealizabilidad y `rolling_scores` vacíos en los runs DMD (reflejado en D7); no usar `final_holdout_split` (mismo corte); Docker sin construir (se cubre en 9.3).

### Decisiones del Arquitecto aplicadas (3.7 y 5.7)

- **3.7 compuerta en dos capas**: `gate_pass` se registra siempre. Capa relativa = el AE supera en RMSE de MP, en **todas** las series, al predictor del promedio y a PCA(d) (sobre valid). Campos: `relative`, `beats_pca`, `beats_mean` (+ por serie). La capa numérica (`--gate-thresholds`) se suma solo si hay umbrales; sin umbrales `numeric` es `None` y no se loguea `gate_numeric`. Commit `2fc3dfb`.
- **5.7 pase DMD encadenado** (D7.6, camino principal): tras el entrenamiento final, la semilla 0 encadena el pase DMD en el mismo job si `gate_pass` es verdadero. Flag `--dmd-after-final` (default `true`); si la compuerta no aprueba, se loguea y no se corre.
- **5.7 modo `dmd` (respaldo)**: con `--ae-run-id` y sin `--ae-dir`, la configuración sale de los params del run, la compuerta de su métrica `gate_pass`, y `encoder.keras`/`decoder_mp.keras` se leen del estado de MLflow con `analysis.load` (`ensure_state(data_dir=--mlflow-dir)` + `artifact_path`). `--ae-dir` se mantiene para pruebas. Rechaza el pase si la compuerta no aprobó.
- **Dependencia operativa a vigilar**: `ensure_state(data_dir=...)` no descarga; el estado lo baja `run()` desde `sinca-mlflow/_mlflow/` antes de usar `--mlflow-dir`. Con `--no-upload` el directorio debe existir con el estado previo. Un job `dmd` en SageMaker depende de que el descargue ocurra (solo si no hay `--no-upload`).
- **Tests**: el AE de prueba (2 épocas) no supera al promedio ni a PCA, así que los tests que necesitan el pase fuerzan la aprobación con `monkeypatch` sobre `metrics.gate_decision`. El caso de rechazo usa la compuerta real. **57 tests pasan**.
- **Sin cambios en `design.md`.**

### Tareas 6.4 y documentación (cierre de código)

- **6.4**: `pipeline/requirements.txt` movido con `git mv` a `requirements-scraper.txt` (raíz). Verificado con `git grep` que nada lo referencia: ni workflows de `.github/`, ni Dockerfiles (`scraper.Dockerfile` instala sus deps inline y solo hace `COPY pipeline/`), ni README. Quedan menciones solo en documentación histórica (`archive/`, `modules/compute/blueprint.md`, `design.md`) que no se modifican. `requirements-dev.txt` tiene un comentario genérico sobre el comportamiento del toolkit, que sigue siendo válido.

- **6.3**: `docker/README.md` documenta build, push y retag con tags explícitos (`lgbm-v1`, `tf-cpu-v1`), la regla de no usar `:latest` y que `sinca-train` es `MUTABLE` (por eso no se reusa un tag publicado). El retag de la imagen LightGBM (9.2) queda descrito como paso del operador, sin ejecutarlo. En `README.md`, el bloque de build de entrenamiento dejó de publicar `:latest` y enlaza a esa guía; los comandos reales de AWS siguen pendientes de ejecución (grupo 9).

- **8.1**: `EXPERIMENT-LOG.md` ya tenía la plantilla general y la entrada de baseline. Se agregó la plantilla específica de autoencoders (arquitectura, `L`, `d`, `w_aux`, compuerta relativa y numérica, reconstrucción, latente, `z_aux`, DMD, comparación contra persistencia y LightGBM). No se anotan resultados: se llenan con valores reales de MLflow tras el humo y los experimentos.

- **8.2**: `pipeline/ae/README.md` documenta los módulos, los tests sin AWS, el experimento local (con `--no-upload`), la ejecución en Docker con `--entrypoint python` sobre la imagen `tf-cpu-v1`, y el Training Job por `launch.py --entry-point ae/run.py --hp ...` con el orden search → final → dmd → reference. Los comandos se derivan de los flags reales de `run.py` y `launch.py`; no se ejecutaron contra AWS.

- **8.7 (parte documental)**: el README raíz tiene una sección de notebooks de exploración del autoencoder: lectura vía `analysis/load.py`, versionado con `git add -f` (porque `*.ipynb` está ignorado), notebooks sin outputs (`nbconvert --clear-output`), instalación de TF solo para nivel 2, y el flujo encender/trabajar/apagar del Space. La parte de código de 8.7 (reutilizar `load.py`, notebooks sin outputs) se había cumplido antes.

### Estado tras el cierre de código (supersede el bullet "Alcance" de arriba)

- Código y documentación de los grupos 1–8 completos, incluidas 3.7, 5.7, 6.3, 6.4, 8.1, 8.2 y la parte documental de 8.7.
- **Pendiente solo el grupo 9–12** (AWS, Training Jobs, humo, punto de control 10.3, verificación en el Space). Son ejecución real con el operador.
- Pendientes de operador anotados en las notas: retag `lgbm-v1` (9.2, procedimiento en `docker/README.md`), build y push de `tf-cpu-v1` (9.3), y verificación en el humo (10.1) de que el Training Job ya no instala dependencias del scraper y de las versiones de `tensorflow`/`numpy`/`pandas`.

### Ejecución real — grupo 9 (en curso)

- **9.1 (completa):** identidad `549024266383` (coincide con el bucket de tfstate de la cuenta personal). Cuotas: `ml.m5.large` = 20, `ml.g4dn.xlarge` = 1. Repo `sinca-train` existe (MUTABLE, scan on push).
- **9.2 (completa):** corrida corta de LightGBM con `lgbm-v1` (`sinca-baseline-train-2026-11-02-20-25-47-337`): `Completed` en 275 s, 6 series entrenadas, persistencia calculada, artefactos y estado MLflow subidos (38 → 75 objetos). **Efecto colateral a limpiar:** los runs de prueba quedaron en `sinca-baseline` con versión `99`; borrarlos antes de 12.1 para que no aparezcan en el dashboard. Entorno: `.venv` del proyecto (SageMaker SDK 2.257.7), Python local.
- **9.2 (detalle del retag):** confirmado con Docker local que `latest` = digest `sha256:31bbdf92…` (la imagen LightGBM). Se creó el tag `lgbm-v1` con `aws ecr put-image` sobre el manifest de ese digest. ECR re-serializa el índice multi-arquitectura al guardarlo, así que el digest del índice de `lgbm-v1` (`sha256:492de5f7…`) difiere del de `latest`. Los manifests hijos son idénticos en ambos (`amd64` `sha256:1c2c504…`, attestation `sha256:12c63cd…`). El tag `lgbm-v1` se sobrescribió 3 veces durante el intento; el estado final es el digest `492de5f7…`. No hizo falta cambiar código: `train/launch.py` ya recibe `--image-uri` con tag explícito y sus ejemplos usan `lgbm-v1`. Falta: corrida corta de LightGBM con `:lgbm-v1` para cerrar 9.2.
- **9.3 (completa):** push de `tf-cpu-v1` OK, digest `sha256:351f5ed8…` (igual al manifest list del build local). `lgbm-v1` (`sha256:492de5f7…`) y `latest` (`sha256:31bbdf92…`) sin cambios. Detalle del diseño: `docker/train-tf.Dockerfile` no resolvía: `sagemaker-training` fija `protobuf<=3.20.3` y `tensorflow-cpu 2.20` exige `protobuf>=5.28`. Solución aplicada: `sagemaker-training` con `--no-deps` y sus deps de runtime sin protobuf (desviación de 6.1, aprobada por el operador). Build local OK; imports OK (tf 2.20.0, protobuf 7.36.2). `pip check` reporta solo ese conflicto, esperado. Falta: validar el entry point `train` en el humo 10.1 sobre SageMaker. Push y digest pendientes.
- **Tag `:latest` de `sinca-train`:** se conserva apuntando a LightGBM sin cambios; la regla de no usar `:latest` en los lanzamientos sigue vigente.
