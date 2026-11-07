> Orden: código (grupos 1–8, sin AWS) → ejecución real (9–12), donde el operador corre los comandos y devuelve el output. La última tarea de ejecución es una verificación funcional.

## 1. Datos para red (`pipeline/ae/data.py`)

- [x] 1.1 Cargar validados con `features/build.load_validated`; tabla horaria de las 6 series MP sobre grilla horaria completa
- [x] 1.2 Canales aux: `pres`, `wspd`, `rain` × 3 estaciones; `temp` de Ñielol y Las Encinas; `blh` (sin `rhum`, `glob`, `temp` de PLC II)
- [x] 1.3 Calendario: Fourier diaria/semanal/anual sin/cos, `dow_sin/cos`, `is_holiday`, `gec_window` (constantes de `features/build.py`)
- [x] 1.4 Interpolar huecos ≤ 6 h; máscara de valores reales; en aux, huecos > 6 h = 0 + columna de máscara
- [x] 1.5 Ventanas `[L × 6]`, L ∈ {24, 72, 168}; descartar ventanas MP con huecos > 6 h; asignar a partición solo si la ventana completa cae en ella (`make_rolling_windows`, `final_holdout_split`, `_select_windows`)
- [x] 1.6 `log1p` en pm25, pm10, rain, blh, wspd; z-score con estadísticas solo de train; recorte ±5 solo en aux; inversión a µg/m³
- [x] 1.7 Reporte de rangos por canal (mín, máx, percentiles, fracción recortada)
- [x] 1.8 Tests con fixture sintético: interpolación, descarte, máscaras, asignación, estadísticas solo de train, sin recorte de MP

## 2. Modelos y loss (`pipeline/ae/models.py`)

- [x] 2.1 AE en Keras: encoder MP, rama aux, latente `z=[z_mp | z_aux]` (d ∈ {8,16,32}; d_aux = d/4 por defecto, ≥ 2); decoder MP sobre el latente completo; decoder aux; salida lineal
- [x] 2.2 Variantes: `mlp`, `conv1d`, `gru` (opción `lstm`), `linear_decoder`
- [x] 2.3 Interruptor `aux_input` (apagado = rama aux solo con calendario)
- [x] 2.4 Loss `w_mp·E_mp + w_aux·E_aux`: MSE enmascarado, promedio por canal, calendario sin loss; compile/train_step según Keras 3
- [x] 2.5 Tests: formas por variante, enmascarado, `w_aux = 0` anula el término aux

## 3. Referencias, métricas y compuerta (`pipeline/ae/metrics.py`)

- [x] 3.1 Referencias: predictor del promedio y PCA con el mismo d, mismas métricas
- [x] 3.2 Métricas por serie en µg/m³: RMSE, razón de varianza, pendiente real vs reconstruido, espectro, RMSE y sesgo por tramo (< 50, 50–100, > 100), captura de máximo, percentiles altos
- [x] 3.3 Diagnósticos del latente: dimensión efectiva (participación de PCA de z) y dimensiones muertas
- [x] 3.4 Ablación de `z_aux` (anular el bloque y medir el empeoramiento de MP)
- [x] 3.5 Compuerta dura sobre MP con umbrales parametrizables (por defecto solo reporta)
- [x] 3.6 Tests con datos conocidos (señal suavizada → razón de varianza < 1; reconstrucción perfecta → pendiente 1)
- [x] 3.7 Compuerta en dos capas (D5): `gate_pass` siempre con la capa relativa (supera al promedio y a PCA(d)); registrar `gate_beats_mean` y `gate_beats_pca`; la capa numérica se suma al fijar umbrales en 10.3. Actualizar `gate_decision` y tests
- [x] 3.8 Compuerta ajustada en 10.3 (D5): (a) relativa = `beats_mean` estricto en las 6 series y `beats_pca` en el agregado (media de razones RMSE_AE/RMSE_PCA < 1), con razones por serie registradas; (b) numérica con defaults en código `{min_var_ratio 0.80, slope_tol 0.20, max_rmse_vs_pca 1.02}`, registrados en el param `gate_thresholds`; `--gate-thresholds` solo sobreescribe (sin JSON por `--hp`); `gate_pass = relative ∧ numeric`. Tests: empate dentro de 2 % pasa, serie 3 % peor falla, razón media ≥ 1 falla, sin PCA/promedio falla
- [x] 3.9 Umbrales de 10.5 (D5): `DEFAULT_GATE_THRESHOLDS = {min_var_ratio 0.75, slope_tol 0.30, max_rmse_vs_pca 1.02}`; tests: var 0.80 con pendiente 0.74 pasa; var 0.70 o pendiente 0.65 falla. Sin otros cambios

## 4. DMD (`pipeline/ae/dmd.py`)

- [x] 4.1 Extraer z_t del AE congelado para cada hora de la partición
- [x] 4.2 DMD exacto global por mínimos cuadrados sobre (z_t, z_{t+1}); rango completo y truncado (`numpy`/`tf.linalg`)
- [x] 4.3 Linealizabilidad: error de z a 1 y 24 pasos (K, K^24) en valid/test
- [x] 4.4 Pronóstico informativo: K^24 z_t → decoder_mp → promedio de las últimas 24 h decodificadas, en µg/m³, alineado con el target de `features/build.py`
- [x] 4.5 Tests: sistema lineal sintético (K recuperado, linealizabilidad ≈ 0); promedio = mean(MP(t+1…t+24h)) en caso trivial

## 5. Entry point y tracking (`pipeline/ae/run.py`)

- [x] 5.1 Entry point: arquitectura, L, d, w_mp, w_aux, aux_input, semilla, modo (search, final, reference, dmd)
- [x] 5.2 Búsqueda aleatoria por arquitectura sobre pocas ventanas equiespaciadas (`_select_windows`), con early stopping; mejor config por arquitectura
- [x] 5.3 Entrenamiento final con todo lo anterior al test, 3 semillas, evaluación única en test
- [x] 5.4 Logging MLflow: runs `sinca-zona-ae_{arq}-v{n}`, tags `modelo`, `tipo`, `stage`, sin `estacion`/`pollutant`; params, `recon_*`, `gate_pass`; artefactos `reconstruction.parquet`, `latent.parquet` (bloques z_mp/z_aux), tabla de diagnósticos, `encoder.keras` y `decoder_mp.keras` por separado; sync con `train/mlflow_log.py`
- [x] 5.5 Referencias (`ref-mean`, `ref-pca-d{d}`) y runs DMD por serie (`ae-{arq}-dmd`) con `predictions.parquet`/`zone_icap.parquet`/`rolling_scores.parquet` (contrato LightGBM), tag `ae_run_id`, `dmd_latent_rmse_1/24` en el run AE, `dmd_operator.npz` (K, rango, autovalores)
- [x] 5.6 Test del entry point con fixture local, sin AWS
- [x] 5.7 Pase DMD: (a) encadenado al entrenamiento final solo si `gate_pass` (flag `--dmd-after-final`, activo por defecto); (b) respaldo: modo `dmd` con `--ae-run-id` descarga `encoder.keras`/`decoder_mp.keras` de `sinca-mlflow/_mlflow/` con `analysis/load.py` (`ensure_state`/`artifact_path`); `--ae-dir` para pruebas. Tests con fixture para ambos caminos
- [x] 5.8 Curva de aprendizaje: `loss` y `val_loss` por epoch (con step) y `best_epoch`; mantener `epochs_run`. Test con fixture
- [x] 5.9 Búsqueda según D6: espacio por arquitectura (d = 16; L ∈ {24, 72}; GRU solo L = 24, hidden {32, 64}; batch {128} en conv1d/GRU); L = 168 aceptado por CLI y datos pero fuera del espacio. Trials por defecto `{linear_decoder 12, mlp 12, conv1d 6, gru 4}` (`--search-trials` sobreescribe). Score = media por serie de RMSE_AE/RMSE_PCA(d, L) en valid, junto al RMSE absoluto. `--search-time-budget` (s, default 10800). `ae_search.json` incremental. Tests: restricciones del espacio, score como razón, tope de tiempo, JSON válido

## 6. Imagen y lanzador

- [x] 6.1 `docker/train-tf.Dockerfile`: solo runtime, `tensorflow-cpu`, patrón de `train.Dockerfile`; sin código ni requirements ajenos en `source_dir`
- [x] 6.2 `train/launch.py`: imagen con tag explícito, tipo de instancia y entry point; sin `:latest`
- [x] 6.3 Documentar en `docker/` el build/push con tags `lgbm-v1` y `tf-cpu-v1`
- [x] 6.4 `git mv pipeline/requirements.txt requirements-scraper.txt` (fuera de `source_dir`, D10); confirmar que nada lo referencia

## 7. Dashboard

- [x] 7.1 `analysis/load.py`: tolerar runs sin `estacion`/`pollutant` y sin `feature_importance.parquet`; cargar `reconstruction.parquet` y métricas `recon_*`/`dmd_*`. Excluir de las vistas por serie los runs AE y `ref-*` (filtro por param `tipo`); registrar `tipo` como param en 5.4/5.5
- [x] 7.2 `analysis/streamlit_app.py` y `analysis/plots.py`: sección aparte para runs AE (reconstrucción, razón de varianza, pendiente, espectro, error por nivel, compuerta, linealizabilidad); runs `ae-{arq}-dmd` por el flujo existente por serie
- [x] 7.3 Ampliar `make_fixture.py` con runs AE y `ae-{arq}-dmd`; verificar que el dashboard degrada sin errores

## 8. Notebooks y documentación del experimento

Los notebooks son exploración scratch, no producción. Siguen `analisis_baseline.ipynb` (JupyterLab del Space, `sinca-dev-role`, lee `validated/` y `_mlflow/`). Dos niveles (D9): nivel 1 sin TF; nivel 2 con TF. `requirements-analysis.txt` no incluye TF.

- [x] 8.1 `EXPERIMENT-LOG.md` con plantilla por experimento
- [x] 8.2 `pipeline/ae/README.md`: experimento local en Docker y Training Job
- [x] 8.3 `notebooks/exploracion_datos_ae.ipynb`: cobertura y huecos, ventanas por L, rangos, efecto de interpolación y recorte aux
- [x] 8.4 `notebooks/exploracion_reconstruccion.ipynb`: real vs reconstruido con zoom, aplanamiento, comparación con `ref-mean`/`ref-pca-d*`, ablación de `z_aux`
- [x] 8.5 `notebooks/exploracion_latente_dmd.ipynb`: nivel 1 (latent, `dmd_operator.npz`, reajuste de DMD, espectro de K, linealizabilidad por época); nivel 2 (encoder/decoder: pronóstico a µg/m³, recorrer una dimensión del latente, anular z_aux o un modo)
- [x] 8.6 `requirements-notebooks-tf.txt` con `tensorflow-cpu` en la misma versión que `train-tf.Dockerfile`; instalación en el README solo para nivel 2
- [x] 8.7 Notebooks vía `analysis/load.py`, sin outputs (`nbconvert --clear-output`); README: versionado con `git add -f` y flujo encender/trabajar/apagar del Space

## 9. Ejecución real — preparación (operador)

- [x] 9.1 Identidad AWS (`sts get-caller-identity --profile $AWS_PROFILE`), cuota `ml.g4dn.xlarge`/`ml.m5.large` para training, repo ECR `sinca-train`
- [x] 9.2 Retag de la imagen LightGBM como `lgbm-v1` (sin reconstruir); ajustar lanzador; corrida corta de LightGBM
- [x] 9.3 Build y push de `tf-cpu-v1` a `sinca-train`; confirmar digest y que `lgbm-v1` sigue disponible
- [x] 9.4 Space: verificar TF preinstalado y su versión, disco y RAM; si difiere, instalar `requirements-notebooks-tf.txt` en un venv y confirmar `import tensorflow`. Registrar en `EXPERIMENT-LOG.md`
- [x] 9.5 Ventanas que sobreviven por L ∈ {24, 72, 168} y reporte de rangos con datos reales; registrar. Si L = 168 deja muy pocas, parar y volver al Arquitecto

## 10. Ejecución real — humo y punto de control

- [x] 10.1 Humo en CPU `ml.m5.large`: 1 ventana, pocas epochs, 4 arquitecturas; tiempo por epoch, métricas, diagnósticos; loguear versiones de tensorflow/numpy/pandas; medir RSS pico (L = 168). **Cerrada con hallazgo:** L = 168 no corrió (OOM); ver notas.
- [x] 10.2 Referencias (`ref-mean`, `ref-pca-d{d}`) sobre las mismas ventanas
- [x] 10.3 **PUNTO DE CONTROL** (decidido por el Arquitecto; lo marca el Implementador). Ver "Decisiones del Arquitecto"
- [x] 10.4 **Calibración de la compuerta** (humo largo, CPU): `version = 0`, L = 24, d = 16, hidden 64, dropout 0.1, lr 1e-3, batch 128, w_aux 0.1, `--epochs 60 --patience 8`. conv1d, mlp y linear_decoder con seeds 0,1,2; gru con seed 0. DMD encadenado solo en conv1d. `--max-run 14400`, una corrida a la vez. Reportar por arquitectura y serie, en valid y test: RMSE AE vs PCA(16) y razón, var_ratio, pendiente, ablación de `z_aux`, dispersión entre semillas, `best_epoch`, mejora de loss entre epochs 40 y 60, tiempo por epoch, RSS; y linealizabilidad y pronóstico DMD si corrió
- [x] 10.5 **PUNTO DE CONTROL CORTO** (decidido por el Arquitecto; lo marca el Implementador). Ver "Decisiones del Arquitecto"

## 11. Ejecución real — experimentos

- [x] 11.1 Búsqueda por arquitectura (D6, valores de 5.9): 4 Training Jobs `--mode search --archs <arq>` en paralelo; 4 ventanas, 15 epochs, patience 8, `--max-run 14400`. Recuperar `ae_search.json` de S3 y registrar en `EXPERIMENT-LOG.md` la mejor config, su score y el historial. Si un job termina por el tope con < 3 trials válidos o falla por memoria, volver al Arquitecto
- [x] 11.2 Sobre conv1d (menor score de 11.1), L = 24, hidden 64, dropout 0.1, lr 1e-3, batch 128; seed 0; 60 epochs; `--dmd-after-final false`; `version = 2`. Corridas: control sin meteo (`aux_input` off), barrido `w_aux ∈ {0, 0.1, 0.5, 1}`, barrido `d ∈ {8, 32}` (d_aux = max(2, d/4)), referencias `ref-mean` y `ref-pca-d{8,32}` a L = 24. `d_aux` no se barre. Desviación respecto de la "ganadora" (L = 72): ver D6
- [x] 11.3 Requiere 10.5 y 3.9. Entrenamiento final de 5 configuraciones: mejor de cada arquitectura de 11.1 (conv1d, linear_decoder y mlp a L = 72; gru a L = 24; ganador como `--hp`) más conv1d a L = 24 (control de L). `--seeds 0,1,2`, `--epochs 60`, `--patience 8`, umbrales de 3.9, `--dmd-after-final true`, `version = 1`, `--max-run 14400`, una corrida a la vez. Referencias `reference` (`version = 1`) para (L, d) ∈ {(24, 16), (72, 16)}. Evaluación única en test. **Antes de 11.4:** reportar por finalista la razón media AE/PCA en test y en valid (D5, "Señal de alerta"); si aprueba en valid con razón de test ≥ 1, anotar "aprobada solo en valid" y llevar al Arquitecto. Si ningún finalista pasa, volver al Arquitecto
- [x] 11.4 Pase DMD global (rango completo y truncado) sobre los AE aprobados, encadenado a 11.3 (5.7); si hace falta repetir sin reentrenar, modo `dmd` con `--ae-run-id`. Registrar linealizabilidad y runs `ae-{arq}-dmd` por serie
- [x] 11.5 Resumen por experimento en `EXPERIMENT-LOG.md`: reconstrucción, aplanamiento, aporte de `z_aux`, linealizabilidad, pronóstico vs persistencia y LightGBM. Incluir: compuerta en valid vs test; var_ratio y pendiente por serie vs `ref_pca_*` en valid y test (anotar series donde el AE aplane más que PCA(16) en más de 0.05); dispersión entre semillas; sesgo de `log1p` por tramo (para la pregunta abierta de D)

## 12. Verificación funcional

- [x] 12.1 Dashboard sobre el estado real de MLflow: runs AE con compuerta, diagnósticos y métricas `dmd_`; runs `ae-{arq}-dmd` comparables por serie contra persistencia y LightGBM; runs sin pronóstico no rompen el dashboard. Adjuntar salida o captura
- [ ] 12.2 Notebooks de la tarea 8 en JupyterLab del Space con datos reales, `Run All`: cargan artefactos (`reconstruction.parquet`, `latent.parquet`, `dmd_operator.npz`) y generan figuras; celdas de nivel 2 cargan `encoder.keras`/`decoder_mp.keras` de un run real, decodifican y recorren una dimensión del latente. Limpiar outputs y **apagar el Space** (Spaces → `sinca-dev` → Stop)

## Notas de implementación

Qué se encontró y qué se hizo cuando algo no calza con lo planificado. Las decisiones del Arquitecto van en su sección.

### Grupos 1–8 (código)

- **Alcance:** código completo de 1–8, incluidas 3.7, 5.7, 6.3, 6.4, 8.1, 8.2 y la parte documental de 8.7.
- **`d` total** (2.1, D2): `z_mp = d − d_aux`, `z_aux = d_aux = max(2, d/4)`. Aprobado.
- **Compuerta** (3.5/3.7): sin umbrales solo reporta; `gate_pass` se registra siempre con la capa relativa; `numeric` es `None` sin umbrales. Commit `2fc3dfb`.
- **Partición** (1.5): usa `TEST_START`/`TEST_END` y `make_rolling_windows`/`_select_windows` de `features.build`, no `final_holdout_split` (mismo corte).
- **Early stopping** (5.3): usa la última ventana valid, que queda dentro del train final (sesgo menor, igual que el baseline).
- **DMD truncado** (5.5): runs por serie solo con rango completo; truncado en `dmd_trunc_latent_rmse_*` y `dmd_operator_trunc.npz`. `rolling_scores.parquet` vacío en runs DMD.
- **Modo `dmd`** (5.5/5.7): lee `encoder.keras`/`decoder_mp.keras` de `--ae-dir` o, con `--ae-run-id`, del estado de MLflow (`ensure_state(data_dir=--mlflow-dir)` + `artifact_path`). `ensure_state` no descarga por sí solo; con `--no-upload` el directorio debe existir. Rechaza el pase si `gate_pass` es falso.
- **Dependencias** (6.4): `pipeline/requirements.txt` movido a `requirements-scraper.txt` con `git mv`; `git grep` confirma que nada lo referencia.
- **Tests:** `tests/ae/` fuera de `source_dir`; 68 pasan.
- **Docker/lanzador** (6.1–6.3): `train-tf.Dockerfile` escrito y construido en 9.3; `launch.py` acepta `--entry-point` y `--hp`, y rechaza imágenes sin tag o con `:latest`; `docker/README.md` documenta build, push y retag.
- **Dashboard** (7.1–7.3): `load.is_series_run` excluye persistencia, referencias y runs AE; vista "Autoencoders" en `streamlit_app.py` y `analysis/plots_ae.py`. Fixture con 27 runs; verificado con `AppTest`.
- **Notebooks** (8.3–8.5): plantillas sin outputs; celdas de nivel 1 de 8.4 y 8.5 ejecutadas sobre fixture; 8.3 pendiente de datos reales.
- **8.2:** comandos del README derivados de los flags reales; no ejecutados contra AWS.
- **Experimento MLflow** (5.4): `sinca-baseline`, configurable con `--experiment`.

### Grupo 9

- **9.1:** cuenta `549024266383`; cuotas `ml.m5.large` = 20, `ml.g4dn.xlarge` = 1; repo `sinca-train` (MUTABLE).
- **9.2:** retag `lgbm-v1` con `aws ecr put-image`. Digest del índice `sha256:492de5f7…` (los hijos son idénticos). Corrida corta de LightGBM `Completed` en 275 s. **Efecto colateral:** runs de prueba `version = 99` en `sinca-baseline`, pendientes de limpieza.
- **9.3:** `tf-cpu-v1` con digest `sha256:351f5ed8…`. `sagemaker-training` con `--no-deps` porque `protobuf<=3.20.3` choca con `tensorflow-cpu` 2.20 (`protobuf>=5.28`); aprobado por el operador. `import tensorflow` OK (2.20.0).
- **9.4:** la imagen del Space trae TF 2.18.0 y keras 3.10.0. Instalar sobre el Python base rompe dependencias y no persiste tras reiniciar. Solución: venv aislado `/home/sagemaker-user/venv-tf` con `requirements-notebooks-tf.txt` (tf 2.20.0, keras 3.15.1), que sí persiste. Falta registrar el kernel de JupyterLab (12.2).
- **9.5:** ventanas train/test: L = 24 43.051/1.999; L = 72 39.414/1.761; L = 168 33.669/1.377.
- **Incidente de cuenta:** la primera corrida usó `AWS_PROFILE` de otra cuenta (`457469184690`) y dio `AccessDenied`. No hubo lecturas ni escrituras; se repitió con `desareca_dev` verificado. El `AWS_PROFILE` de la shell apunta a una cuenta ajena: usar siempre `--profile desareca_dev`.
- El tag `:latest` de `sinca-train` se conserva en LightGBM; la regla de no usarlo en lanzamientos sigue vigente.

### Grupo 10

- **Bugs de CLI:** SageMaker pasa `-d 16`; `run.py` acepta forma corta y larga. Los nombres de job no admiten `_`: `linear_decoder` se lanza como `linear-decoder`.
- **10.1 humo** (3 epochs, L = 24 salvo indicación):

  | Arq | s/epoch | Total | RSS pico | RMSE test medio |
  |---|---|---|---|---|
  | linear_decoder L=24 | 2.2 | 134 s | 1.8 GB | 19.4 |
  | mlp L=24 | 3.0 | 119 s | 1.8 GB | 19.7 |
  | conv1d L=24 | 16.0 | 162 s | 1.8 GB | 16.2 |
  | gru L=24 | 42.7 | 245 s | 1.8 GB | 21.3 |
  | gru L=72 | 121.3 | 506 s | 3.5 GB | 35.8 |
  | gru L=168 | — | — | OOM (8 GB) | — |

  Runtime: TF 2.20.0, Keras 3.15.1, NumPy 2.4.6, pandas 2.3.3. Compuerta 0 en todas (3 epochs, sin conclusiones de calidad).
- **10.2:** RMSE test medio: promedio 44.8 (L = 24) / 45.6 (L = 72); PCA(16) 17.1 / 22.8.
- **10.4:** conv1d (3 semillas, DMD en seed 0), mlp (3), linear_decoder (3), gru (1); todos `Completed`. Razón media AE/PCA(16) en valid: conv1d 0.895 (desv. entre semillas 1.64 %), mlp 1.081, linear_decoder 1.097, gru 1.024. Fallos de conv1d por pendiente o var_ratio, no por RMSE.

### Grupo 11

- **11.1** (score = razón vs PCA(16), valid): conv1d 1.02 (6 trials; L = 72, hidden 64, dropout 0.1, batch 128); linear_decoder 1.04 (12; L = 72, hidden 32, dropout 0, batch 64); mlp 1.05 (12; L = 72, hidden 32, dropout 0, batch 64); gru 1.16 (4; L = 24, hidden 64, dropout 0.1, batch 128). Ningún trial gana a PCA(16). Arquitectura para 11.2: conv1d. Todos dentro del tope de tiempo.
- **Discrepancia de `lr` en conv1d:** la nota de 11.1 decía `1e-3`, pero `ae_search.json` dice `0.003`. 11.2 usó `0.001` y 11.3 usó `0.003`. Sin resolver (ver 11.5).
- **11.2** (9 jobs en serie, `scripts/ae/run_11_2_serie.ps1`, todos `Completed`): `gate_pass` base 1, noaux 0, waux0 1, waux05 0, waux1 0, d8 0, d32 0. `ref-mean` queda registrado dos veces (una por job de referencia).
- **Preparación de 11.2:** `AWS_PROFILE` de la shell apuntaba a una cuenta ajena; todos los comandos usaron `--profile desareca_dev`. El README raíz (líneas 66–78) muestra `--profile $env:AWS_PROFILE`: conviene corregirlo.
- **11.3** (`scripts/ae/run_11_3_finales.ps1`, `lr 0.003` por decisión del operador): 7 jobs `Completed`, ~4 h 40 min. Un primer intento falló por el warning de deprecación del SDK leído como error en PowerShell; se corrigió validando por exit code.

  | Finalista | L | Compuerta s0/s1/s2 | Razón valid | Razón test |
  |---|---|---|---|---|
  | conv1d | 24 | 1/1/1 | 0.904 | 0.916 |
  | gru | 24 | 1/1/0 | 0.953 | 0.976 |
  | conv1d | 72 | 0/0/0 | 0.854 | 0.945 |
  | mlp | 72 | 0/0/0 | 1.015 | 1.025 |
  | linear_decoder | 72 | 0/0/0 | 1.062 | 1.056 |

  Ningún finalista queda como "aprobada solo en valid" (ninguno con razón de test ≥ 1). conv1d L = 72 tiene brecha valid–test de 0.09 y `best_epoch` en 52–60. gru L = 24 cambia de veredicto entre semillas.
- **11.4:** cerrada con el pase encadenado de la semilla 0 (decisión del operador, D6). conv1d L = 24 (`8f348a…`) y gru L = 24 (`fcc939…`) tienen `dmd_operator.npz` y `dmd_operator_trunc.npz`. Linealizabilidad (1 paso / 24 pasos): conv1d 0.147 / 0.779 (truncado 0.480 / 0.813); gru 0.057 / 0.372 (truncado 0.192 / 0.386). Las semillas 1 y 2 no tienen DMD: repetirlo exige decidir el nombre de run (no lleva semilla) y probar el modo `dmd` con descarga desde `_mlflow/`, que aún no corrió en SageMaker.
- **11.5:** resumen en `EXPERIMENT-LOG.md`. La compuerta recalculada coincide con MLflow en 15/15 corridas. En test: conv1d L = 24 aprueba 3/3; gru L = 24 aprueba 1/3; conv1d L = 72 aprueba 1/3. Sin FLAG de aplanamiento en test para conv1d L = 24. Sesgo por tramo de `log1p`: mismo patrón que PCA(16), no separable del aplanamiento general. Pronóstico DMD vs LightGBM v1 sobre filas comunes: RMSE media 20.97 (conv1d) y 20.69 (gru) frente a 21.25; RMSE ponderado peor (33.28 y 31.09 frente a 28.17). Persistencia no es comparable fila a fila: no guardó `predictions.parquet`.
- **Pendientes para 12.1/12.2:** runs `lightgbm` `version = 99` a limpiar en `sinca-baseline`; discrepancia de `lr` (11.2 frente a 11.3).

### Pendiente para el Arquitecto: SDK SageMaker v2 → v3

- AWS avisó que la v2 está en mantenimiento desde el 6-jul-2026 y termina soporte el 6-jul-2027. La v3 rompe la API de `launch.py`; requiere change propio. Se mantiene el pin v2. Decidir si y cuándo.

### Ajustes 12.1 (dashboard)

- **Filtros AE** (7.2, `streamlit_app.py`): selectores Familia y Versión antes del selector de run. Verificado en navegador.
- **12.1 cerrada** por aceptación del operador. Sin captura adjunta; verificación: 12 runs `ae-*-dmd` con contrato completo, sin excepciones en ambas vistas.
- **Carga lenta:** `ensure_state` baja 370 objetos / 355 MiB en serie en cada arranque del contenedor. Pendiente decidir si se paraleliza o se persiste en volumen.

## Decisiones del Arquitecto

- **Punto de control 10.3 (2026-11-03):**
  1. Compuerta relativa con tolerancia: `beats_mean` estricto en todas las series; `beats_pca` en el agregado; tolerancia de 2 % por serie en la capa numérica (3.8).
  2. Umbrales: `max_rmse_vs_pca = 1.02` fijo; `min_var_ratio = 0.80` y `slope_tol = 0.20` provisionales hasta 10.5.
  3. `L = 168` fuera de la búsqueda; el módulo de datos lo sigue soportando.
  4. GRU solo `L = 24`, `hidden ≤ 64`, 4 trials.
  5. Todo en CPU `ml.m5.large` (desviación de `project-decisions.md` §3, registrada en §9).
  6. Presupuesto de búsqueda `{linear_decoder 12, mlp 12, conv1d 6, gru 4}`, 4 ventanas, 15 epochs; 4 jobs en paralelo (`search` no usa MLflow).
  7. `d_aux = max(2, d/4)` fijo; `d = 16` en la búsqueda; `d ∈ {8, 32}` solo en 11.2.
  - **Hallazgo:** la búsqueda puntuaba por RMSE absoluto, que no es comparable entre L ni d. El score pasa a ser la razón RMSE_AE/RMSE_PCA(d, L), con `d = 16` fijo en la búsqueda.
  - Antes de 10.4 y 11.1 se cierran 3.8, 5.8 y 5.9 con tests.
- **Punto de control corto 10.5 (2026-11-03):**
  1. Umbrales: `min_var_ratio` 0.80 → 0.75; `slope_tol` 0.20 → 0.30; `max_rmse_vs_pca` 1.02 sin cambio. Criterio: nivel de PCA(16) + margen ~0.05 (tarea 3.9).
  2. Epochs del final: 60 (la loss de train mejora 1.5–3.6 % entre epochs 40 y 60).
  3. Tolerancia de 2 % por serie se mantiene (desviación entre semillas de conv1d 1.64 %).
  4. 11.2 arranca con conv1d a `L = 24`. 11.3 suma conv1d `L = 24` como quinto finalista y reporta la razón en test antes de 11.4.
  - **Alerta:** ningún trial de búsqueda gana a PCA(16) en valid, pero la calibración da 0.895 en valid dentro del train. Por eso 11.3 reporta test antes de dar por buena la compuerta.
- **Decisiones aplicadas en el código (3.7 y 5.7):** `gate_pass` siempre con capa relativa; la capa numérica se suma solo si hay umbrales. DMD encadenado en la semilla 0 si pasa la compuerta; modo `dmd` como respaldo con `--ae-run-id`.
- **SDK v2 → v3:** pendiente de decisión del Arquitecto (ver grupo 11).
