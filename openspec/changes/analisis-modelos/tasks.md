## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar que el change `baseline` está archivado y `openspec/specs/modelado/spec.md` existe (base del delta `modelado` de este change)

## 1. Persistencia de diagnósticos en el training

- [x] 1.1 `pipeline/train/train.py`: al cierre de cada run LightGBM, construir y loguear con `mlflow.log_artifact` (dentro de `_mlflow/mlruns/`, no a `model-artifacts/`):
  - `predictions.parquet` — concat de `rolling_frame` + `test_frame` con columna `split` (`train`/`valid`/`test`), columnas `timestamp, estacion, pollutant, y_true, y_pred, split`
  - `feature_importance.parquet` — gain por feature y serie (`booster.feature_importance("gain")`)
  - `zone_icap.parquet` — `ICAP_zona` real vs pred del test held-out (`evaluate.zone_icap(test_frame)`), columnas `timestamp, icap_true, icap_pred`

## 2. Módulo de análisis (`pipeline/analysis/`)

- [x] 2.1 Crear `pipeline/analysis/__init__.py` y `pipeline/analysis/load.py` — `download_state` a un dir local, listar runs/experimento desde `mlflow.sqlite`, ubicar y leer los 3 parquet por run; soportar `--data-dir` (estado local) vs S3 directo; degradar a "solo métricas" con aviso si falta `predictions.parquet`
- [x] 2.2 Crear `pipeline/analysis/plots.py` — figuras Plotly puras (independientes de Streamlit): series real vs pred con bandas ICAP (alerta 100 / preemergencia 200 / emergencia 300), `ICAP_zona` real vs pred, feature importance horizontal, barra RMSE LightGBM vs persistencia
- [x] 2.3 Crear `pipeline/analysis/streamlit_app.py` — widgets (experimento, run, serie) + pestañas + `@st.cache_data` para no re-descargar

## 3. Tooling local (contenedor) y fixture sintético

- [x] 3.1 Crear `requirements-analysis.txt` (streamlit, plotly, pandas, mlflow, boto3, pyarrow)
- [x] 3.2 Crear `docker/analysis.Dockerfile` — imagen con solo el runtime instalado desde `requirements-analysis.txt`, `ENTRYPOINT ["streamlit", "run"]`
- [x] 3.3 Crear fixture sintético en `pipeline/analysis/fixtures/` (dataset señuelo con predicciones, feature importance e `ICAP_zona` mock) para iterar la estética sin AWS

## 4. Pruebas locales de gráficos (iteración, sin AWS)

- [x] 4.1 Construir la imagen y correr el dashboard en contenedor contra el fixture (montando el código como volumen); afinar la definición de gráficos junto al operador hasta cerrarla

## 5. Documentación

- [x] 5.1 Actualizar `README.md` — sección "Dashboard de análisis": correr local en contenedor (`docker build`/`docker run` + fixture) y en el Space (`pip install -r requirements-analysis.txt` + `streamlit run` + port-forward del 8501)
- [x] 5.2 Actualizar `project-decisions.md` §8 — insertar la fase `analisis-modelos` después de `baseline` y renumerar las siguientes
- [ ] 5.4 Verificar que `notebooks/analisis_baseline.ipynb` (base, provisto por el arquitecto) abre en el Space, y documentar en `README.md` el flujo de notebooks: encender (Studio → Spaces → `sinca-dev` → Open), trabajar (rol del Space, sin credenciales), y apagar (Spaces → `sinca-dev` → Stop) para evitar sobrecostos

## 6. Ejecución real (operador corre y pega output)

- [ ] 6.1 Reset del tracking (borrado aprobado, "partir de 0"): `aws s3 rm s3://sinca-mlflow/_mlflow/ --recursive` y `aws s3 rm s3://sinca-mlflow/model-artifacts/ --recursive`; pegar output real
- [ ] 6.2 Re-correr el Training Job completo: `python pipeline/train/launch.py --role-arn <arn> --image-uri <uri>:latest --wait` → 7 runs limpios (`sinca-{estacion}-lightgbm-v1` × 6 + `sinca-persistence-baseline`)
- [ ] 6.3 Verificar en S3 que aparecen `predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet` bajo `s3://sinca-mlflow/_mlflow/mlruns/.../artifacts/`
- [ ] 6.4 Levantar el dashboard en el Space (`git pull` + `pip install -r requirements-analysis.txt` + `streamlit run pipeline/analysis/streamlit_app.py --server.port 8501`) y port-forward del 8501

## 7. Verificación funcional

- [ ] 7.1 Ver el dashboard con datos reales: series real vs pred, `ICAP_zona`, RMSE vs persistencia y feature importance, confirmando con output/captura real que coinciden con las métricas de MLflow

## Notas de implementación

- **1.1 (split de `predictions.parquet`)**: se persisten `rolling_frame` como
  `split="valid"` y `test_frame` como `split="test"`. **No hay filas `train`**:
  el design §1 pide "rolling + test" y no existe un frame de predicciones de
  train (solo se calcula el `train_loss`). Validado localmente con un run
  mínimo (`--no-upload`, Nielol/pm10, 1 trial): `predictions` 28910 filas,
  `feature_importance` 22 filas, `zone_icap` 1897 filas, esquemas correctos.
- **1.1 (`zone_icap.parquet` es por serie, no por zona)**: `evaluate.zone_icap`
  necesita las 6 series para tomar el máximo por timestamp; como se loguea al
  cierre de **cada** run, el `zone_icap.parquet` de un run es el ICAP de **esa**
  serie (una estación/contaminante). El `ICAP_zona` real se calcula en el
  dashboard concatenando las predicciones de test de los 6 runs
  (`load.zone_icap_from_runs`), que es lo que muestra la pestaña "ICAP zona".
  Queda a criterio del arquitecto si además quiere un `zone_icap.parquet` global
  en un run aparte.
- **2.2 (bandas ICAP en el gráfico de concentración)**: el ICAP es un índice, no
  una concentración. En el gráfico de serie las bandas se dibujan en el
  **equivalente en µg/m³** de ICAP 100/200/300 (`icap_to_concentration`:
  pm10 150/195/240; pm25 50/80/110), no literalmente en 100/200/300. En el
  gráfico de `ICAP_zona` sí son 100/200/300 (eje índice).
- **2.3 (`width="stretch"`)**: se usa `width="stretch"` en `plotly_chart`/
  `dataframe` (Streamlit >= 1.50) en vez de `use_container_width` (deprecado).
  `AppTest` (headless, dentro del contenedor) detectó un bug inicial en las
  tuplas de bandas; corregido. Estado final: 4 tabs, 0 excepciones, métricas OK.
- **3.3 (fixture = estado MLflow real)**: `make_fixture.py` genera
  `pipeline/analysis/fixtures/state/` con `mlflow.sqlite` + `mlruns/` reales
  (7 runs: 6 LightGBM + persistencia, cada uno con los 3 parquet). Así el
  dashboard corre el mismo camino que con S3 (`--data-dir`), sin AWS.
- **4.1 (estado)**: imagen `sinca-analysis` construida; contenedor sirve en
  `:8501` (`/_stcore/health` = `ok`) y `AppTest` pasa contra el fixture. Falta
  la revisión visual del operador para cerrar la definición de gráficos.
- **5.4 (estado)**: el flujo encender/trabajar/apagar del notebook quedó
  documentado en `README.md`; falta que el operador confirme que
  `notebooks/analisis_baseline.ipynb` abre en el Space.
- **4.1 (UX cerrada por ahora, iterativa)**: se afinó el dashboard con el
  operador. Cambios respecto del diseño inicial:
  - Flujo **serie primero** + selector único **"Modelos (comparar)"** por
    `(familia, versión)` (p. ej. `lightgbm v1`, `lightgbm v2`, `dmd v1`), que
    alimenta **series e ICAP**. La familia se lee de param/tag `modelo` (o se
    parsea del `run_name`); `train.py` ya setea el tag `modelo`.
  - **ICAP_zona** se agrupa por `(modelo, versión)` (`zone_icap_by_model_version`),
    nunca mezcla familias/versiones. Respeta el **split** (test/valid).
  - **RMSE vs persistencia**: dos gráficos **lado a lado** (test y valid) para la
    **serie seleccionada**, filtrados por el selector de modelos. El de valid
    muestra **media por ventana con barras de error min–max** y tooltip con
    `media/sd/min/max`; la tabla agrega `valid_sd`.
  - El **fixture** se amplió a 19 runs (lightgbm v1/v2 + dmd v1 × 6 series +
    persistencia) para poder comparar familias.
  - El operador aclara que la revisión de gráficos es **continua**; se cierra
    4.1 por ahora y se seguirá iterando.
- **4.1 (4º diagnóstico `rolling_scores.parquet`)**: para el rango min–max del
  CV hacía falta el RMSE por ventana (antes solo media/std). `train.py` ahora
  persiste `rolling_scores.parquet` (`window_index, rmse`) junto a los otros 3
  diagnósticos. Es un agregado respecto de los 3 artefactos que listaba 1.1;
  validado con un run local (se generan los 4 parquet).
- **Feature importance — PENDIENTE Fase 4 (stand by)**: el **gráfico** es
  agnóstico (lee `feature_importance.parquet` con esquema
  `estacion, pollutant, feature, gain`; si falta, el run se omite sin romper).
  El **cálculo** es específico del modelo: LightGBM usa `feature_importance("gain")`;
  para DMD/autoencoder (Fase 4) hay que definir el método (recomendado:
  permutation importance sobre validación, agnóstico) y persistir el mismo
  esquema. Evaluar además renombrar `gain` → `importance` para no atar la
  columna a terminología de árboles. **Decisión del arquitecto**; reabrir al
  encarar Fase 4.
