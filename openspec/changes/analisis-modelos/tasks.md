## 0. Pre-work (operador)

- [ ] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [ ] 0.2 Confirmar que el change `baseline` está archivado y `openspec/specs/modelado/spec.md` existe (base del delta `modelado` de este change)

## 1. Persistencia de diagnósticos en el training

- [ ] 1.1 `pipeline/train/train.py`: al cierre de cada run LightGBM, construir y loguear con `mlflow.log_artifact` (dentro de `_mlflow/mlruns/`, no a `model-artifacts/`):
  - `predictions.parquet` — concat de `rolling_frame` + `test_frame` con columna `split` (`train`/`valid`/`test`), columnas `timestamp, estacion, pollutant, y_true, y_pred, split`
  - `feature_importance.parquet` — gain por feature y serie (`booster.feature_importance("gain")`)
  - `zone_icap.parquet` — `ICAP_zona` real vs pred del test held-out (`evaluate.zone_icap(test_frame)`), columnas `timestamp, icap_true, icap_pred`

## 2. Módulo de análisis (`pipeline/analysis/`)

- [ ] 2.1 Crear `pipeline/analysis/__init__.py` y `pipeline/analysis/load.py` — `download_state` a un dir local, listar runs/experimento desde `mlflow.sqlite`, ubicar y leer los 3 parquet por run; soportar `--data-dir` (estado local) vs S3 directo; degradar a "solo métricas" con aviso si falta `predictions.parquet`
- [ ] 2.2 Crear `pipeline/analysis/plots.py` — figuras Plotly puras (independientes de Streamlit): series real vs pred con bandas ICAP (alerta 100 / preemergencia 200 / emergencia 300), `ICAP_zona` real vs pred, feature importance horizontal, barra RMSE LightGBM vs persistencia
- [ ] 2.3 Crear `pipeline/analysis/streamlit_app.py` — widgets (experimento, run, serie) + pestañas + `@st.cache_data` para no re-descargar

## 3. Tooling local (contenedor) y fixture sintético

- [ ] 3.1 Crear `requirements-analysis.txt` (streamlit, plotly, pandas, mlflow, boto3, pyarrow)
- [ ] 3.2 Crear `docker/analysis.Dockerfile` — imagen con solo el runtime instalado desde `requirements-analysis.txt`, `ENTRYPOINT ["streamlit", "run"]`
- [ ] 3.3 Crear fixture sintético en `pipeline/analysis/fixtures/` (dataset señuelo con predicciones, feature importance e `ICAP_zona` mock) para iterar la estética sin AWS

## 4. Pruebas locales de gráficos (iteración, sin AWS)

- [ ] 4.1 Construir la imagen y correr el dashboard en contenedor contra el fixture (montando el código como volumen); afinar la definición de gráficos junto al operador hasta cerrarla

## 5. Documentación

- [ ] 5.1 Actualizar `README.md` — sección "Dashboard de análisis": correr local en contenedor (`docker build`/`docker run` + fixture) y en el Space (`pip install -r requirements-analysis.txt` + `streamlit run` + port-forward del 8501)
- [ ] 5.2 Actualizar `project-decisions.md` §8 — insertar la fase `analisis-modelos` después de `baseline` y renumerar las siguientes
- [ ] 5.4 Verificar que `notebooks/analisis_baseline.ipynb` (base, provisto por el arquitecto) abre en el Space, y documentar en `README.md` el flujo de notebooks: encender (Studio → Spaces → `sinca-dev` → Open), trabajar (rol del Space, sin credenciales), y apagar (Spaces → `sinca-dev` → Stop) para evitar sobrecostos

## 6. Ejecución real (operador corre y pega output)

- [ ] 6.1 Reset del tracking (borrado aprobado, "partir de 0"): `aws s3 rm s3://sinca-mlflow/_mlflow/ --recursive` y `aws s3 rm s3://sinca-mlflow/model-artifacts/ --recursive`; pegar output real
- [ ] 6.2 Re-correr el Training Job completo: `python pipeline/train/launch.py --role-arn <arn> --image-uri <uri>:latest --wait` → 7 runs limpios (`sinca-{estacion}-lightgbm-v1` × 6 + `sinca-persistence-baseline`)
- [ ] 6.3 Verificar en S3 que aparecen `predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet` bajo `s3://sinca-mlflow/_mlflow/mlruns/.../artifacts/`
- [ ] 6.4 Levantar el dashboard en el Space (`git pull` + `pip install -r requirements-analysis.txt` + `streamlit run pipeline/analysis/streamlit_app.py --server.port 8501`) y port-forward del 8501

## 7. Verificación funcional

- [ ] 7.1 Ver el dashboard con datos reales: series real vs pred, `ICAP_zona`, RMSE vs persistencia y feature importance, confirmando con output/captura real que coinciden con las métricas de MLflow
