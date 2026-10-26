## Why

El baseline (Fase 3) quedó entrenado y verificado en MLflow, pero sus resultados solo se leen como tablas de métricas/parámetros en la UI de MLflow o como reportes JSON: no hay forma amigable de ver las series predichas vs observadas, el `ICAP_zona`, ni la importancia de features. Para decidir con calidad en la Fase 4 (autoencoder + DMD/Koopman) hace falta un dashboard interactivo de análisis de modelos, y para alimentarlo, que cada run persista sus diagnósticos.

## What Changes

- **Dashboard Streamlit interactivo** (on-demand): por run muestra resumen de RMSE LightGBM vs persistencia por serie/estación, series real vs pred con bandas ICAP, `ICAP_zona` real vs pred, y feature importance.
- **Módulo de análisis versionable** (`pipeline/analysis/`): lee el estado de MLflow (`sinca-mlflow/_mlflow/`) y produce figuras Plotly.
- **Persistencia de diagnósticos en cada run** (`pipeline/train/train.py`): loguea como artefactos MLflow `predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet`.
- **Tooling**: imagen Docker local (`docker/analysis.Dockerfile`) para iterar el dashboard en contenedor; `requirements-analysis.txt` compartido con el venv del Space; README con el flujo de levantar/apagar.
- **Reset del tracking MLflow** (borrado directo, aprobado) y **re-correr el baseline** para regenerar los runs con diagnósticos persistidos.
- **Notebook de análisis puntual** (`notebooks/analisis_baseline.ipynb`): punto de partida para análisis exploratorios en SageMaker Studio, con guía de encendido/trabajo/apagado para evitar sobrecostos.
- **Scripts de start/stop del Space** (`infra/scripts/start-space.ps1` / `stop-space.ps1`, no versionados): arrancan/apagan el app JupyterLab del Space de forma idempotente, evitando el quirk de la consola que a veces no recrea el app al "Open" (deja el Space `InService` con `list-apps` vacío).

## Capabilities

### New Capabilities

- `analisis-modelos`: dashboard Streamlit interactivo y módulo de análisis que leen el estado de MLflow y visualizan predicciones, `ICAP_zona` e importancia de features.

### Modified Capabilities

- `modelado`: se agrega la persistencia por run de diagnósticos (predicciones del held-out y de la ventana deslizante, feature importance, `ICAP_zona`) como artefactos de tracking MLflow.
- `dev-environment`: se agregan scripts locales de start/stop del Space (no versionados, con `.example` de referencia) para un ciclo de encendido/apagado confiable del app JupyterLab.

## Impact

- **Repo**: nuevo `pipeline/analysis/` (`load.py`, `plots.py`, `streamlit_app.py`), `docker/analysis.Dockerfile`, `requirements-analysis.txt`, `notebooks/analisis_baseline.ipynb`; modificación de `pipeline/train/train.py` (persistir diagnósticos). Nuevos `infra/scripts/start-space.ps1.example` y `stop-space.ps1.example` (referencia con placeholders; los `.ps1` reales son locales y gitignored).
- **Infraestructura AWS**: sin cambios (usa `sinca-dev-role` existente y el Space on-demand). Se re-corre el Training Job del baseline (~USD 0.07).
- **Datos**: se borra y regenera el estado de tracking en `s3://sinca-mlflow/_mlflow/` (reset explícito y aprobado por el operador).
- **No incluye**: reentrenamiento/inferencia automatizada (Fase 6), comparativa contra referencia pública (Fase 5 `evaluacion`), ni features/modelos nuevos (Fase 4).
