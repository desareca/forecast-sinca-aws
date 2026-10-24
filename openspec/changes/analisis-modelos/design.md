## Context

El baseline (Fase 3) está entrenado y verificado en MLflow (`sinca-mlflow/_mlflow/`), pero sus resultados son legibles solo como métricas/parámetros vía la UI de MLflow o como reportes JSON. No existe ningún componente de visualización (ni notebooks, ni plots; `evaluate.py` solo emite dicts numéricos). Faltan dos cosas para "ver" el modelo: (1) que cada run **persista sus predicciones/importancia/ICAP**, y (2) un **dashboard interactivo** que las consuma.

Decisiones cerradas que condicionan el diseño: tracking MLflow self-hosted (SQLite+S3, logging directo modo 1), dev environment SageMaker Studio Space (`ml.t3.large` + EFS + SSH-over-SSM), todo on-demand/apagable, costo < USD 5/mes, y `sinca-dev-role` ya puede leer `sinca-mlflow/_mlflow/*` y `sinca-data/validated/*`.

## Goals / Non-Goals

**Goals:**
- Dashboard Streamlit interactivo que permita revisar el baseline de forma amigable (series, ICAP_zona, feature importance, RMSE vs persistencia).
- Persistir por run los diagnósticos necesarios para alimentar ese dashboard.
- Un módulo de análisis versionable (`pipeline/analysis/`) reutilizable en Fases 4-6.
- Iteración de gráficos en local (contenedor Docker + fixture sintético), sin AWS.
- Un notebook base para análisis puntuales ad-hoc en el Space.

**Non-Goals:**
- No comparación contra la referencia pública del MMA (eso es Fase 5 `evaluacion`).
- No features/modelos nuevos, ni reentrenamiento/inferencia automatizada (Fase 4/6).
- No un servicio 24/7 ni un deploy Fargate del dashboard (se corre on-demand).
- No tocar la lógica de entrenamiento (loss, split, hiperparámetros) — solo persistir artefactos al final de cada run.

## Decisions

### 1. Persistir diagnósticos en el run, no re-predecir en el dashboard

`train.py` ya calcula `rolling_frame`/`test_frame` (con `y_true`/`y_pred`) y los descarta. En vez de que el dashboard re-predecya desde el modelo guardado (acoplado y propenso a drift de features en F4), se **persiste** al cierre de cada run:
- `predictions.parquet` — `[timestamp, estacion, pollutant, y_true, y_pred, split]` (rolling + test).
- `feature_importance.parquet` — gain por feature y serie (vía `booster.feature_importance("gain")`).
- `zone_icap.parquet` — `ICAP_zona` real vs pred por timestamp (test held-out, vía `evaluate.zone_icap`).

Se loguean con `mlflow.log_artifact` (dentro de `_mlflow/mlruns/.../artifacts/`) — no a `model-artifacts/`, que `sinca-dev-role` no lee. Alternativa considerada (re-predecir) descartada por acoplamiento a la versión exacta de `build_features`.

### 2. Módulo `pipeline/analysis/` agnóstico del entorno

- `load.py`: reutiliza `train.mlflow_log.download_state` para bajar `mlflow.sqlite` + `mlruns/` a un dir local; lista runs/experimento desde el sqlite; ubica y lee los tres parquet por run. Acepta `--data-dir` (estado local, para iterar) o S3 directo (consulta real).
- `plots.py`: figuras Plotly puras (independientes de Streamlit, testables).
- `streamlit_app.py`: frontend delgado (widgets + `@st.cache_data`).

Por qué dos capas: permite iterar `plots.py` sin Streamlit y reutilizarlo en F4/F5 (comparativas) y en futuros reportes HTML.

### 3. Plotly como librería de gráficos

Elegida (aprobada) por zoom/hover nativos, bandas horizontales triviales (las 3 bandas ICAP como `hrect`) y salida HTML interactiva. Alternativa considerada: Altair (declarativo/liviano) — descartada porque las bandas/annotations de ICAP y el zoom son más directas en Plotly.

### 4. Ejecución: local en Docker (iterar), Space under port-forward (consulta)

- **Local**: `docker/analysis.Dockerfile` con solo el runtime (instalado desde `requirements-analysis.txt`), código montado como volumen → hot-reload sin rebuild. Evita instalar deps en la máquina local y reproduce el runtime.
- **Space**: el mismo `requirements-analysis.txt` se instala con `pip` (vía `~/.on_start`); se corre `streamlit run` y se accede por port-forward de VS Code (SSH-over-SSM). No se usa Docker en el Space (no hay daemon).

Un único `requirements-analysis.txt` (streamlit, plotly, pandas, mlflow, boto3, pyarrow) es la fuente de verdad de deps para ambos.

### 5. Fuente de datos local: fixture sintético; real: S3

La iteración de la estética usa un **fixture sintético** (`tests/fixtures/analysis/` o similar) montado junto al código, sin red ni credenciales. Para datos reales desde el contenedor local se monta `~/.aws` y se pasan `AWS_PROFILE`/`AWS_DEFAULT_REGION`.

### 6. Reset del tracking + re-run (state limpio)

Los 7 runs actuales del baseline no tienen predicciones persistidas y comparten `run_name` por estación (colisión pm10/pm25). Se **borra** `s3://sinca-mlflow/_mlflow/` (aprobado por el operador, "partir de 0") y se **re-corre** el Training Job completo (~38 min, ~USD 0.07) para regenerar los runs con diagnósticos. Alternativa considerada (mantener runs viejos + re-correr) descartada: duplica run_name y deja runs sin predicciones que el dashboard tendría que degradar.

### 7. Notebooks de análisis puntual (complemento al dashboard)

Además del dashboard (vista "fija" de resultados), se deja un notebook base (`notebooks/analisis_baseline.ipynb`) para que el operador desarrolle análisis puntuales/ad-hoc en JupyterLab del Space. Corre con el rol `sinca-dev-role` (sin credenciales) y lee las mismas fuentes que el dashboard (validated + MLflow). No es código de producción ni fuente de verdad: es scratch persistido en el EBS del Space. Para evitar sobrecostos, la guía de encendido/trabajo/apagado va en el README.

## Risks / Trade-offs

- **El dashboard depende de diagnósticos persistidos** → si un run no los tiene (p. ej. corridas viejas), degrada a "solo métricas" con aviso, no falla.
- **Re-run regenera runs con mismos nombres (`-v1`) pero IDs nuevos** → mitigado con el reset del tracking.
- **`download_state` baja todo `mlruns/`** → para el baseline es chico (despreciable); si crece en F4, evaluar bajada selectiva por run.
- **Divergencia Docker local vs venv Space** → mismo `requirements-analysis.txt`; riesgo bajo, mitigado revisando que la app corra igual en ambos.
- **Costo del Space encendido mientras se consulta** → ~USD 0.06/h; documentar "apagar al terminar" (patrón `costos`).

## Migration Plan

1. Agregar la persistencia de diagnósticos en `pipeline/train/train.py` (cierre de cada run).
2. Crear `pipeline/analysis/` (load, plots, streamlit_app) + `docker/analysis.Dockerfile` + `requirements-analysis.txt` + fixture.
3. Iterar los gráficos en local (contenedor + fixture); cerrar la definición.
4. Reset del tracking (`aws s3 rm`) y re-correr el Training Job.
5. Verificar predicciones/importancia/ICAP en S3 y levantar el dashboard en el Space (verificación funcional).
6. Rollback: el código es aditivo (`pipeline/analysis/`, nuevas columnas de artefacto); reversión = `git checkout` + no re-correr. El reset del tracking es irreversible por decisión aprobada.

## Open Questions

- **`zone_icap` del rolling window**: por ahora se persiste solo el `ICAP_zona` del test held-out (lo que decide la comparativa); el del rolling queda si se ve necesaria esa vista. Resolver antes de codificar si se quiere ambosa.
- **Evolución a Fargate**: si el dashboard se quisiera compartir/compartir con más usuarios, migrar a Fargate on-demand (patrón del servidor MLflow); fuera de scope ahora, reabrir en Fase 5-6.
