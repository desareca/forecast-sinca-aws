## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar `sinca-data/validated/` poblado (backfill Fase 2) y MLflow self-hosted operativo

## 1. Feature engineering (`pipeline/features/`)

- [x] 1.1 Crear `pipeline/features/build.py` — leer `validated/`, construir features (lags 1h/24h/48h/72h/168h, meteorología con lag 24h, altura de capa límite, feriados, día de semana cyclic, ventana GEC, Fourier diaria/semanal/anual)
- [x] 1.2 Calcular target — promedio móvil 24h a t+24h por serie (`mean(MP(t+1 … t+24h))`)
- [x] 1.3 Implementar split — ventana deslizante (train 2 años / valid 1 mes / step 1 mes) + test final 3 meses held-out
- [x] 1.4 Ajustar `MET_COLUMNS` y `_station_base` en `pipeline/features/build.py` — decisión del arquitecto (post-mortem 1er run): quitar `wdir` (ninguna estación la publica) y rellenar con `NaN` las columnas met ausentes por estación (`glob` en Las Encinas) antes del lag, para no perder radiación en Nielol/PLC y mantener `FEATURE_COLUMNS` estable.

## 2. Persistencia baseline

- [x] 2.1 Crear `pipeline/baseline/persistence.py` — naive (predicción = último promedio móvil 24h observado)

## 3. Entrenamiento LightGBM

- [x] 3.1 Crear `pipeline/train/train.py` — 6 modelos independientes, RMSE ponderado (`α=1`, `umbral_regular` = ICAP 100: MP10 150 / MP2.5 50)
- [x] 3.2 Implementar búsqueda de hiperparámetros por serie (random search + early stopping, ~20 trials, espacio del design §9)
- [x] 3.3 Crear `pipeline/train/mlflow_log.py` — logging directo (modo 1), convención `sinca-{estacion}-{modelo}-v{n}`, subida de `mlflow.sqlite` + `mlruns/` a `sinca-mlflow/_mlflow/`

## 4. Evaluación

- [x] 4.1 Crear `pipeline/evaluate/evaluate.py` — rolling window + métricas (`train_loss`, `mse`/`rmse` por serie, bloque ICAP oficial, bloque percentil propio)

## 5. Imagen de entrenamiento

- [x] 5.1 Crear `docker/train.Dockerfile` — runtime LightGBM (`lightgbm`, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `boto3`, `pyarrow`)

## 6. Terraform

- [x] 6.1 Crear `infra/ecr-train.tf` — repo ECR `sinca-train`
- [x] 6.2 Actualizar `infra/iam-training.tf` — finalizar `sinca-training-role` (read `sinca-data/validated/*`, r/w `sinca-mlflow/_mlflow/*` + `sinca-mlflow/model-artifacts/*`, logs `/aws/sagemaker/*`)

## 7. Documentación

- [x] 7.1 Actualizar `README.md` (estado Fase 3; cómo correr el baseline)
- [x] 7.2 Actualizar `project-decisions.md` §3 (cómputo training confirmado) y §8 (baseline activo)
- [x] 7.3 Crear `EXPERIMENT-LOG.md` (convención prompt §6)

## 7b. Ajustes de tooling (post-mortem del 1er run)

- [x] 7b.1 Crear `requirements-dev.txt` (raíz) con `sagemaker>=2,<3` y `boto3>=1.34,<2` para el ambiente de lanzamiento local (el `.venv` ya cumple con `sagemaker==2.257.7`); vive **fuera de `pipeline/`** para que el toolkit `sagemaker-training` no lo instale en el contenedor. Documentar en el docstring de `pipeline/train/launch.py` que requiere SDK v2.
- [x] 7b.2 `pipeline/train/launch.py`: reconfigurar stdout a UTF-8 al arrancar (`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`) para que el streaming de logs no reviente con `UnicodeEncodeError` en Windows cp1252.

## 8. Ejecución real (operador corre y pega output)

- [x] 8.1 Build de la imagen (sin push todavía): `docker build -f docker/train.Dockerfile -t sinca-train .`
- [x] 8.2 `terraform plan` y `terraform apply` (crea ECR + finaliza IAM)
- [x] 8.3 Push de la imagen a ECR (el repo recién existe tras el apply)
- [x] 8.4 Correr Training Job manual (baseline completo: persistencia + LightGBM)
- [x] 8.5 Verificar métricas en MLflow (UI o listar `sinca-mlflow/_mlflow/mlruns/`)

## 9. Verificación funcional

- [x] 9.1 Smoke test: baseline de punta a punta, confirmar métricas (persistencia + LightGBM) en MLflow con output real

## Notas de implementación

- **3.1/3.2 (random search sobre una muestra de ventanas)**: el design §9 pide
  seleccionar por RMSE ponderado "promedio sobre ventanas de validación". Para
  acotar el costo del job (est. < USD 0.10) el search se evalúa sobre una
  muestra equiespaciada de ventanas (`--search-windows`, default 12 de 42), no
  sobre todas; la **evaluación** de la ventana deslizante completa sí se corre
  sobre las 42. Si el operador quiere el search completo: `--search-windows 0`.
- **3.3 (artefactos con URI local)**: se implementa el logging directo literal
  del spec/design §6: SQLite local + `mlruns/` local, y al terminar se suben
  `mlflow.sqlite` + `mlruns/` a `s3://sinca-mlflow/_mlflow/`. Los `artifact_uri`
  de los runs nuevos quedan como `file://` locales, así que en la UI del
  servidor (que usa `--artifacts-destination` S3) se ven métricas y parámetros
  pero el artefacto por run puede no resolverse. Alternativa no usada:
  `artifact_location` directo a S3 (consistente con el servidor, pero no
  coincide con "sube `mlruns/`" del spec).
- **4.1 (bloque percentil a nivel horario)**: `recall_top10pct`/`recall_top5pct`/
  `pr_auc_top10pct` se calculan sobre `ICAP_zona` **por timestep horario**, no
  agregado por "día" (el prompt §5 dice "top 5/10% de días"). A nivel de ventana
  (1 mes) el agregado diario deja ~3 días críticos y queda inestable; el nivel
  horario es más robusto y es el mismo dato que consume el ICAP oficial.
- **5.1 (deps extra en la imagen)**: además de las librerías del design §7 se
  agrega `sagemaker-training` (el script mode del SDK de SageMaker exige el
  entrypoint `train` en la imagen; la imagen sigue siendo solo runtime) y
  `libgomp1` por apt (LightGBM/scikit-learn enlazan OpenMP y no viene en
  `python:3.11-slim`). Además se instala `gcc` + `libc6-dev`: verificado contra
  PyPI que **ninguna** versión de `sagemaker-training` publica wheel (todas son
  sdist), y compila una extensión C mínima (`gethostname`) al instalarse; sin
  gcc el build falla con `command 'gcc' failed: No such file or directory`.
- **6.2 (permisos ECR en el rol de training)**: SageMaker exige que el rol de
  ejecución pueda bajar la imagen custom de ECR, así que el rol ahora incluye
  `ecr:BatchCheckLayerAvailability`/`BatchGetImage`/`GetDownloadUrlForLayer`
  acotados al repo `sinca-train` y `ecr:GetAuthorizationToken` (`*`, obligatorio
  por API). No estaba en la lista de recursos del spec de IAM ("sin más
  permisos"): es un requisito de la plataforma, no un permiso de negocio. El S3
  queda exactamente como pide el spec (validated read, `_mlflow/*` r/w,
  `model-artifacts/*` r/w, logs).
- **Archivos no listados en `tasks.md` (necesarios para 8.3/8.4)**: se creó
  `pipeline/train/launch.py` (lanzador del Training Job vía SDK de SageMaker) y
  se agregaron los outputs `train_ecr_repository_url`/`train_ecr_repository_arn`
  en `infra/outputs.tf`. Sin ellos no hay forma de disparar el job desde 8.4.
- **3.1 (código vía `source_dir` + `code_location`)**: para no pedirle al rol
  read sobre el bucket por defecto de SageMaker (`sagemaker-<region>-<account>`),
  el launcher fija `code_location=s3://sinca-mlflow/_mlflow/code` (prefijo que el
  rol ya puede leer) y `output_path=s3://sinca-mlflow/model-artifacts`.
- **8.4 (BLOQUEO: cuota de SageMaker Training = 0 en la cuenta)**: el Training
  Job no se puede crear: `ResourceLimitExceeded: The account-level service limit
  'ml.m5.large for training job usage' is 0 Instances`. Verificado contra
  Service Quotas: **0 de las 216** cuotas `... for training job usage` están por
  encima de 0 (la cuota agregada `Number of instances across all training jobs`
  sí es 4, pero no alcanza). El código y la infra quedaron OK (imagen en ECR,
  rol finalizado, `launch.py --dry-run` arma el Estimator sin errores); falta
  que AWS apruebe la cuota `L-611FA074` (`ml.m5.large for training job usage`).
  **Decisión pendiente del arquitecto/operador**: pedir el aumento de cuota (no
  tiene costo) o cambiar el cómputo del training (cualquier alternativa hoy
  también está en 0).**
- **8.4 (desbloqueo en curso — cuota solicitada)**: el operador optó por pedir
  el aumento de cuota. Solicitado vía `aws service-quotas
  request-service-quota-increase` (perfil `desareca_dev`, `us-east-1`):
  `L-611FA074` (`ml.m5.large for training job usage`), desired value `1`, nivel
  ACCOUNT. Request id `706043b57a55493a833c634c9a152f9eIbHaA0ds`. El pedido pasó
  de `PENDING` a `CASE_OPENED` (AWS abrió el caso de soporte `179276467000517`
  el 2026-10-23 11:11) → **requiere revisión manual, ya no se auto-aprueba**
  (esperable en cuenta nueva; 1–2 días hábiles). `get-service-quota` sigue
  devolviendo `Value: 0.0`, así que 8.4 sigue bloqueado hasta que el estado pase
  a `APPROVED`.
  Seguimiento:
  `aws service-quotas get-requested-service-quota-change --request-id 706043b57a55493a833c634c9a152f9eIbHaA0ds --region us-east-1 --profile desareca_dev`
  (o `... list-requested-service-quota-change-history-by-quota --service-code sagemaker --quota-code L-611FA074 ...`).
- **8.4 (desbloqueado — cuota concedida)**: el 2026-10-24 la cuota efectiva ya
  figura en `Value: 20.0` (`get-service-quota`), aunque el objeto del request
  siguió mostrando `CASE_OPENED`. AWS concedió 20 en vez del 1 pedido. Con eso se
  lanzó el Training Job `sinca-baseline-train-2026-10-24-18-04-19-349`
  (`ml.m5.large`, `InProgress` → `Training`), sin `ResourceLimitExceeded`.
- **8.4 (SDK de SageMaker: v3 rompe `launch.py`)**: el 2026-10-21 salió la línea
  **SageMaker Python SDK v3** (3.27.0), que **elimina** `Estimator`, `Model` y
  `Predictor` (`pip install sagemaker` sin pin instala v3). `pipeline/train/launch.py`
  usa la API v2 (`sagemaker.estimator.Estimator`, `sagemaker.session.Session`), así
  que **no se debe actualizar el SDK**. El `.venv` del repo ya tiene
  `sagemaker==2.257.7` (v2) + `sagemaker_core 1.0.78` + `boto3 1.43.108`, que es lo
  correcto. La **imagen Docker no se ve afectada**: instala `sagemaker-training`
  (toolkit de script-mode del contenedor), paquete distinto del SDK.
  **Riesgo latente para el arquitecto**: no hay pin de versión del SDK en el repo
  (no existe `requirements.txt` de dev) — conviene agregar `sagemaker>=2,<3` y, si
  se quiere migrar a v3, es un cambio de código (`ModelTrainer` + `SourceCode`).
- **8.4 (crash del launcher por encoding en Windows, no del job)**: con `--wait`,
  `estimator.fit` streamea logs de CloudWatch y el `print` de `sagemaker.logs`
  revienta con `UnicodeEncodeError: 'charmap' codec` (stdout cp1252 de Windows).
  El proceso cliente muere **después** de crear el job, que sigue corriendo en
  SageMaker. Workaround para futuras corridas: setear `PYTHONIOENCODING=utf-8` o
  `PYTHONUTF8=1` antes de invocar `launch.py` (o usar `--no-logs`). No se cambió
  el código todavía: queda a criterio del arquitecto si se fija en `launch.py`.
- **8.4 (`requirements.txt` del `pipeline/` se instala en el contenedor)**: al usar
  `source_dir=str(PIPELINE_DIR)`, el toolkit `sagemaker-training` detecta el
  `pipeline/requirements.txt` y lo instala en runtime (baja `pandera`,
  `beautifulsoup4`, etc. — deps del scraper, ajenas al training). No rompe, pero
  agrega tiempo de arranque y peso. Si molesta, el arquitecto puede decidir mover
  el `requirements.txt` del scraper fuera del `source_dir` del training.
- **8.4 (1er run FALLÓ — esquema meteorológico no uniforme entre estaciones)**:
  job `sinca-baseline-train-2026-10-24-18-04-19-349` → `Failed` (`Billable 79s`).
  Traceback real: `KeyError: "['wdir'] not in index"` en
  `features/build.py:191` (`_station_base`), que hace
  `station_df[["timestamp", *MET_COLUMNS]]`. Causa raíz: `MET_COLUMNS` es una
  constante global (`temp, rhum, pres, wdir, wspd, glob`) pero **las variables
  meteorológicas disponibles difieren por estación** en `validated/sinca`:
  - `padre_las_casas_ii`: temp, rhum, pres, rain, wspd, **glob** (sin wdir)
  - `nielol`: temp, rhum, pres, rain, wspd, **glob** (sin wdir)
  - `las_encinas`: temp, rhum, pres, rain, wspd (**sin glob, sin wdir**)
  **Ninguna estación publica `wdir`** (el scraper solo descubre lo que expone la
  página SINCA; el pandera es `strict=False`/`required=False`, así que no descarta
  nada). El design §2 pide "viento (dir/vel) y radiación", pero la fuente no
  entrega dirección de viento.
  **Opciones (decisión del arquitecto/operador)**:
  - **A (recomendada)**: por estación, crear como `NaN` las `MET_COLUMNS` que
    falten antes del lag (LightGBM tolera NaN) → mantiene el set de features del
    design y deja `FEATURE_COLUMNS` estable. Es un cambio chico en `build.py`.
  - **B**: recortar `MET_COLUMNS` a la intersección (`temp, rhum, pres, wspd`) →
    pierde radiación donde sí existe (Nielol/PLC) y dirección de viento.
  - **C**: traer viento/radiación desde Open-Meteo → cambio en el pipeline de
    datos (Fase 2) + backfill; mayor alcance.
  Bloquea 8.4/8.5/9.1 hasta resolver.
- **1.4 (implementado y validado local contra S3)**: `MET_COLUMNS` quedó en
  `("temp", "rhum", "pres", "wspd", "glob")` y `_station_base` crea como `NaN`
  las met vars ausentes por estación antes del lag. Prueba local de
  `build_features` sobre `s3://sinca-data/validated/`: `(257942, 27)`, 6 series,
  `met_wdir_lag24` ausente, `FEATURE_COLUMNS` completo, `target` 100% no-nulo.
  **Observación de calidad de dato**: `met_glob_lag24` es muy dispersa
  (NaN: Las Encinas 100%, PLC 99.8%, Nielol 74.4%) — la radiación casi no está
  en la fuente; no bloquea, pero probablemente aporte poco al modelo (anotado
  como hipótesis para Fase 4).
- **7b.1/7b.2 (implementado)**: `requirements-dev.txt` en la raíz con
  `sagemaker>=2,<3` + `boto3>=1.34,<2` (fuera de `pipeline/` para que no lo
  instale el toolkit), docstring de `launch.py` actualizado, y `main()`
  reconfigura `sys.stdout` a UTF-8 (`errors="replace"`).
- **8.4 (RESOLUCIÓN — decisión del arquitecto)**: opción **A, sin `wdir`**. `wdir`
  se quita de `MET_COLUMNS` (ninguna estación la publica: dejarla como columna
  `NaN` perpetua sería una feature fantasma sin señal y ensuciaría el feature
  importance). Las demás columnas met ausentes por estación (`glob` en Las
  Encinas) se rellenan con `NaN` antes del lag, preservando radiación donde sí
  existe (Nielol/PLC). `FEATURE_COLUMNS` queda estable salvo que `met_wdir_lag24`
  desaparece (consistente entre `build.py` y `train.py`). `rain` **NO** se agrega
  como feature (disponible en las 3 estaciones pero fuera del scope del design
  §2; queda como hipótesis para Fase 4). Tooling: pin `sagemaker>=2,<3` vía
  `requirements-dev.txt` (tareas 7b.1/7b.2); el `requirements.txt` de
  `pipeline/` se deja donde está (no rompe; reevaluar en Fase 4 si pesa el
  arranque).
- **8.4/8.5/9.1 (rerun OK — baseline completo)**:
  - Job `sinca-baseline-train-2026-10-24-18-33-19-397` → **Completed**.
    `Billable 2292 s` (~38 min, ~USD 0.07 en `ml.m5.large`). 6 series LightGBM +
    persistencia, sin errores. El fix de 1.4 y el de encoding de 7b.2 funcionaron.
  - MLflow: estado en `s3://sinca-mlflow/_mlflow/` (17 objetos; `mlflow.sqlite`
    + `mlruns/`). `mlflow.sqlite`: `integrity_check=ok`, **7 runs FINISHED** en
    el experimento `sinca-baseline` (6 LightGBM + 1 persistencia), **234**
    `latest_metrics`, params completos. Nota: los dos runs de una misma estación
    (pm10/pm25) comparten nombre `sinca-{estacion}-lightgbm-v1` por la convención
    de 3.3; se distinguen por el param `pollutant`.
  - **Resultado test (3 meses held-out, RMSE)** LightGBM vs persistencia:
    | serie | LightGBM | persistencia |
    |---|---|---|
    | Las Encinas pm10 | 19.665 | 24.031 |
    | Las Encinas pm25 | 19.453 | 22.082 |
    | Nielol pm10 | 17.665 | 20.307 |
    | Nielol pm25 | 16.596 | 18.523 |
    | PLC pm10 | 28.751 | 26.056 |
    | PLC pm25 | 27.748 | 23.205 |
    LightGBM mejora en 4/6 series (Las Encinas y Nielol) pero **empeora en PLC
    (pm10 y pm25)**. No es un error de pipeline: es un resultado real del baseline.
    Hipótesis para el arquitecto/Fase 4: PLC es la serie de mayor variabilidad y
    `glob` está casi toda `NaN` ahí (99.8%), así que el modelo puede estar
    sobreajustando o faltándole señal; evaluar features adicionales
    (`rain`, viento de Open-Meteo) o un baseline por estación.
  - `test_weighted_rmse` difiere de `test_rmse` en las series pm25 (p. ej. Nielol
    pm25: 42.37 vs 16.60) por el peso ICAP (`α=1`, `umbral_regular` 50/150):
    es esperado, no un bug.
  - Incidente menor de verificación (no del job): al consultar el `mlflow.sqlite`
    local, reusar el mismo cursor de `sqlite3` dentro de un loop cortaba la
    iteración y mostraba 1 run en vez de 7. El objeto en S3 está correcto
    (MD5 = ETag, `integrity_check=ok`).

