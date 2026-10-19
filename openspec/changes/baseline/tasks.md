## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar `sinca-data/validated/` poblado (backfill Fase 2) y MLflow self-hosted operativo

## 1. Feature engineering (`pipeline/features/`)

- [x] 1.1 Crear `pipeline/features/build.py` — leer `validated/`, construir features (lags 1h/24h/48h/72h/168h, meteorología con lag 24h, altura de capa límite, feriados, día de semana cyclic, ventana GEC, Fourier diaria/semanal/anual)
- [x] 1.2 Calcular target — promedio móvil 24h a t+24h por serie (`mean(MP(t+1 … t+24h))`)
- [x] 1.3 Implementar split — ventana deslizante (train 2 años / valid 1 mes / step 1 mes) + test final 3 meses held-out

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

## 8. Ejecución real (operador corre y pega output)

- [x] 8.1 Build de la imagen (sin push todavía): `docker build -f docker/train.Dockerfile -t sinca-train .`
- [x] 8.2 `terraform plan` y `terraform apply` (crea ECR + finaliza IAM)
- [x] 8.3 Push de la imagen a ECR (el repo recién existe tras el apply)
- [ ] 8.4 Correr Training Job manual (baseline completo: persistencia + LightGBM)
- [ ] 8.5 Verificar métricas en MLflow (UI o listar `sinca-mlflow/_mlflow/mlruns/`)

## 9. Verificación funcional

- [ ] 9.1 Smoke test: baseline de punta a punta, confirmar métricas (persistencia + LightGBM) en MLflow con output real

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

