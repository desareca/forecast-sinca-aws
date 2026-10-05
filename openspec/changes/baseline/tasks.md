## 0. Pre-work (operador)

- [ ] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [ ] 0.2 Confirmar `sinca-data/validated/` poblado (backfill Fase 2) y MLflow self-hosted operativo

## 1. Feature engineering (`pipeline/features/`)

- [ ] 1.1 Crear `pipeline/features/build.py` — leer `validated/`, construir features (lags 1h/24h/48h/72h/168h, meteorología con lag 24h, altura de capa límite, feriados, día de semana cyclic, ventana GEC, Fourier diaria/semanal/anual)
- [ ] 1.2 Calcular target — promedio móvil 24h a t+24h por serie (`mean(MP(t+1 … t+24h))`)
- [ ] 1.3 Implementar split — ventana deslizante (train 2 años / valid 1 mes / step 1 mes) + test final 3 meses held-out

## 2. Persistencia baseline

- [ ] 2.1 Crear `pipeline/baseline/persistence.py` — naive (predicción = último promedio móvil 24h observado)

## 3. Entrenamiento LightGBM

- [ ] 3.1 Crear `pipeline/train/train.py` — 6 modelos independientes, RMSE ponderado (`α=1`, `umbral_regular` = ICAP 100: MP10 150 / MP2.5 50)
- [ ] 3.2 Implementar búsqueda de hiperparámetros por serie (random search + early stopping, ~20 trials, espacio del design §9)
- [ ] 3.3 Crear `pipeline/train/mlflow_log.py` — logging directo (modo 1), convención `sinca-{estacion}-{modelo}-v{n}`, subida de `mlflow.sqlite` + `mlruns/` a `sinca-mlflow/_mlflow/`

## 4. Evaluación

- [ ] 4.1 Crear `pipeline/evaluate/evaluate.py` — rolling window + métricas (`train_loss`, `mse`/`rmse` por serie, bloque ICAP oficial, bloque percentil propio)

## 5. Imagen de entrenamiento

- [ ] 5.1 Crear `docker/train.Dockerfile` — runtime LightGBM (`lightgbm`, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `boto3`, `pyarrow`)

## 6. Terraform

- [ ] 6.1 Crear `infra/ecr-train.tf` — repo ECR `sinca-train`
- [ ] 6.2 Actualizar `infra/iam-training.tf` — finalizar `sinca-training-role` (read `sinca-data/validated/*`, r/w `sinca-mlflow/_mlflow/*` + `sinca-mlflow/model-artifacts/*`, logs `/aws/sagemaker/*`)

## 7. Documentación

- [ ] 7.1 Actualizar `README.md` (estado Fase 3; cómo correr el baseline)
- [ ] 7.2 Actualizar `project-decisions.md` §3 (cómputo training confirmado) y §8 (baseline activo)
- [ ] 7.3 Crear `EXPERIMENT-LOG.md` (convención prompt §6)

## 8. Ejecución real (operador corre y pega output)

- [ ] 8.1 Build de la imagen (sin push todavía): `docker build -f docker/train.Dockerfile -t sinca-train .`
- [ ] 8.2 `terraform plan` y `terraform apply` (crea ECR + finaliza IAM)
- [ ] 8.3 Push de la imagen a ECR (el repo recién existe tras el apply)
- [ ] 8.4 Correr Training Job manual (baseline completo: persistencia + LightGBM)
- [ ] 8.5 Verificar métricas en MLflow (UI o listar `sinca-mlflow/_mlflow/mlruns/`)

## 9. Verificación funcional

- [ ] 9.1 Smoke test: baseline de punta a punta, confirmar métricas (persistencia + LightGBM) en MLflow con output real
