# Imágenes Docker de entrenamiento

Esta carpeta tiene los Dockerfiles de las imágenes de entrenamiento. Todas las
imágenes son **solo runtime**: el código viaja por `source_dir`/`entry_point` desde
`pipeline/` en cada Training Job.

| Dockerfile | Repositorio ECR | Tag explícito | Uso |
|---|---|---|---|
| `train.Dockerfile` | `sinca-train` | `lgbm-v1` | Baseline LightGBM (Fase 3). |
| `train-tf.Dockerfile` | `sinca-train` | `tf-cpu-v1` | Autoencoders TensorFlow (`ae-reconstruccion`). |
| `scraper.Dockerfile` | (pipeline de datos) | — | Scraper de la Fase 2; instala sus deps inline. |
| `analysis.Dockerfile` | (dashboard local) | — | Dashboard Streamlit, `requirements-analysis.txt`. |

## Reglas

- **Nunca `:latest`.** Cada imagen de training lleva un tag explícito y `pipeline/train/launch.py`
  rechaza imágenes sin tag o con `:latest`.
- **El repositorio `sinca-train` es `MUTABLE`** (`infra/ecr-train.tf`): un tag ya publicado
  se puede sobrescribir. Por eso, al cambiar dependencias de una imagen, se publica un tag
  nuevo (`tf-cpu-v2`, `lgbm-v2`, …) y no se reusa uno existente.
- **Versiones fijadas:** `tensorflow-cpu==2.20.0` en `train-tf.Dockerfile` debe coincidir con
  `requirements-notebooks-tf.txt` (los notebooks de nivel 2 cargan los `.keras` de esa imagen).

## Build local

Desde la raíz del repo:

```powershell
docker build -f docker/train.Dockerfile -t sinca-train:lgbm-v1 .
docker build -f docker/train-tf.Dockerfile -t sinca-train:tf-cpu-v1 .
```

## Push a ECR

Los comandos usan el profile AWS explícito y obtienen la cuenta desde la identidad activa
(no hay IDs de cuenta escritos en el repo):

```powershell
$env:AWS_PROFILE = "<profile>"
$acct = aws sts get-caller-identity --query Account --output text
$reg  = "$acct.dkr.ecr.us-east-1.amazonaws.com"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $reg

docker tag sinca-train:tf-cpu-v1 "$reg/sinca-train:tf-cpu-v1"
docker push "$reg/sinca-train:tf-cpu-v1"
```

Verificar el digest y los tags publicados (salida real, no asumir):

```powershell
aws ecr describe-images --repository-name sinca-train --query "imageDetails[].{tags:imageTags,digest:imageDigest}"
```

## Retag de la imagen LightGBM existente (tarea 9.2, operador)

La imagen LightGBM que ya existe en ECR se publica con el tag explícito `lgbm-v1`
**sin reconstruirla**. Se copia el manifiesto del tag actual en lugar de descargar la imagen:

```powershell
$manifest = aws ecr batch-get-image --repository-name sinca-train --image-ids imageTag=latest `
    --query "images[0].imageManifest" --output text
aws ecr put-image --repository-name sinca-train --image-tag lgbm-v1 --image-manifest $manifest
```

Después de correrlo, confirmar con `describe-images` que `lgbm-v1` apunta al mismo digest que
la imagen original, y luego correr una corrida corta de LightGBM para validarla (tarea 9.2).

## Lanzar un Training Job con la imagen elegida

`pipeline/train/launch.py` toma el tag como argumento explícito:

```powershell
# Baseline LightGBM
python pipeline/train/launch.py --profile $env:AWS_PROFILE --role-arn <training_role_arn> `
    --image-uri "$reg/sinca-train:lgbm-v1" --dry-run

# Autoencoders (entry point ae/run.py)
python pipeline/train/launch.py --profile $env:AWS_PROFILE --role-arn <training_role_arn> `
    --image-uri "$reg/sinca-train:tf-cpu-v1" --entry-point ae/run.py `
    --hp mode=final --hp arch=gru --hp d=16 --hp length=72 --dry-run
```

Quitar `--dry-run` para lanzar el job real (siempre con el operador corriendo el comando y
revisando el output).
