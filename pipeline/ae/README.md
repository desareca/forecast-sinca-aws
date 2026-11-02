# Autoencoders y pase de DMD (`pipeline/ae/`)

Módulo del change `ae-reconstruccion`: autoencoders con rama auxiliar y latente en bloques,
compuerta de reconstrucción, y pase de DMD global sobre el latente. Entry point único:
`ae/run.py`, con modos `search`, `final`, `reference` y `dmd`.

| Archivo | Contenido |
|---|---|
| `data.py` | Grilla horaria, huecos ≤ 6 h, ventanas, particiones, normalización (solo train). |
| `models.py` | Encoders/decoders Keras (`mlp`, `conv1d`, `gru`, `lstm`, `linear_decoder`) y pérdida enmascarada. |
| `metrics.py` | Métricas de reconstrucción, referencias (promedio, PCA(d)), latente y compuerta en dos capas. |
| `dmd.py` | DMD exacto por mínimos cuadrados (numpy), linealizabilidad y pronóstico informativo. |
| `run.py` | Entry point: búsqueda, entrenamiento final, referencias y DMD, con logging a MLflow. |

Los módulos `data`, `metrics` y `dmd` son numpy puro; TensorFlow solo se importa en `models`
y `run`.

## Tests (sin AWS)

Desde la raíz del repo, con el venv activo:

```powershell
python -m pytest tests/ae -q
```

Requiere `tensorflow-cpu==2.20.0` para los tests de modelos y de punta a punta. Los tests
usan datos sintéticos y un MLflow SQLite temporal.

## Correr un experimento local (sin AWS)

El entry point lee datos validados en disco con el mismo layout que `validated/` en S3
(`sinca/`, `open_meteo/`, `feriados/`) y escribe MLflow en un directorio local:

```powershell
python pipeline/ae/run.py --mode final --arch gru --d 16 --length 72 --w-aux 0.1 `
    --local-validated-dir ./validated --mlflow-dir ./mlflow-local `
    --model-dir ./model --output-dir ./output --no-upload
```

- `--no-upload` evita leer y escribir S3. Con él, `--mlflow-dir` debe contener el estado previo
  si se usa `--mode dmd --ae-run-id` (el run de AE se busca ahí).
- Por defecto el experimento es `sinca-baseline` (`--experiment` para cambiarlo).
- Para ajustar pocas épocas en una prueba: `--epochs 2 --patience 1`.

### En Docker, con la misma imagen del Training Job

La imagen `sinca-train:tf-cpu-v1` (ver `docker/README.md`) tiene el runtime. Como su
`ENTRYPOINT` es el de `sagemaker-training`, para correr local se sobreescribe con `python`
y se monta el repo y los datos:

```powershell
docker run --rm --entrypoint python `
    -v "${PWD}:/work" -w /work `
    sinca-train:tf-cpu-v1 `
    pipeline/ae/run.py --mode final --arch gru --d 16 --length 72 `
    --local-validated-dir /work/validated --mlflow-dir /work/mlflow-local `
    --model-dir /work/model --output-dir /work/output --no-upload
```

## Training Job (SageMaker)

El lanzador (`pipeline/train/launch.py`) empaqueta `pipeline/` como `source_dir` y usa el
entry point `ae/run.py`. Los hiperparámetros se pasan con `--hp CLAVE=VALOR`:

```powershell
python pipeline/train/launch.py --profile $env:AWS_PROFILE --role-arn <training_role_arn> `
    --image-uri <repo>/sinca-train:tf-cpu-v1 --entry-point ae/run.py `
    --hp mode=final --hp arch=gru --hp d=16 --hp length=72 --hp w_aux=0.1 --wait
```

Orden típico del trabajo (cada paso con el operador corriendo el comando):

1. `--mode search` por arquitectura, con `--search-trials` y `--search-windows` acotados (sale
   un JSON con la mejor configuración de cada arquitectura).
2. `--mode final` con `--seeds 0,1,2` para la configuración elegida. Con la compuerta
   aprobada, el pase de DMD de la semilla 0 corre en el mismo job (`--dmd-after-final`, default
   `true`).
3. `--mode dmd --ae-run-id <id>` solo si hace falta repetir el pase de DMD sin reentrenar. Lee
   `encoder.keras` y `decoder_mp.keras` del estado de MLflow y se niega si la compuerta no aprobó.
4. `--mode reference` para los runs `ref-mean` y `ref-pca-d{d}` sobre las mismas ventanas.

Los artefactos de cada run quedan en MLflow (`sinca-mlflow/_mlflow/`):
`reconstruction.parquet`, `latent.parquet`, `diagnostics.parquet`, `encoder.keras`,
`decoder_mp.keras`, y `dmd_operator.npz` / `dmd_operator_trunc.npz` cuando hay pase de DMD.
