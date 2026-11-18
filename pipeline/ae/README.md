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

`<AWS_PROFILE>` es el profile de la cuenta personal. **No uses `$env:AWS_PROFILE`**: la shell
puede traer el profile de otra cuenta.

```powershell
python pipeline/train/launch.py --profile <AWS_PROFILE> --role-arn <training_role_arn> `
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

## Partición `ae-dmd` (change `ae-dmd-prep`)

Partición nueva (`features/partition.py`), la que usa `dmd-operador-global`. `holdout` sigue
siendo el defecto y no cambia.

- Tramos (fechas locales, `[inicio, fin)`): `train` 2021-01 a 2026-02, `es` 2026-03 (early
  stopping), `sel` 2026-04 a 2026-06 (selección), `test` 2026-07-01 a 2026-10-04.
- `--partition ae-dmd`: normaliza con `train`; early stopping con `es`; compuerta y métricas
  de reconstrucción en `sel`. El tramo `test` solo aporta latentes (`latent.parquet`): no hay
  métricas de test ni pronóstico en este change.
- Pase DMD tras el `final`, **en cada semilla y sin mirar la compuerta** (la compuerta se
  registra igual, pero no bloquea). Con `--mode dmd --partition ae-dmd` tampoco se exige.
- Pronóstico de DMD solo en `sel`, con embargo de 24 h (`row_mask`): ningún target usa datos
  del tramo siguiente.
- `--mlflow-runs all|none|ID,ID` (defecto `all`): descarga selectiva de MLflow. En `--mode dmd`
  se agrega siempre `--ae-run-id`.

```powershell
python pipeline/ae/run.py --mode final --arch conv1d --length 24 --d 16 --partition ae-dmd `
    --seeds 0,1,2 --mlflow-runs none --local-validated-dir ./validated --mlflow-dir ./mlflow-local `
    --model-dir ./model --output-dir ./output --no-upload
```

Nombres de run (todos con `-s{seed}`): AE `sinca-zona-ae_{arq}-aedmd-v{n}-s{seed}`;
DMD por serie `sinca-{slug}-ae-{arq}-dmd-aedmd-v{n}-s{seed}`; referencias `ref-mean-L{L}-v{n}`
y `ref-pca-d{d}-L{L}-v{n}`. Los runs de `holdout` cambian el nombre del AE a
`sinca-zona-ae_{arq}-v{n}-s{seed}` (los históricos no se renombran salvo con el plan de
`scripts/mlflow/maintenance.py`).

Las cifras de referencia de esta partición no son comparables con las de `holdout`: otro
corte y otro tramo de evaluación.

## Contraste con `pydmd`

Cada pase DMD compara su estimador numpy con `pydmd==2025.8.1` sobre el segmento contiguo más
largo de `train` (`dmd.pydmd_crosscheck`): diferencia de autovalores y, en rango completo, de
la predicción a 1 paso. Tolerancia `1e-6`.

- Resultado en el run del AE: `dmd_pydmd_ok` (1 = coinciden), `dmd_pydmd_eig_maxdiff_full`,
  `dmd_pydmd_pred1_maxdiff_full`, `dmd_pydmd_eig_maxdiff_trunc`, y el tag `pydmd_version`
  (`no-disponible` si la imagen no trae `pydmd`).
- Una discrepancia no corta el job: queda en el log como warning y `dmd_pydmd_ok = 0`.
- La imagen `sinca-train:tf-cpu-v2` trae `pydmd` (`docker/train-tf-v2.Dockerfile`). Plan B:
  `scripts/ae/pydmd_check.py` en un venv aparte (`requirements-dmd.txt`) sobre `latent.parquet`.
- `pydmd` no entra al entrenamiento TF: trabaja sobre latentes como arrays.
