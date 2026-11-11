## Why

El baseline (Fase 3) fijó la vara: LightGBM mejora a la persistencia en 4/6
series pero **empeora en Padre Las Casas II**, y trata las 6 series de forma
independiente sin explotar la información cruzada entre estaciones. La línea
autoencoder + DMD/Koopman (Fase 4, prevista desde `prompt-sinca-mlops.md`
§4/§5) parte de un requisito previo: que el **autoencoder reconstruya bien las
6 series de MP**, sin aplanar los picos (un AE tiende a suavizar la señal, justo
donde están los episodios que importan para el ICAP). Si la reconstrucción falla,
cualquier dinámica ajustada sobre el latente falla también. Este change
establece y mide esa base, con varias arquitecturas de AE, antes de invertir en
experimentos de DMD (change siguiente, `ae-dmd`).

## What Changes

- **Familia de autoencoders en TensorFlow + Keras** con cuatro variantes de
  encoder/decoder: MLP, Conv1D, GRU/LSTM y decoder lineal. Cada una con un
  encoder de MP y una rama auxiliar, un latente en bloques (`z_MP`, `z_aux`),
  un decoder de MP (que ve todo el latente) y un decoder auxiliar.
- **Datos para red neuronal**: ventanas `[L × 6]` de las 6 series de MP
  (`L ∈ {24, 72, 168}` h) como entrada y objetivo de reconstrucción. Variables
  auxiliares (meteo, capa límite) entran por la rama auxiliar hacia el latente;
  el calendario entra sin loss. Manejo de huecos, transformación (`log1p` +
  z-score ajustado en train, recorte solo en auxiliares) y revisión de rangos.
- **Loss con pesos separados**: `w_mp · error(MP) + w_aux · error(aux)`, cada
  bloque como promedio por canal.
- **Compuerta de reconstrucción** (dura sobre MP) con referencias (predictor del
  promedio y PCA del mismo tamaño) y diagnósticos de aplanamiento (razón de
  varianza, pendiente real-vs-predicho, espectro, error por nivel de
  concentración, colapso del latente, ablación de `z_aux`).
- **Primer pase de DMD global estándar** sobre el latente congelado de los AE
  que pasan la compuerta, con parámetros por defecto (sin optimizar): mide la
  linealizabilidad del latente (error de `z` a 1 y 24 pasos) y un pronóstico
  informativo comparable con persistencia y LightGBM.
- **Búsqueda por arquitectura**: la mejor configuración de cada variante
  (optimizando solo parámetros del AE) sobre pocas ventanas equiespaciadas, más
  el test final held-out sin tocar en el tuning.
- **Runtime de training TF**: imagen en el repo ECR `sinca-train` con tags
  explícitos (`lgbm-v1`, `tf-cpu-v1`), sin `:latest`; corrida de humo en CPU para
  decidir CPU o GPU.
- **Tracking y dashboard**: runs en MLflow con artefactos de reconstrucción y
  métricas `dmd_`; el dashboard tolera runs sin métricas de pronóstico y muestra
  reconstrucción.
- **Notebooks de exploración** (además del dashboard): plantillas para explorar
  los datos de entrada, la reconstrucción y el latente/DMD en JupyterLab del
  Space, sobre artefactos persistidos de cada run. Nivel 1 sin TensorFlow
  (diagnósticos y análisis del DMD sobre el latente); nivel 2 con TensorFlow
  para lo que pasa por el decoder (pronósticos en µg/m³, travesías del latente).
- **`EXPERIMENT-LOG.md`**: entrada por experimento.

Fuera de alcance (pasa a `ae-dmd`): optimización de parámetros del DMD, DMD
local/por regímenes (catálogo PA-14), DMD con control, Koopman end-to-end, loss
conjunta ICAP, medición de variación estacional del operador. Fuera de ambos
changes: LightGBM v2 y Holt-Winters/SARIMA.

## Capabilities

### New Capabilities
- `modelado-ae`: preparación de datos para red neuronal, autoencoders con rama
  auxiliar y latente en bloques, loss con pesos separados, compuerta de
  reconstrucción y diagnósticos de aplanamiento, y primer pase de DMD global
  sobre el latente.

### Modified Capabilities
- `tracking-experimentos`: convención de nombre/tag de runs para modelos
  conjuntos que cubren las 6 series y para runs de reconstrucción sin target de
  pronóstico.
- `analisis-modelos`: el dashboard muestra reconstrucción y métricas `dmd_` y
  tolera runs sin métricas de pronóstico ni `feature_importance.parquet`.

## Impact

- **Código**: nuevo módulo `pipeline/ae/` (datos, modelos, loss, métricas,
  compuerta, DMD en numpy, entry point). Reutiliza `features/build.py`
  (carga y partición), `evaluate/evaluate.py` (métricas de pronóstico del pase
  DMD), `train/mlflow_log.py` y `train/launch.py` (gana selección de imagen,
  instancia y entry point).
- **Notebooks**: tres plantillas nuevas en `notebooks/` (datos, reconstrucción,
  latente/DMD), siguiendo la convención de `analisis_baseline.ipynb`.
- **Docker / ECR**: `docker/train-tf.Dockerfile`; la imagen LightGBM pasa a tag
  explícito `lgbm-v1`.
- **Infra (Terraform)**: sin cambios. `sinca-training-role` ya tiene pull
  acotado a `sinca-train` y permisos S3 suficientes.
- **Cuotas**: `ml.g4dn.xlarge for training job usage` = 1 (verificado); no hace
  falta pedir aumento. La corrida de humo decide si se usa.
- **Costos**: Training Jobs on-demand, disparo manual; corridas chicas en CPU
  `ml.m5.large`, dentro del presupuesto del proyecto.
- **Dependencias**: `tensorflow` (CPU; Keras 3 incluido). DMD con `numpy`, sin
  librerías externas.
