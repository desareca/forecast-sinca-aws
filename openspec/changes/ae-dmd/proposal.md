> **Estado: BORRADOR EN DISCUSIÓN.** Este documento y `design.md` se iteran con
> el operador antes de generar `specs/` y `tasks.md`. Los puntos marcados
> **PA-n** (puntos de ajuste) están abiertos y se detallan en `design.md` §
> "Puntos de ajuste". Nada de esto es implementable hasta que estén cerrados.

## Why

El baseline (Fase 3) fijó la vara: LightGBM mejora a la persistencia en 4/6
series (−10% a −18% RMSE en test) pero **empeora en Padre Las Casas II**
(pm10/pm25), la serie de mayor variabilidad. Además, LightGBM trata las 6
series de forma independiente y predice un único escalar (promedio 24h), sin
explotar ni la dinámica temporal horaria ni la información cruzada entre
estaciones. La Fase 4 abre la línea **autoencoder + DMD/Koopman** prevista
desde el inicio (`prompt-sinca-mlops.md` §4/§5): un modelo que aprende un
espacio latente donde la dinámica regional de MP es (aproximadamente) lineal,
y que por construcción produce trayectorias multi-step y comparte información
entre las 3 estaciones. Es ahora porque el baseline y el dashboard de análisis
(Fase 3.5) ya están cerrados y permiten comparar 1 a 1.

## What Changes

- **Nueva familia de modelos AE+DMD/Koopman** (`pipeline/koopman/` o similar,
  TensorFlow + Keras), entrenada en SageMaker Training Job, con al menos la **variante
  base** (RMSE ponderado por serie) comparable 1 a 1 contra persistencia y
  LightGBM sobre el **mismo test held-out** y el mismo target evaluado
  (`mean(MP(t+1 … t+24h))`).
- **Escalera interna de la línea** (PA-1): de menor a mayor complejidad, para
  saber de dónde viene la ganancia (p. ej. DMD lineal sobre embedding de
  retardos → AE + DMD en dos etapas → Deep Koopman end-to-end).
- **Variante experimental de loss conjunta** (PA-9): `Σ RMSE_ponderado(serie_i)
  + λ · loss(max_predicho, ICAP_zona_real)`, como experimento adicional, no
  reemplazo.
- **Preparación de datos para red neuronal**: manejo de huecos (las redes no
  toleran `NaN` como LightGBM), normalización ajustada solo en train, ventanas
  de entrada/salida (PA-4, PA-5).
- **Runtime de entrenamiento con TensorFlow + Keras**: imagen ECR en el mismo
  repo `sinca-train` con tags explícitos (PA-8, cerrado) siguiendo el patrón `compute` (solo runtime, código por `source_dir`).
- **Tracking**: runs AE+DMD en MLflow con el mismo set de métricas y los mismos
  artefactos de diagnóstico que el baseline (`predictions.parquet`,
  `zone_icap.parquet`, `rolling_scores.parquet`), más diagnósticos propios
  del operador de Koopman (autovalores / períodos) en lugar de
  `feature_importance.parquet` (PA-10).
- **Dashboard**: ajuste mínimo para que la familia nueva aparezca en las
  comparativas existentes (ya es agnóstico de familia vía `model_family`).
- **`EXPERIMENT-LOG.md`**: una entrada por experimento de la escalera.
- **Fuera de alcance a confirmar** (PA-11): mejoras a LightGBM (features
  cruzadas entre estaciones, `rain`/viento Open-Meteo para PLC),
  Holt-Winters/SARIMA de la escalera original.

## Capabilities

### New Capabilities
- `modelado-koopman`: entrenamiento y evaluación de la línea AE+DMD/Koopman
  (preparación de datos para red neuronal, arquitectura, loss base y conjunta,
  rollout multi-step, evaluación comparable con el baseline, diagnósticos del
  operador lineal).

### Modified Capabilities
- `modelado`: el requisito de evaluación con ventana deslizante y el de
  persistencia de diagnósticos se generalizan a "cualquier familia de modelo"
  (hoy redactados para LightGBM); posible ajuste del protocolo de ventanas para
  modelos costosos (PA-6).
- `tracking-experimentos`: convención de nombre/tag para un modelo **conjunto**
  que cubre las 6 series en un solo entrenamiento (PA-7).
- `analisis-modelos`: el dashboard muestra la familia AE+DMD y tolera la
  ausencia de `feature_importance.parquet` (o muestra su artefacto
  equivalente).
- `iam`: solo si se crea un repo ECR nuevo (pull acotado a ese repo en
  `sinca-training-role`). Sin cambios de permisos S3.

## Impact

- **Código**: nuevo módulo de modelado TF/Keras bajo `pipeline/`; reutiliza
  `features/build.py` (partición, carga), `evaluate/evaluate.py` (métricas,
  `zone_icap`) y `train/mlflow_log.py` (sync de estado). `launch.py` gana
  selección de entry point / instancia.
- **Docker / ECR**: imagen de training con TensorFlow (CPU o GPU según PA-3),
  tag explícito en `sinca-train`; la imagen LightGBM pasa a tag explícito
  también (deja de usarse `:latest`).
- **Infra (Terraform)**: posible repo ECR nuevo + statement de pull en
  `sinca-training-role`. Nada más; buckets y prefijos no cambian.
- **Cuotas**: si se usa GPU, la cuota `ml.g4dn.xlarge for training job usage`
  probablemente esté en 0 → pedido a Service Quotas con 1-2 días hábiles de
  anticipación (nota del blueprint `compute`).
- **Costos**: Training Jobs on-demand, disparo manual; objetivo mantenerse
  dentro de < USD 5/mes del proyecto (estimación en `design.md`).
- **Dependencias**: `tensorflow` (Keras 3 incluido; nada más pesado). DMD se
  implementa con `numpy`/`tf.linalg`, sin librerías externas como `pydmd` salvo decisión
  explícita).
