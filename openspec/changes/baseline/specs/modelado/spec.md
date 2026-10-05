## ADDED Requirements

### Requirement: Feature engineering desde datos validados

El sistema SHALL construir features desde los datos validados (`sinca-data/validated/`) para el entrenamiento: lags de la serie propia (1h, 24h, 48h, 72h, 168h), meteorología con lag 24h, altura de capa límite, feriados, día de semana (cyclic), ventana GEC, y Fourier (diaria, semanal, anual).

#### Scenario: Features construidas
- **WHEN** se ejecuta el feature engineering sobre los datos validados
- **THEN** se genera un dataset con las features y el target por serie

### Requirement: Target promedio móvil 24h a t+24h

El sistema SHALL calcular el target de cada serie como el promedio móvil de 24h a t+24h (`mean(MP(t+1 … t+24h))`), que es el valor que consume la fórmula oficial del ICAP.

#### Scenario: Target calculado
- **WHEN** se construye el dataset de entrenamiento
- **THEN** cada fila tiene el target `mean(MP(t+1 … t+24h))` de su serie

### Requirement: Baseline de persistencia

El sistema SHALL disponer de un baseline de persistencia (naive) que predice el último promedio móvil 24h observado, sin entrenamiento, como piso de referencia.

#### Scenario: Predicción naive
- **WHEN** se evalúa el baseline de persistencia
- **THEN** la predicción a t+24h es el último promedio móvil 24h observado

### Requirement: Entrenamiento LightGBM por serie

El sistema SHALL entrenar 6 modelos LightGBM independientes (MP2.5/MP10 × 3 estaciones), cada uno minimizando su RMSE ponderado, con búsqueda de hiperparámetros por serie (random search + early stopping).

#### Scenario: Seis modelos entrenados
- **WHEN** se ejecuta el entrenamiento
- **THEN** se entrenan 6 modelos, uno por serie, cada uno con sus hiperparámetros óptimos

### Requirement: Evaluación con ventana deslizante

El sistema SHALL evaluar con ventana deslizante de tamaño fijo (train 2 años, valid 1 mes, step 1 mes) y un test final held-out de 3 meses, reportando el promedio de la métrica sobre las ventanas más la métrica del test final.

#### Scenario: Métricas de evaluación
- **WHEN** se evalúa el modelo
- **THEN** se reportan `train_loss`, `mse`/`rmse` por serie, bloque ICAP oficial (`recall_alerta`/`recall_preemergencia`/`recall_emergencia`) y bloque percentil propio (`recall_top10pct`/`recall_top5pct`/`pr_auc_top10pct`)

### Requirement: Tracking MLflow en modo logging directo

El sistema SHALL loguear los resultados a MLflow self-hosted en modo logging directo (sin servidor), con convención `sinca-{estacion}-{modelo}-v{n}`, subiendo `mlflow.sqlite` y `mlruns/` a `sinca-mlflow/_mlflow/` al terminar.

#### Scenario: Run logueado
- **WHEN** se entrena un modelo
- **THEN** se loguea un run en MLflow con métricas, parámetros y artefactos, y se sube a `sinca-mlflow/_mlflow/`
