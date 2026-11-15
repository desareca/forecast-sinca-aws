## ADDED Requirements

### Requirement: Partición de selección `ae-dmd`

El sistema SHALL ofrecer, además de la partición `holdout` actual (defecto, sin cambios), una partición con nombre `ae-dmd` con cuatro tramos `[inicio, fin)` en hora local `America/Santiago`: train `2021-01-01 → 2026-03-01`, early stopping (`es`) `2026-03-01 → 2026-04-01`, validación de selección (`sel`) `2026-04-01 → 2026-07-01` y test `2026-07-01 → 2026-10-05`.

#### Scenario: Selección de partición
- **WHEN** un entrenamiento se lanza sin indicar partición
- **THEN** usa la partición `holdout` con el mismo comportamiento que antes de este change

#### Scenario: Tramos de `ae-dmd`
- **WHEN** se pide la partición `ae-dmd`
- **THEN** sus tramos son exactamente los cuatro definidos y el test coincide con el held-out `2026-07-01 → 2026-10-04`

### Requirement: Pertenencia de ventanas y filas a un tramo

Una ventana de AE SHALL pertenecer a un tramo solo si cae completa dentro de él. Una fila con target `mean(MP(t+1…t+24))` SHALL pertenecer a un tramo solo si `inicio ≤ t < fin − 24 h` (embargo de 24 h).

#### Scenario: Embargo en el borde del tramo
- **WHEN** una fila tiene `t` dentro de las últimas 24 h de un tramo
- **THEN** esa fila no se usa para entrenar, detener ni evaluar en ese tramo

#### Scenario: Ventana que cruza tramos
- **WHEN** una ventana de AE empieza en un tramo y termina en el siguiente
- **THEN** la ventana no pertenece a ninguno de los dos

### Requirement: Reentrenamiento de AE en la partición `ae-dmd`

El sistema SHALL entrenar un AE en la partición `ae-dmd` normalizando con su train, deteniendo con `es` y calculando métricas de reconstrucción, referencias (promedio y PCA del mismo `d`), compuerta, ablación de `z_aux` y diagnósticos del latente sobre `sel`. El sistema SHALL guardar el latente de los cuatro tramos y NO SHALL calcular métricas, reconstrucciones ni pronósticos sobre el test en esta partición. Cada semilla SHALL registrarse como un run propio.

#### Scenario: Latentes de los cuatro tramos
- **WHEN** termina el entrenamiento de un AE en `ae-dmd`
- **THEN** `latent.parquet` contiene filas con `split` en `train`, `es`, `sel` y `test`

#### Scenario: Test sin métricas
- **WHEN** se registra un AE entrenado en `ae-dmd`
- **THEN** el run no tiene métricas ni filas de reconstrucción del test

### Requirement: Pase de DMD en la validación de selección

En la partición `ae-dmd`, el sistema SHALL aplicar el DMD exacto actual (rango completo y truncado) sobre el latente congelado de cada semilla entrenada, con el operador ajustado solo con latentes de train, SHALL registrar la linealizabilidad en `es` y `sel` y SHALL registrar un run por serie con el pronóstico `mean(MP(t+1…t+24))` de las filas de `sel`. En esta partición el pase NO SHALL depender de la compuerta de reconstrucción.

#### Scenario: Semilla que no aprueba la compuerta
- **WHEN** un AE entrenado en `ae-dmd` no aprueba la compuerta
- **THEN** el pase de DMD se ejecuta igual y el run del AE conserva `gate_pass = 0`

#### Scenario: Predicciones del pase en `sel`
- **WHEN** termina el pase de DMD de una semilla
- **THEN** existen 6 runs por serie con `predictions.parquet` cuyas filas tienen `split = sel` y `t` dentro de `sel` con embargo

### Requirement: Referencias de pronóstico en la validación de selección

El sistema SHALL producir, para la partición `ae-dmd`, `predictions.parquet` de persistencia y de LightGBM sobre las filas de `sel`. LightGBM SHALL reajustarse por serie con el train y el `es` de la partición, usando los hiperparámetros de un run fuente indicado explícitamente por serie y sin nueva búsqueda. El sistema SHALL rechazar la ejecución si falta el run fuente de alguna serie o si su serie no coincide.

#### Scenario: Mismo corte de información
- **WHEN** se reajusta LightGBM para `ae-dmd`
- **THEN** ninguna fila de entrenamiento o de early stopping tiene target que use datos posteriores a `2026-04-01`

#### Scenario: Run fuente ausente
- **WHEN** falta el run fuente de una serie
- **THEN** la ejecución termina con error antes de entrenar y no registra runs

### Requirement: Contraste del DMD numpy con `pydmd`

Cuando `pydmd` esté disponible en el entorno, el sistema SHALL ajustar sobre el mismo segmento contiguo de latente el DMD exacto numpy y el de `pydmd` (rango completo y truncado), SHALL registrar la máxima diferencia entre autovalores y, en rango completo, entre las predicciones a un paso, y SHALL marcar el contraste como aprobado solo si ambas diferencias son ≤ `1e-6`. Si `pydmd` no está disponible, el sistema SHALL registrarlo y continuar.

#### Scenario: Implementaciones coincidentes
- **WHEN** se contrasta un sistema lineal conocido
- **THEN** las diferencias son ≤ `1e-6` y el contraste queda aprobado

#### Scenario: Sin `pydmd`
- **WHEN** el pase de DMD corre en un entorno sin `pydmd`
- **THEN** el pase termina sin error y el run queda marcado como sin contraste

## MODIFIED Requirements

### Requirement: Compuerta de reconstrucción y diagnósticos de aplanamiento

El sistema SHALL evaluar la reconstrucción de MP en µg/m³ por serie y SHALL compararla contra un predictor del promedio y contra PCA del mismo tamaño de latente. El sistema SHALL calcular diagnósticos de aplanamiento (razón de varianza, pendiente real vs reconstruido, espectro de potencia, error y sesgo por tramos de concentración, captura de máximo y percentiles altos), diagnósticos del latente (dimensión efectiva, dimensiones muertas) y la ablación de `z_aux`. La compuerta SHALL ser dura sobre MP, con umbrales fijados tras la corrida de humo. En la partición `holdout`, solo los modelos que la pasan SHALL ser elegibles para el pase de DMD y el ranking. En la partición `ae-dmd`, la compuerta SHALL calcularse sobre `sel` y SHALL registrarse como informativa, sin bloquear el pase de DMD.

#### Scenario: Modelo que no supera la referencia
- **WHEN** en la partición `holdout` la reconstrucción de MP de un modelo no supera a PCA del mismo tamaño en el agregado (media de las razones `RMSE_AE/RMSE_PCA` por serie ≥ 1), o excede la tolerancia de 2 % frente a PCA en alguna serie
- **THEN** la compuerta lo marca como no aprobado y queda excluido del pase de DMD

#### Scenario: Empate con PCA dentro de la tolerancia
- **WHEN** en una serie el RMSE del AE es igual o hasta 2 % mayor que el de PCA del mismo tamaño y en el agregado el AE supera a PCA
- **THEN** esa serie no impide aprobar la compuerta

#### Scenario: Compuerta sin umbrales numéricos
- **WHEN** no se aplican umbrales numéricos a los diagnósticos de aplanamiento
- **THEN** `gate_pass` se calcula igualmente con los criterios relativos (el AE supera al predictor del promedio en todas las series y a PCA del mismo tamaño en el agregado) y los umbrales numéricos se suman al criterio cuando se aplican

#### Scenario: Compuerta informativa en `ae-dmd`
- **WHEN** un AE entrenado en la partición `ae-dmd` no aprueba la compuerta en `sel`
- **THEN** se registra `gate_pass = 0` y el modelo sigue siendo elegible para el pase de DMD

#### Scenario: Ablación del bloque auxiliar
- **WHEN** se evalúa un modelo con `z_aux` puesto en cero
- **THEN** se registra el empeoramiento de la reconstrucción de MP como medida del aporte auxiliar

#### Scenario: Diagnósticos persistidos
- **WHEN** termina el entrenamiento de un AE
- **THEN** el run registra las métricas de reconstrucción y de aplanamiento por serie y los artefactos de reconstrucción (`reconstruction.parquet`), de latente (`latent.parquet`, con columnas por bloque `z_mp`/`z_aux`) y los submodelos `encoder.keras` y `decoder_mp.keras`
