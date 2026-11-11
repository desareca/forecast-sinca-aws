## ADDED Requirements

### Requirement: Preparación de datos para red neuronal

El sistema SHALL construir, desde los datos validados (`sinca-data/validated/`), ventanas `[L × 6]` de las 6 series de MP (pm25/pm10 × 3 estaciones) sobre una grilla horaria completa, con `L ∈ {24, 72, 168}` horas, junto con ventanas auxiliares (`pres`, `wspd` y `rain` × 3 estaciones, `temp` de Ñielol y Las Encinas, `blh`) y canales de calendario. Los huecos de ≤ 6 horas SHALL interpolarse linealmente; una ventana de MP con un hueco > 6 horas SHALL descartarse; en los canales auxiliares un hueco > 6 horas SHALL quedar en 0 con una máscara de entrada. Una muestra SHALL asignarse a train, valid o test solo si toda su ventana cae dentro de ese conjunto.

#### Scenario: Ventanas de MP con huecos
- **WHEN** una ventana de MP contiene un hueco de más de 6 horas
- **THEN** la muestra se descarta de entrenamiento y evaluación, y los huecos de 6 horas o menos quedan interpolados con la máscara de valores reales preservada

#### Scenario: Canales auxiliares excluidos
- **WHEN** se arma el conjunto de canales auxiliares
- **THEN** no se incluyen `rhum`, `glob` ni `temp` de Padre Las Casas II

### Requirement: Transformación, recorte y revisión de rangos

El sistema SHALL aplicar `log1p` a `pm25`, `pm10`, `rain`, `blh` y `wspd`, y z-score por canal con media y desviación calculadas solo con el train de cada partición. El sistema SHALL recortar a ±5 desviaciones solo los canales auxiliares y NO SHALL recortar el MP. Antes de entrenar, el sistema SHALL producir un reporte por canal con mínimo, máximo, percentiles y fracción recortada.

#### Scenario: Estadísticas solo de train
- **WHEN** se normaliza una partición
- **THEN** la media y la desviación usadas provienen únicamente del conjunto de train de esa partición

#### Scenario: Reporte de rangos
- **WHEN** se prepara el dataset para entrenar
- **THEN** se genera el reporte de rangos por canal y no se recorta ningún valor de MP

### Requirement: Autoencoder con rama auxiliar y latente en bloques

El sistema SHALL implementar, en TensorFlow + Keras, autoencoders con un encoder de MP, una rama auxiliar que recibe meteo, capa límite y calendario, un latente `z = [z_mp | z_aux]`, un decoder de MP que recibe el latente completo y un decoder auxiliar que reconstruye meteo y capa límite. El sistema SHALL ofrecer las variantes `mlp`, `conv1d`, `gru` (o `lstm`) y `linear_decoder`, con salida de decoder lineal.

#### Scenario: Variantes disponibles
- **WHEN** se solicita entrenar una arquitectura
- **THEN** el sistema construye el modelo de la variante pedida con el mismo contrato de entrada y salida

#### Scenario: Decoder de MP con latente completo
- **WHEN** el decoder de MP reconstruye una ventana
- **THEN** recibe tanto `z_mp` como `z_aux`

#### Scenario: Control sin meteo
- **WHEN** se entrena con `aux_input` apagado
- **THEN** la rama auxiliar recibe solo calendario

### Requirement: Loss con pesos separados

El sistema SHALL minimizar `w_mp · E_mp + w_aux · E_aux`, donde cada término es el error cuadrático medio enmascarado (solo valores reales) promediado por canal, con `w_mp = 1` y `w_aux` configurable (barrido `{0, 0.1, 0.5, 1}`). El calendario NO SHALL tener término de loss.

#### Scenario: Peso auxiliar nulo
- **WHEN** `w_aux = 0`
- **THEN** la loss total depende solo del error de reconstrucción del MP

#### Scenario: Valores interpolados
- **WHEN** se calcula el error en una posición con valor interpolado o faltante
- **THEN** esa posición no contribuye a la loss ni a las métricas

### Requirement: Compuerta de reconstrucción y diagnósticos de aplanamiento

El sistema SHALL evaluar la reconstrucción de MP en µg/m³ por serie y SHALL compararla contra un predictor del promedio y contra PCA del mismo tamaño de latente. El sistema SHALL calcular diagnósticos de aplanamiento (razón de varianza, pendiente real vs reconstruido, espectro de potencia, error y sesgo por tramos de concentración, captura de máximo y percentiles altos), diagnósticos del latente (dimensión efectiva, dimensiones muertas) y la ablación de `z_aux`. La compuerta SHALL ser dura sobre MP, con umbrales fijados tras la corrida de humo, y solo los modelos que la pasan SHALL ser elegibles para el pase de DMD y el ranking.

#### Scenario: Modelo que no supera la referencia
- **WHEN** la reconstrucción de MP de un modelo no supera a PCA del mismo tamaño en el agregado (media de las razones `RMSE_AE/RMSE_PCA` por serie ≥ 1), o excede la tolerancia de 2 % frente a PCA en alguna serie
- **THEN** la compuerta lo marca como no aprobado y queda excluido del pase de DMD

#### Scenario: Empate con PCA dentro de la tolerancia
- **WHEN** en una serie el RMSE del AE es igual o hasta 2 % mayor que el de PCA del mismo tamaño y en el agregado el AE supera a PCA
- **THEN** esa serie no impide aprobar la compuerta

#### Scenario: Compuerta sin umbrales numéricos
- **WHEN** no se aplican umbrales numéricos a los diagnósticos de aplanamiento
- **THEN** `gate_pass` se calcula igualmente con los criterios relativos (el AE supera al predictor del promedio en todas las series y a PCA del mismo tamaño en el agregado) y los umbrales numéricos se suman al criterio cuando se aplican

#### Scenario: Ablación del bloque auxiliar
- **WHEN** se evalúa un modelo con `z_aux` puesto en cero
- **THEN** se registra el empeoramiento de la reconstrucción de MP como medida del aporte auxiliar

#### Scenario: Diagnósticos persistidos
- **WHEN** termina el entrenamiento de un AE
- **THEN** el run registra las métricas de reconstrucción y de aplanamiento por serie y los artefactos de reconstrucción (`reconstruction.parquet`), de latente (`latent.parquet`, con columnas por bloque `z_mp`/`z_aux`) y los submodelos `encoder.keras` y `decoder_mp.keras`

### Requirement: Selección por arquitectura y protocolo de evaluación

El sistema SHALL realizar una búsqueda de hiperparámetros por arquitectura sobre pocas ventanas equiespaciadas del rolling y SHALL reportar la mejor configuración de cada arquitectura. El modelo final de cada finalista SHALL entrenarse con todo lo anterior al test y evaluarse una sola vez en el test held-out `2026-07-01 → 2026-10-04`, con 3 semillas para reportar la varianza de la reconstrucción. El sistema NO SHALL optimizar parámetros del DMD en esta etapa.

#### Scenario: Test sin tuning
- **WHEN** se ejecuta la búsqueda de hiperparámetros
- **THEN** el test held-out no participa en ninguna decisión de selección

#### Scenario: Mejor configuración por arquitectura
- **WHEN** termina la búsqueda
- **THEN** se dispone de la mejor configuración de cada arquitectura, no de un único ganador, elegida por la razón `RMSE_AE/RMSE_PCA` (comparable entre `L`) y no por el RMSE absoluto

### Requirement: Pase de DMD global sobre el latente congelado

El sistema SHALL ajustar un DMD global estándar (mínimos cuadrados sobre pares `(z_t, z_{t+1})`, rango completo y una variante truncada, sin búsqueda de parámetros) sobre el latente congelado de los AE aprobados por la compuerta. El sistema SHALL reportar la linealizabilidad del latente (error de `z` a 1 y 24 pasos) y SHALL derivar un pronóstico informativo decodificando `K^24 z_t` con el decoder de MP y promediando las últimas 24 horas de la ventana decodificada, evaluado sobre las mismas filas y el mismo target que LightGBM.

#### Scenario: Pronóstico comparable
- **WHEN** se evalúa el pase de DMD en el test held-out
- **THEN** se reportan las métricas de pronóstico del baseline sobre la intersección de filas con LightGBM, por serie y para `ICAP_zona`

#### Scenario: Modelo no aprobado
- **WHEN** un AE no pasa la compuerta
- **THEN** no se le aplica el pase de DMD

#### Scenario: Pase encadenado al entrenamiento final
- **WHEN** el entrenamiento final de un AE termina y la compuerta se aprueba
- **THEN** el pase de DMD se ejecuta en el mismo job sin descargar artefactos, y un modo independiente puede repetirlo descargando `encoder.keras` y `decoder_mp.keras` del run

#### Scenario: Operador persistido
- **WHEN** se ajusta el DMD de un AE aprobado
- **THEN** el run registra el operador (`K`, rango y autovalores) como artefacto `dmd_operator.npz`

### Requirement: Runtime de training TF con tags explícitos

El sistema SHALL disponer de una imagen de training TensorFlow solo-runtime en el repositorio ECR `sinca-train`, con tags explícitos (`lgbm-v1`, `tf-cpu-v1`) y sin uso de `:latest`, con el código entregado por `source_dir`/`entry_point` desde `pipeline/`.

#### Scenario: Imágenes coexistentes
- **WHEN** se publica la imagen TF
- **THEN** la imagen LightGBM sigue disponible con su tag explícito y no queda sobrescrita

#### Scenario: Corrida de humo en CPU
- **WHEN** se ejecuta la corrida de humo en CPU
- **THEN** se mide el tiempo por epoch de cada arquitectura para decidir CPU o GPU
