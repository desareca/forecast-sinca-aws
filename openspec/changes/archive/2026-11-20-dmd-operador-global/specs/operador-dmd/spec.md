# Spec Delta

## Purpose

Operador lineal global post-hoc (DMD/Koopman) sobre representaciones de las 6 series de MP
(retardos, PCA y latentes de AE), con evaluación común, compuertas de decisión y confirmación única
en test, para elegir el mejor operador interpretable frente a LightGBM y persistencia.

## ADDED Requirements

### Requirement: Cadena común de evaluación

El sistema SHALL evaluar toda variante con una misma cadena: estado → operador → rollout → decodificación → promedio de las 24 h siguientes → métrica, igual para Hankel, PCA y AE. La métrica SHALL calcularse en `sel` sobre un conjunto fijo de filas por serie: las de los pases DMD v2 de `ae-dmd-prep` ∩ LightGBM v2 ∩ persistencia v2, el mismo para todas las variantes.

#### Scenario: Mismas filas
- **WHEN** se comparan dos variantes de una etapa
- **THEN** ambas se evalúan sobre las mismas filas por serie y el resumen reporta `n` por serie

#### Scenario: Variante sin pronóstico en una fila
- **WHEN** una variante no produce pronóstico para una fila del conjunto fijo
- **THEN** la evaluación falla con un error que identifica la variante y la serie, sin reducir el conjunto en silencio

#### Scenario: Variante con memoria mayor
- **WHEN** una variante necesita una ventana de 48 h que no es elegible en algunas filas
- **THEN** se compara emparejada sobre su intersección con el conjunto fijo, con su `n`, y el conjunto fijo de las demás no cambia

#### Scenario: Continuidad con `ae-dmd-prep`
- **WHEN** se evalúa el DMD exacto de rango completo sobre el latente de un AE `version = 2`
- **THEN** sus pronósticos coinciden con los del pase DMD de `ae-dmd-prep` de esa semilla, dentro de la tolerancia relativa del test

### Requirement: Métricas con intervalo

El sistema SHALL reportar por serie el RMSE ponderado (`w = 1 + α·max(0, y − umbral ICAP100)`, `α = 1`), la razón contra la mejor referencia de la serie (la de menor RMSE ponderado entre LightGBM y persistencia), la razón contra cada referencia y el skill contra persistencia, con intervalo por bootstrap de bloques de días emparejado. Una diferencia entre variantes SHALL contar solo si su intervalo excluye cero.

#### Scenario: Comparación emparejada
- **WHEN** se compara una variante contra la de referencia de la etapa
- **THEN** se reporta la diferencia de RMSE ponderado con su intervalo calculado sobre los mismos remuestreos

#### Scenario: Reproducibilidad
- **WHEN** se repite una evaluación con la misma semilla de bootstrap
- **THEN** los intervalos son idénticos

### Requirement: Diagnósticos obligatorios

Cada variante evaluada SHALL registrar: error vs horizonte (1…24 h), error en episodios (filas con `y ≥` umbral ICAP100), espectro del operador (autovalor, período y vida media por modo) y pendiente y sesgo por tramo de concentración.

#### Scenario: Diagnósticos presentes
- **WHEN** termina la evaluación de una variante
- **THEN** su run contiene los cuatro diagnósticos como artefactos tabulares

### Requirement: Información permitida y ajuste en train

El operador y todo parámetro derivado (rango, pesos, factores de retransformación, PCA, normalización) SHALL ajustarse solo con filas de `train` (con embargo de 24 h donde haya target); marzo 2026 (`es`) solo SHALL usarse como diagnóstico. El rollout NO SHALL usar meteo posterior a `t`: la meteo del horizonte SHALL ser persistencia diurna `meteo(t+k) = meteo(t+k−24)` y el calendario SHALL ser el real de `t+k`.

#### Scenario: Sin meteo futura
- **WHEN** se alteran los valores de meteo posteriores a `t` en los datos de entrada
- **THEN** el pronóstico emitido en `t` no cambia

#### Scenario: Marzo fuera del ajuste
- **WHEN** se ajusta cualquier operador o parámetro de la cadena
- **THEN** ninguna muestra de `es`, `sel` o `test` entra al ajuste, y ningún `t` cuyo `t+24` caiga fuera de `train` aporta a pesos o factores de retransformación

### Requirement: Oráculo de representación

Para cada representación con decodificador (PCA y AE), el sistema SHALL calcular el pronóstico oráculo (codificar la ventana real que termina en `t+24`, decodificarla y promediar sus 24 h) y su razón contra la mejor referencia con intervalo. El oráculo SHALL marcarse como diagnóstico. La lectura "techo en la representación" SHALL darse si el intervalo de esa razón alcanza 0.95 o más.

#### Scenario: Techo por semilla
- **WHEN** se evalúa el oráculo de conv1d y GRU
- **THEN** se reporta para cada una de las 3 semillas y su media, con la lectura por intervalo

### Requirement: Representaciones de la escalera

El sistema SHALL construir, con las mismas ventanas y entradas: Hankel de MP (24 retardos), Hankel de MP + 12 canales meteo, Hankel por estación, Hankel con 48 retardos, PCA(12) de la ventana de MP + PCA(4) de la ventana auxiliar, y los latentes conv1d y GRU de las 3 semillas. En Hankel el rango por defecto SHALL ser el umbral óptimo de Gavish–Donoho.

#### Scenario: Hankel recupera dinámica conocida
- **WHEN** se ajusta DMD sobre el Hankel de una sinusoide amortiguada sintética
- **THEN** el espectro recupera su período y su vida media dentro de la tolerancia del test

#### Scenario: Rango automático
- **WHEN** se ajusta un operador sobre Hankel sin rango explícito
- **THEN** se usa y se registra el rango de Gavish–Donoho

### Requirement: Control numpy y `pydmd`

El DMD exacto en numpy SHALL ser el control y, cuando `pydmd` esté disponible, SHALL coincidir con `pydmd` en autovalores dentro de `1e-6` sobre datos sintéticos. Los demás estimadores (ponderado, fbDMD, recorte) SHALL implementarse en numpy y validarse contra su fórmula en coordenadas completas y contra la verdad de sistemas sintéticos.

#### Scenario: Coincidencia del DMD exacto
- **WHEN** se corre la suite de tests con `pydmd` instalado
- **THEN** el DMD exacto coincide con `pydmd` en autovalores dentro de `1e-6`

#### Scenario: fbDMD contra la verdad
- **WHEN** se ajusta fbDMD sobre un sistema sintético con ruido de medición y de proceso
- **THEN** sus autovalores quedan más cerca de los verdaderos que los del DMD exacto

### Requirement: Ponderación de picos y retransformación

El sistema SHALL ofrecer el ajuste por mínimos cuadrados ponderados del operador, con peso de cada par igual al máximo de `w` entre las 6 series, y la corrección de retransformación de `log1p` por smearing de Duan con factores estimados en `train`. Ambas SHALL poder combinarse con cualquier estimador y forma de rollout.

#### Scenario: Pesos unitarios
- **WHEN** todos los pesos son 1
- **THEN** el ajuste ponderado coincide con el DMD exacto

### Requirement: Exógenas en el rollout

El sistema SHALL ofrecer el rollout autónomo y el exógeno, donde solo el bloque de MP es estado (`z_mp,t+1 = K_mm·z_mp,t + K_ma·z_aux,t`). En AE y PCA, en cada paso el bloque auxiliar se recalcula desde la ventana auxiliar con meteo y máscaras por persistencia diurna y calendario real. En Hankel la entrada SHALL ser la meteo por persistencia diurna y el calendario de cada paso, sin máscaras.

#### Scenario: Ventana auxiliar del horizonte
- **WHEN** se arma la entrada auxiliar del paso `k`
- **THEN** cada hora `τ > t` usa la meteo (y en AE/PCA la máscara) observadas en `τ − 24` y el calendario real de `τ`

### Requirement: Horizonte del operador

El operador principal SHALL ser horario con rollout de 24 pasos. El sistema SHALL ofrecer el operador directo a 24 h solo como referencia de precisión.

#### Scenario: Referencia directa
- **WHEN** se evalúa el operador directo a 24 h
- **THEN** se reporta junto al horario y se marca como referencia, no como candidato

### Requirement: Estimadores y estabilidad

El sistema SHALL ofrecer DMD exacto, recorte de autovalores a `|λ| ≤ 1 − ε` y fbDMD (siempre evaluado con recorte), y SHALL estimar la incertidumbre por modo con bootstrap por bloques de días del `train`, emparejando modos entre re-estimaciones y reportando intervalos de período y vida media y la frecuencia de aparición de cada modo.

#### Scenario: Operador recortado estable
- **WHEN** se aplica el recorte a un operador con algún `|λ| > 1 − ε`
- **THEN** el operador resultante es real y su radio espectral es ≤ `1 − ε`

### Requirement: Rango y modos sin grilla

El sistema SHALL partir del rango completo en PCA y AE (Gavish–Donoho solo se reporta; en Hankel acota el rango) y eliminar hacia atrás, de a uno (un par conjugado junto), el modo de menor empeoramiento del error ponderado a 24 h mientras la diferencia emparejada contra la configuración inicial no sea significativamente peor; SHALL detenerse en el primer paso significativamente peor conservando el previo y registrar la curva de precisión vs número de modos. `d` del AE no cambia.

#### Scenario: Modos de ruido eliminados
- **WHEN** se seleccionan modos en un sistema sintético con 2 modos relevantes y modos de ruido
- **THEN** se eliminan los modos de ruido y se conservan los 2 relevantes

#### Scenario: Comparación contra la configuración inicial
- **WHEN** quitar modos uno a uno empeora poco en cada paso pero el acumulado respecto de la configuración inicial es significativo
- **THEN** la eliminación se detiene en el paso previo al acumulado significativo

#### Scenario: Curva registrada
- **WHEN** termina la selección de modos de una representación
- **THEN** el run contiene el error ponderado a 24 h por número de modos con su intervalo y el orden de eliminación

### Requirement: Modo diario protegido

En la selección de modos el par conjugado de período ≈ 24 h SHALL conservarse siempre (el target de 24 h lo cancela y la métrica no puede juzgarlo); sus armónicos (≈ 12 h, ≈ 8 h) SHALL protegerse solo si aparecen en ≥ 80 % del bootstrap por modo.

#### Scenario: Diario conservado
- **WHEN** se seleccionan modos de un operador con un par de período ≈ 24 h que no cambia el error a 24 h
- **THEN** el par diario se conserva y la selección no colapsa a un solo modo

### Requirement: Compuertas con decisión del operador

Cada etapa (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6) SHALL cerrarse con sus resultados registrados en `EXPERIMENT-LOG.md` y una decisión del operador antes de iniciar la siguiente; la variante ganadora SHALL arrastrarse a las etapas siguientes. La escalera SHALL entregar a lo más 2 representaciones.

#### Scenario: Etapa siguiente sin decisión
- **WHEN** una etapa termina sin decisión registrada
- **THEN** la etapa siguiente no se ejecuta

### Requirement: Confirmación única en test

El ganador SHALL reajustarse con datos hasta 2026-06-30 manteniendo el rango y los modos fijados en 1.6 (emparejados con los del reajuste, sin re-seleccionar) y evaluarse una sola vez en test (2026-07-01 → 2026-10-04) contra LightGBM y persistencia con el mismo corte de información, sobre filas comunes. Ninguna cifra de test SHALL usarse para elegir variantes.

#### Scenario: Una sola evaluación
- **WHEN** ya existe un run de test del operador en esta versión
- **THEN** una nueva evaluación en test se rechaza salvo indicación explícita del operador

#### Scenario: Modos fijados
- **WHEN** se reajusta el operador ganador con datos hasta junio
- **THEN** conserva el rango y el número de modos de 1.6 y cada modo conservado es el emparejado con uno seleccionado en `sel`

### Requirement: Veredicto del criterio de éxito

El sistema SHALL emitir un único veredicto. Precisión: razón contra la mejor referencia ≤ 1.05 en ≥ 4/6 series (una serie falla solo si todo el intervalo queda sobre 1.05); PLC II además skill > 0 vs persistencia; en las 6 series, el intervalo del skill vs persistencia no queda entero bajo 0. Interpretabilidad: ≤ 10 modos, dominantes en las 3 semillas y en ≥ 80 % del bootstrap, con intervalos que se solapan.

#### Scenario: Serie con intervalo que cruza el umbral
- **WHEN** la razón puntual de una serie es 1.07 y su intervalo es [1.02, 1.12]
- **THEN** la serie cuenta como que cumple la precisión

#### Scenario: Piso de persistencia
- **WHEN** en una serie el intervalo del skill vs persistencia queda entero bajo 0
- **THEN** el veredicto de precisión es "no cumple", aunque ≥ 4/6 series cumplan la razón

### Requirement: Entregables de interpretabilidad

El change SHALL entregar para el ganador: espectro con intervalos de período y vida media, formas de modo por estación y contaminante (fase como desfase), descomposición de episodios por modo, estabilidad de modos entre semillas y bootstrap, y la matriz de forzamiento `K_ma` si el rollout es exógeno.

#### Scenario: Entregables completos
- **WHEN** se cierra el change
- **THEN** existen las cinco piezas en el notebook versionado con outputs
