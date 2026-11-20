# Spec Delta

## Purpose

Koopman entrenado de extremo a extremo (encoder, operador lineal estructurado y decoder) sobre las
6 series de MP, para obtener modos identificables entre semillas con precisión comparable al mejor
operador lineal post-hoc, LightGBM y persistencia.

## ADDED Requirements

### Requirement: Representación del Koopman entrenado

El encoder SHALL recibir la ventana de 24 h de MP (6 canales), meteo (12) y sus máscaras de disponibilidad hasta `t`, sin calendario, y producir un único latente `z` de dimensión `d = 10` (8 en el remedio B4). Arquitecturas: GRU(64) + Dense y Conv1D(64, k = 5) + Dense. El decoder principal SHALL ser lineal (`z` → ventana 24×6 de MP); el decoder GRU no lineal SHALL evaluarse como comparación preregistrada.

#### Scenario: Calendario fuera del encoder
- **WHEN** se alteran solo los canales de calendario de la ventana de entrada
- **THEN** el latente `z_t` no cambia

#### Scenario: Descomposición exacta
- **WHEN** el decoder es lineal
- **THEN** la ventana decodificada es la suma exacta de las contribuciones de cada bloque de `z` más el sesgo

### Requirement: Operador estructurado

El operador SHALL ser global, horario y autónomo en el horizonte: `z_{t+1} = K z_t + B u_{t+1}`, con `u` = día de semana, feriado y ventana GEC de `t+1`. `K` SHALL ser diagonal por bloques: un bloque 2×2 diario de período fijo 24 h, dos bloques 2×2 libres y cuatro reales, con decaimiento `ρ = exp(−(δ + softplus(a)))` y techo de vida media de 1000 h.

#### Scenario: Radio acotado
- **WHEN** termina cualquier paso de optimización
- **THEN** todo `|λ|` de `K` es ≤ `2^(−1/1000)` y el período del bloque diario es 24 h

#### Scenario: Variantes de control
- **WHEN** se entrena la variante de K denso, la de período diario libre o la sin `B·u`
- **THEN** se registran como variantes de F1 del encoder GRU y no reemplazan al principal sin decisión del operador

### Requirement: Información permitida y presupuesto fijo

Encoder, decoder, `K`, `B`, normalización y factores de Duan SHALL ajustarse solo con `train` (2021-01 → 2026-02, embargo de 24 h donde haya target); marzo solo como diagnóstico. El horizonte NO SHALL usar meteo posterior a `t`; el calendario de `t+k` sí. Cada fase SHALL entrenar un número fijo de épocas, sin early stopping.

#### Scenario: Sin meteo futura
- **WHEN** se alteran los valores de meteo posteriores a `t`
- **THEN** el pronóstico emitido en `t` no cambia

### Requirement: Pérdidas incrementales

F1 SHALL usar reconstrucción + linealidad a 1 paso; F2 SHALL agregar predicción multi-paso hasta 24 h con currículo de horizonte 1 → 6 → 24; F3 SHALL agregar el RMSE ponderado del promedio de 24 h en µg/m³. F3 SHALL incluir la variante con `L_tgt` sobre ICAP_zona (máximo suave). Cada término SHALL normalizarse por su escala inicial; F2 y F3 SHALL partir de los pesos de la fase previa.

#### Scenario: Variante ICAP_zona
- **WHEN** se entrena la variante `icap` de F3
- **THEN** se reporta con el criterio D12 y las métricas de ICAP_zona, junto a F3 sin `icap`

#### Scenario: Duan comparable
- **WHEN** se evalúa un modelo de F1 o F2
- **THEN** se reporta su pronóstico con y sin corrección de Duan post-hoc

### Requirement: Entrenamiento estable

El entrenamiento SHALL incluir: épocas iniciales con encoder congelado (solo `K`, `B` y decoder) cuando el encoder arranca en caliente, calentamiento lineal del learning rate, recorte de norma del gradiente y orden canónico de bloques por decaimiento al registrar.

#### Scenario: Calentamiento congelado
- **WHEN** un encoder GRU arranca desde una semilla GRU v2
- **THEN** sus pesos no cambian durante las épocas de calentamiento

### Requirement: Escala del decoder lineal

Con decoder lineal, la pérdida SHALL incluir `L_norma = Σ_b (‖W_b‖_F² / n_b − 1)²` sobre las columnas `W_b` del decoder de cada bloque de K (`n_b` = 2 o 1), sin normalizar por escala inicial. Tras el ajuste inicial por mínimos cuadrados, cada bloque SHALL reescalarse (decoder, salida del encoder y filas de `B`) sin cambiar `K` ni las predicciones.

#### Scenario: Reescalado neutro
- **WHEN** se aplica el reescalado por bloque al modelo inicial
- **THEN** reconstrucción y pronóstico a 24 h coinciden con los previos dentro de la tolerancia del test y `L_norma` es ≈ 0

#### Scenario: Sin colapso de escala
- **WHEN** el entrenamiento lleva el latente hacia `z → 0`
- **THEN** `L_norma` crece y penaliza la solución

### Requirement: Evaluación común

Cada modelo SHALL evaluarse con la cadena y las filas fijas de `operador-dmd` (rollout de 24 pasos con `u` conocido, decodificación, promedio, bootstrap emparejado, diagnósticos) contra LightGBM, persistencia y GRU+DMD+Duan de `dmd-operador-global`, y SHALL reportar el veredicto del criterio D12 y la interpretabilidad del §9.

#### Scenario: Referencia de 7.1
- **WHEN** se evalúa un modelo en `sel`
- **THEN** se reporta la diferencia emparejada contra GRU+DMD+Duan de 7.1 con su intervalo, sobre las mismas filas

### Requirement: Modos dominantes

El aporte de un modo SHALL ser la suma ponderada (`w` de la métrica) del cuadrado de su contribución al promedio decodificado de 24 h en `sel` (par conjugado junto; exacta con decoder lineal, por jacobiano con el no lineal). Los dominantes SHALL ser el conjunto mínimo de modos, en orden de aporte, que reúne ≥ 80 % del aporte total. La definición SHALL quedar fija antes de correr F1.

#### Scenario: Dominantes reproducibles
- **WHEN** se calculan los dominantes dos veces sobre el mismo modelo
- **THEN** el conjunto y los aportes son idénticos

### Requirement: Compuerta de identificabilidad

Tras F1, la compuerta SHALL aprobar solo si cada modo dominante aparece en las 3 semillas (emparejamiento húngaro, período y vida media a ≤ 25 %; reales solo vida media) y en ≥ 80 % del bootstrap de cada semilla, con intervalos de período y vida media que se solapan entre semillas. El bootstrap SHALL re-estimar DMD sobre el latente aprendido en bloques de 7 días de `train`.

#### Scenario: Modo presente en dos semillas
- **WHEN** un modo dominante de una semilla no tiene pareja en una de las otras dos
- **THEN** la compuerta no aprueba

### Requirement: Remedios de identificabilidad

Si la compuerta no aprueba, el sistema SHALL probar en orden, uno a la vez y solo con el encoder principal: B1 λ_lin ∈ {0.1, 10} (solo si `L_lin` normalizado > 10× `L_rec` al final de F1), B2 consistencia encoder∘decoder, B3 L1 sobre amplitudes, B4 `d = 8`, deteniéndose en el primero que apruebe. Si ninguno aprueba, el change SHALL pausarse para decisión del operador entre C1 (encoder lineal Hankel/PCA con K y pérdida entrenados) y C2 (ensamble de semillas).

#### Scenario: Sin desbalance
- **WHEN** la compuerta falla y `L_lin` normalizado al final de F1 es ≤ 10× `L_rec`
- **THEN** B1 no se ejecuta y se pasa a B2

#### Scenario: Remedio que aprueba
- **WHEN** B3 aprueba la compuerta
- **THEN** B4 no se ejecuta y F2 parte del modelo de B3

### Requirement: Selección de encoder

El conv1d somero SHALL pasar a F2 solo si en F1 iguala o supera al GRU en identificabilidad (resultado de la compuerta) y no es significativamente peor en precisión.

#### Scenario: conv1d peor
- **WHEN** el conv1d somero no aprueba la compuerta y el GRU sí
- **THEN** F2 y F3 se entrenan solo con GRU

### Requirement: Eliminación de modos

Tras cada fase, el sistema SHALL aplicar la regla 1.6 enmendada sobre `K` entrenado: partir de todos los bloques, quitar de a uno el de menor empeoramiento contra la configuración inicial mientras no sea significativo, detenerse en el primero significativamente peor y no quitar nunca el bloque diario. Los bloques finales SHALL fijarse antes del cierre.

#### Scenario: Diario protegido
- **WHEN** quitar el bloque diario no empeora el error a 24 h
- **THEN** el bloque diario se conserva

### Requirement: Cómputo en AWS

Entrenamiento, evaluación y compuertas de todas las fases SHALL correr en SageMaker Training Jobs CPU, uno por semilla y en serie, orquestados según `orquestacion-experimentos`. El cómputo local SHALL limitarse a tests unitarios con datos sintéticos.

#### Scenario: Decisión de compuerta
- **WHEN** termina el job de compuerta de una fase
- **THEN** escribe en S3 la rama elegida por las reglas, los ganadores por semilla y las cifras clave

### Requirement: Confirmación única en test

El modelo final SHALL reajustar `K` y `B` (y Duan si aplica) con datos hasta 2026-06-30, con encoder y decoder congelados y los bloques fijados, y evaluarse una sola vez en test contra LightGBM `lightgbm-v1`, persistencia y el test de GRU+DMD+Duan de 7.1, sobre filas comunes.

#### Scenario: Una sola evaluación
- **WHEN** ya existe un run de test del Koopman en esta versión
- **THEN** una nueva evaluación en test se rechaza salvo indicación explícita del operador

### Requirement: Interpretabilidad del Koopman

El change SHALL entregar para el modelo final: espectro con intervalos, formas de modo por estación y contaminante, descomposición de episodios por modo, estabilidad entre semillas y bootstrap, matriz `B`, y la coherencia entre DMD sobre el latente y el `K` aprendido.

#### Scenario: Coherencia registrada
- **WHEN** termina la evaluación de un modelo
- **THEN** su run contiene los autovalores de `K` y los del DMD exacto sobre su latente de train, emparejados
