# Spec Delta

## Purpose

Ejecución en serie, en AWS, de una campaña de Training Jobs con lista fija de jobs, selección
preregistrada antes de test, reanudación por job, avisos y modo seco, sin aprobaciones intermedias
del operador.

## ADDED Requirements

### Requirement: Campaña en serie con lista fija

La campaña SHALL ejecutarse como una máquina de estados Standard de Step Functions que recorre una lista fija de jobs (entrenamiento → selección → cierre → test), lanza cada Training Job con integración síncrona y nunca más de uno a la vez. La máquina no SHALL ramificar según resultados ni esperar aprobaciones; las reglas de selección viven en el código de evaluación.

#### Scenario: Sin solape
- **WHEN** una fase tiene varios jobs
- **THEN** cada job empieza solo después de que el anterior terminó

#### Scenario: Sin pausas
- **WHEN** termina el job de selección
- **THEN** el cierre y el test se lanzan sin esperar intervención del operador

### Requirement: Selección antes de test

El job de test SHALL evaluar solo la configuración que el job de selección registró en S3 y MLflow, y SHALL fallar si ese registro no existe. La selección SHALL quedar registrada antes de que se lance el cierre.

#### Scenario: Selección ausente
- **WHEN** el job de test no encuentra el registro de selección de la campaña
- **THEN** termina con error sin evaluar en test

### Requirement: Salto de jobs completados

Antes de lanzar cada job, la máquina SHALL comprobar su marcador de éxito en S3, que el job escribe solo después de publicar su run con los artefactos requeridos, y SHALL saltarlo si existe. Los marcadores SHALL identificarse por campaña y clave estable del job, no por ejecución.

#### Scenario: Reanudar tras un fallo
- **WHEN** el job 8 de 12 de una fase falló y se inicia una nueva ejecución de la misma campaña
- **THEN** los jobs 1–7 no se lanzan otra vez y la ejecución empieza por el job 8

### Requirement: Checkpoints y retoma

Cada job de entrenamiento SHALL guardar al final de cada época un checkpoint (pesos, estado del optimizador, época y estado de RNG) en un prefijo de S3 por campaña y clave, y al iniciar SHALL retomar desde el último sin reiniciar calentamiento, learning rate ni currículo, de modo que el total de épocas sea el presupuesto fijo. Los checkpoints no se borran automáticamente.

#### Scenario: Retoma a mitad de job
- **WHEN** un job falla después de guardar la época 2 de 4 y se vuelve a lanzar con la misma clave
- **THEN** el nuevo job empieza en la época 3 y termina con 4 épocas en total

### Requirement: Avisos

El sistema SHALL avisar por email (tópico SNS) el inicio de la campaña, el fin de cada fase con un resumen breve, la configuración seleccionada con sus cifras clave, el cierre, el resultado de test y todo fin de ejecución `FAILED`, `TIMED_OUT` o `ABORTED`. El email del operador NO SHALL estar en el repo.

#### Scenario: Fallo de un job
- **WHEN** un Training Job de la campaña termina en `Failed`
- **THEN** la ejecución falla sin reintentar y llega un aviso de fallo

### Requirement: Reintentos acotados

La máquina SHALL reintentar a lo más 2 veces, con espera creciente, solo los errores transitorios de servicio de SageMaker (capacidad, límites, throttling), y no SHALL reintentar jobs que fallan por error de código.

#### Scenario: Error transitorio
- **WHEN** la creación de un job falla por límite de recursos
- **THEN** se reintenta con espera y, si vuelve a fallar dos veces, la ejecución falla y avisa

### Requirement: Modo seco

La campaña SHALL poder ejecutarse en modo seco con una semilla, donde los jobs no entrenan ni escriben en MLflow, para recorrer la lista fija completa (más selección, cierre y test) y, en la misma prueba, verificar la falla forzada a mitad de un job, la reanudación con salto por marcador, la retoma desde checkpoint y los avisos.

#### Scenario: Recorrido seco con falla y reanudación
- **WHEN** en modo seco con semilla 0 se fuerza un fallo tras la época 2 del job 3 de F1 y luego se lanza una nueva ejecución de la misma campaña
- **THEN** la primera ejecución termina `FAILED` sin reintento y con aviso; la segunda salta los jobs 1–2, el job 3 retoma en la época 3 y la campaña termina con éxito, con los avisos de inicio, fin de cada fase, selección, cierre y test, sin runs nuevos en MLflow
