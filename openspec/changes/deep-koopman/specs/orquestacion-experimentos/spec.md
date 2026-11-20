# Spec Delta

## Purpose

Ejecución en serie, en AWS, de una campaña de Training Jobs con compuertas, pausas con aprobación
del operador, avisos y modo seco, sin que el implementador espere en sesión.

## ADDED Requirements

### Requirement: Máquina de estados en serie

La campaña SHALL ejecutarse como una máquina de estados Standard de Step Functions que lanza cada Training Job con integración síncrona y nunca más de uno a la vez. La rama tras cada compuerta SHALL leerse de la decisión escrita en S3 por el job de compuerta; la orquestación no SHALL contener reglas de decisión propias.

#### Scenario: Sin solape
- **WHEN** una fase tiene varios jobs
- **THEN** cada job empieza solo después de que el anterior terminó

#### Scenario: Rama desconocida
- **WHEN** la decisión en S3 trae una rama fuera de la lista permitida
- **THEN** la ejecución falla y avisa, sin lanzar jobs nuevos

### Requirement: Salto de jobs completados

Antes de lanzar cada job o compuerta, la máquina SHALL comprobar su marcador de éxito en S3, que el job escribe solo después de publicar su run con los artefactos requeridos, y SHALL saltarlo si existe. Los marcadores SHALL identificarse por campaña y clave estable del job, no por ejecución.

#### Scenario: Reanudar tras un fallo
- **WHEN** el job 8 de 12 de una fase falló y se inicia una nueva ejecución de la misma campaña
- **THEN** los jobs 1–7 no se lanzan otra vez y la ejecución empieza por el job 8

### Requirement: Checkpoints y retoma

Cada job de entrenamiento SHALL guardar al final de cada época un checkpoint (pesos, estado del optimizador, época y estado de RNG) en un prefijo de S3 por campaña y clave, y al iniciar SHALL retomar desde el último sin reiniciar calentamiento, learning rate ni currículo, de modo que el total de épocas sea el presupuesto fijo. Los checkpoints no se borran automáticamente.

#### Scenario: Retoma a mitad de job
- **WHEN** un job falla después de guardar la época 2 de 4 y se vuelve a lanzar con la misma clave
- **THEN** el nuevo job empieza en la época 3 y termina con 4 épocas en total

### Requirement: Pausas con aprobación

La máquina SHALL detenerse con un token de tarea ante: rama de ambigüedad o cambio de diseño, compuerta final fallida y siempre antes de test, y SHALL continuar solo cuando el operador devuelve el token con la rama elegida por comando autenticado con su profile.

#### Scenario: Pausa antes de test
- **WHEN** termina el cierre
- **THEN** la máquina espera la aprobación del operador y no lanza el job de test sin ella

### Requirement: Avisos

El sistema SHALL avisar por email (tópico SNS) el resultado de cada compuerta con sus cifras clave, cada pausa con el comando de aprobación, el cierre, el resultado de test y todo fin de ejecución `FAILED`, `TIMED_OUT` o `ABORTED`. El email del operador NO SHALL estar en el repo.

#### Scenario: Fallo de un job
- **WHEN** un Training Job de la campaña termina en `Failed`
- **THEN** la ejecución falla sin reintentar y llega un aviso de fallo

### Requirement: Reintentos acotados

La máquina SHALL reintentar a lo más 2 veces, con espera creciente, solo los errores transitorios de servicio de SageMaker (capacidad, límites, throttling), y no SHALL reintentar jobs que fallan por error de código.

#### Scenario: Error transitorio
- **WHEN** la creación de un job falla por límite de recursos
- **THEN** se reintenta con espera y, si vuelve a fallar dos veces, la ejecución falla y avisa

### Requirement: Modo seco

La campaña SHALL poder ejecutarse en modo seco, donde los jobs no entrenan ni escriben en MLflow y las compuertas devuelven ramas forzadas por la entrada, para recorrer las ramas principales (pasa, B1, pausa y aprobación, pausa de test), la reanudación con salto de jobs y la retoma desde checkpoint, y verificar los avisos.

#### Scenario: Reanudación seca
- **WHEN** en modo seco se fuerza un fallo tras la época 2 del job 8 de F1 y luego se reanuda la campaña
- **THEN** la reanudación salta los jobs 1–7 y el job 8 retoma en la época 3

#### Scenario: Recorrido seco
- **WHEN** se ejecuta en modo seco con las ramas `b1` y `pausa` forzadas
- **THEN** la ejecución pasa por B1 y por las dos pausas, termina con éxito y deja un aviso por cada compuerta, pausa y cierre, sin runs nuevos en MLflow
