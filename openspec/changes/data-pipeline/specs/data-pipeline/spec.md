## ADDED Requirements

### Requirement: Ingesta de las 3 estaciones SINCA

El sistema SHALL ingerir datos horarios de calidad del aire y meteorología de las 3 estaciones activas de la zona saturada (Padre Las Casas II ID 263, Ñielol, Las Encinas) mediante scraping del endpoint CSV de SINCA (`apub.tsindico2.cgi`, con `macropath`/`macro` descubierto desde la página de la estación), parametrizado por ID de estación.

#### Scenario: Ingesta parametrizada por estación
- **WHEN** el scraper se ejecuta con los IDs de las 3 estaciones
- **THEN** obtiene series horarias de MP2.5, MP10 y meteorología para cada estación

#### Scenario: Modo backfill e incremental
- **WHEN** el scraper se ejecuta con un rango de fechas (backfill) o sin rango (incremental)
- **THEN** ingiere el histórico solicitado o las últimas 24-48 horas, respectivamente

### Requirement: Ingesta de variables auxiliares

El sistema SHALL ingerir la altura de capa límite desde Open-Meteo (JSON, horaria) y los feriados desde Nager.Date API (`date.nager.at/api/v3/PublicHolidays/{year}/CL`, una vez por corrida, filtrando feriados regionales por el campo `counties`), y SHALL calcular el día de semana y la ventana GEC (1 abr–15 sep) sin fuente externa.

#### Scenario: Variables auxiliares disponibles
- **WHEN** se ejecuta una corrida del pipeline
- **THEN** quedan disponibles la altura de capa límite horaria, los feriados, el día de semana y la ventana GEC

### Requirement: Cálculo de ICAP

El sistema SHALL calcular el ICAP por estación aplicando la fórmula oficial piecewise-lineal del D.S. 12/2011 (y su equivalente MP2.5) al promedio móvil de 24h de MP10/MP2.5, con 3 anclas por contaminante (MP10: 0→0, 100→150, 500→330 µg/m³; MP2.5: 0→0, 100→50, 500→170 µg/m³) e interpolación lineal entre anclas, y SHALL derivar el `ICAP_zona` en post-proceso como el máximo entre las 3 estaciones y el peor contaminante.

#### Scenario: ICAP por estación
- **WHEN** se dispone de las series horarias de MP2.5/MP10 de una estación
- **THEN** se calcula el ICAP de esa estación con la fórmula oficial

#### Scenario: ICAP_zona derivado
- **WHEN** se calculan los ICAP de las 3 estaciones
- **THEN** el `ICAP_zona` es el máximo entre las 3 estaciones, tomando el peor contaminante

### Requirement: Validación de calidad con pandera

El sistema SHALL validar los datos ingeridos con pandera antes de persistir, cubriendo validez (tipos y rangos), completitud (% de nulos), unicidad (sin duplicados `estación,timestamp`), oportunidad (frescura del último timestamp) y consistencia (unidades/encoding).

#### Scenario: Datos que pasan la validación
- **WHEN** un lote ingerido cumple el schema y los rangos
- **THEN** se persiste en `validated/`

#### Scenario: Datos rechazados
- **WHEN** un lote falla una regla de validación
- **THEN** se mueve a `quarantine/` con el motivo de rechazo adjunto (dimensión y regla que falló)

### Requirement: Persistencia en Parquet con convención de prefijos

El sistema SHALL persistir los datos en Parquet bajo la convención `sinca-data/{raw,validated,quarantine}/`, con particionado por dataset/estación y temporal (año/mes).

#### Scenario: Ciclo raw → validated
- **WHEN** el pipeline ingiere y valida un lote
- **THEN** los datos crudos quedan en `raw/` y los validados en `validated/`, en Parquet particionado

### Requirement: Schedule de ejecución diaria

El sistema SHALL disponer de una regla de EventBridge Schedule que dispare la task Fargate del scraper diariamente (~01:00 hora local), tras un smoke test manual exitoso.

#### Scenario: Corrida diaria automática
- **WHEN** se alcanza la hora programada
- **THEN** la task Fargate del scraper se ejecuta y persiste los datos del día anterior
