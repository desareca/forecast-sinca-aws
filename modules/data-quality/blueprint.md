# Blueprint: data-quality

**Cuándo usar este blueprint**: cuando el proyecto ingiere datos o entrena
modelos sobre datos ingeridos (cubre ETL simple y MLOps). Este blueprint es
independiente y está **referenciado desde `cicd/blueprint.md`** como el paso de
validación del pipeline. La validación de calidad **no es opcional ni solo
informativa**: puede detener el pipeline si la calidad cae bajo un umbral.

## Las 6 dimensiones de calidad (checklist de diseño)

Al definir cualquier ingesta nueva, revisar estas seis dimensiones como
checklist de diseño — no hace falta "implementar" una por una, pero sí decidir
conscientemente cuáles aplican y cómo se miden:

| Dimensión | Pregunta base | Ejemplo |
|---|---|---|
| **Completitud** | ¿Faltan valores/registros? | % nulos por columna; registros esperados vs. recibidos |
| **Exactitud** | ¿Los valores son correctos? | rango plausible de una métrica, discrepancias vs. fuente |
| **Consistencia** | ¿Los valores son coherentes entre sí? | misma unidad/encoding en todos lados; FK válidas |
| **Validez** | ¿Los valores cumplen el formato/tipo? | tipos, rango de fechas, enums permitidos |
| **Unicidad** | ¿Hay duplicados no esperados? | clave natural única, sin filas repetidas |
| **Oportunidad** | ¿Los datos están frescos / a tiempo? | retraso máximo tolerable entre origen y disponibilidad |

## Herramientas

- **pandera (default)**: para validación de schema/rangos en Python. Se
  definen `DataFrameSchema`/`SchemaModel` tipados y se valida el frame antes de
  escribir/entrenar. Es la herramienta por defecto de esta plantilla.
- **AWS Glue Data Quality (basado en Deequ)**: alternativa cuando el proyecto
  ya usa Glue/Athena y la validación ocurre en ese stack. No justifica
  introducir Glue solo por la validación — usar pandera salvo que Glue ya esté
  presente.

## Convención de cuarentena en S3

Convención de carpetas por cada dataset ingerido:

```
<prefijo>/<dataset>/
├── raw/         # datos tal como llegan del origen (sin tocar)
├── validated/   # datos que pasaron la validación (listos para usar)
└── quarantine/  # datos rechazados, con motivo de rechazo adjunto
```

- Todo lo que entra cae en `raw/` primero.
- La validación lee `raw/`, y mueve lo que aprueba a `validated/` y lo que no
  a `quarantine/`.
- Cada lote en `quarantine/` lleva **adjunto el motivo de rechazo** (un
  archivo de reporte, o un prefijo/metadata con la dimensión y la regla que
  falló). No basta con separar — hay que saber *por qué* se rechazó para poder
  remediar.

## La validación como paso de pipeline (referencia desde cicd)

El paso de validación es parte de la **SageMaker Pipeline** (ver
`cicd/blueprint.md`) y:

- Se ejecuta **antes de entrenar**, sobre los datos `validated/` (o como el
  gate que decide si `raw/` pasa a `validated/`).
- Tiene un **umbral definido**: si la calidad cae por debajo (ej. completitud
  de una columna crítica < 99%, o tasa de cuarentena > x%), **detiene el
  pipeline**. No sigue como si nada.
- El umbral y las dimensiones críticas se definen en `project-decisions.md`,
  no se improvisan durante la implementación.

## Logging de métricas de calidad

Las métricas de calidad (completitud por columna, tasa de cuarentena, checks
que fallaron) se loguean en el **mismo MLflow self-hosted** (ver
`tracking-experimentos/blueprint.md`), junto a las métricas del modelo — para
tener historial de calidad de datos **por corrida**, no solo el resultado del
modelo. Esto permite correlacionar después "el modelo empeoró" con "la calidad
de los datos bajó esa corrida".

## Parámetros a definir por proyecto (en `project-decisions.md`)

- Cuáles de las 6 dimensiones aplican y sus reglas concretas.
- Tooling (pandera por defecto; Glue Data Quality si ya hay Glue).
- Umbral de detención del pipeline y dimensiones críticas.
- Estructura exacta de `raw/`/`validated/`/`quarantine/` y formato del motivo
  de rechazo.
