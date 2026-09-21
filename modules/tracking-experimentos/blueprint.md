# Blueprint: tracking-experimentos

**Cuándo usar este blueprint**: cuando el proyecto entrena modelos y necesita
historial de experimentos (métricas, parámetros, artefactos) y eventualmente
un registro de modelos. Es una alternativa de **bajo costo** al MLflow
gestionado de SageMaker.

## Por qué MLflow self-hosted y no el gestionado

SageMaker ofrece MLflow gestionado, pero implica infraestructura que puede
quedar facturando (tracking server levantado). Para un proyecto personal, un
servidor MLflow self-hosted con **SQLite como backend, persistido en S3**, y
levantado solo cuando se lo necesita, logra lo mismo a costo marginal cero
(solo el S3 + el tiempo de cómputo esporádico que lo levanta).

## Arquitectura del patrón

```
      ┌─────────────────────────────────────────┐
      │  Fargate on-demand (se prende/apaga)     │
      │  contenedor: servidor mlflow --backend   │
      │  store-uri sqlite:///...                  │
      └──────────────┬──────────────────────────┘
                     │ 1. descarga mlruns + mlflow.sqlite de S3
                     │ 2. corre (logging de métricas desde tu código)
                     │ 3. sube los archivos de vuelta a S3
                     ▼
            s3://<bucket>/<prefijo>/_mlflow/
              ├── mlflow.sqlite      (base de datos de experimentos)
              └── mlruns/            (artefactos, modelos, métricas)
```

- El backend es un archivo **SQLite** (`mlflow.sqlite`), no una base de datos
  24/7.
- Los artefactos (modelos, plots, etc.) viven en `mlruns/` en el mismo S3.
- Un contenedor Fargate **descarga los archivos de S3 al arrancar**, levanta
  el servidor MLflow (o permite logging directo), y **los sube de vuelta al
  terminar**. No queda nada corriendo entre experimentos.

## Modos de uso

1. **Logging directo sin servidor**: la forma más liviana. Tu código de
   entrenamiento (Training Job o Fargate) importa `mlflow`, configura el
   `tracking_uri` a un archivo SQLite local montado en el trabajo, loguea
   métricas/parámetros/artefactos, y al terminar sube el `.sqlite` + `mlruns/`
   a S3. No hace falta levantar un servidor para loguear.
2. **Servidor interactivo (solo cuando querés ver la UI)**: levantar el
   contenedor Fargate que descarga de S3 y expone la UI de MLflow. Se usa
   para explorar, y se apaga al terminar.

Elegir el modo según necesidad: para entrenar de forma automatizada, modo 1;
para inspeccionar resultados a mano, modo 2 puntual.

## Consistencia (un dato importante del patrón)

SQLite como backend no maneja bien escrituras concurrentes desde varios
procesos a la vez. Reglas:

- **Una sola fuente de escritura** por corrida: no loguear desde varios jobs
  en paralelo contra el mismo `.sqlite` en S3 — cada corrida usa su copia y
  la sube; si hay corridas paralelas, serializar la escritura final.
- El archivo de verdad es la **copia en S3**; la local es efímera. Subir
  siempre al terminar, y descargar siempre al empezar, para no perder
  historia.

## Registro de modelos (independiente del tracking)

El historial de experimentos (tracking) y el registro de modelos son cosas
separadas. La decisión de **dónde registrar los modelos versionados** es por
proyecto (ver `cicd/blueprint.md` sección Model Registry):

- **SageMaker Model Registry**: nativo de SageMaker, integra con gate de
  aprobación y deploy; útil si el proyecto ya vive en SageMaker.
- **MLflow Model Registry (self-hosted)**: dentro del mismo servidor SQLite+S3,
  cero infra extra; útil si no se quiere acoplar a SageMaker.

El tracking self-hosted funciona igual en ambos casos: loguea métricas,
parámetros y artefactos; el registro decide dónde se versiona el modelo.

## IAM — mínimo privilegio

- El cómputo que hace logging necesita `s3:GetObject`/`PutObject`/`ListBucket`
  acotado **solo al prefijo** `_mlflow/` (descargar y subir), nada más.

## Parámetros a definir por proyecto (en `project-decisions.md`)

- Prefijo S3 del tracking (`<prefijo>/_mlflow/`).
- Modo de uso predominante (logging directo vs servidor interactivo).
- Decisión de registro: SageMaker Registry vs MLflow self-hosted.
- Qué se loguea además de métricas de modelo: ver `data-quality/blueprint.md`
  (métricas de calidad de datos por corrida).
