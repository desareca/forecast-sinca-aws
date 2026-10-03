# Blueprint: compute

**Cuándo usar este blueprint**: siempre que el proyecto necesite decidir dónde
corre el cómputo — ingesta, transformación, o entrenamiento. Es una **guía de
decisión**, no una receta fija: el objetivo es elegir el servicio según el
patrón de carga, no aplicar un default.

## Principio rector: que el cómputo se prenda y se apague solo

En proyectos personales el costo importa tanto como la corrección. La pregunta
de fondo antes de elegir es: **¿esto queda corriendo si no lo apago?** Preferir
siempre opciones on-demand (ver `costos/blueprint.md`) — el tipo de cómputo
debe alinearse con cuán esporádica y pesada es la carga.

## Árbol de decisión

```
¿Qué tipo de carga es?
│
├─ Liviana, event-driven, corta (<15 min), sin runtime pesado
│   → Lambda
│       - Sin VPC salvo que sea estrictamente necesario.
│       - Layers para dependencias pesadas (pandas/numpy) en vez de
│         empaquetarlas.
│       - Vigilar límite 250 MB + timeout 15 min.
│
├─ Entrenamiento de un modelo (batch, puede durar minutos a horas)
│   → SageMaker Training Job
│       - Se prende para entrenar y se apaga al terminar (paga solo el tiempo).
│       - Instancia según carga: CPU (ml.m5 / ml.t3) para tabular,
│         GPU (ml.g4dn / ml.g5) para PyTorch/TF.
│       - Artefactos de salida a S3, no se deja nada corriendo.
│
├─ Runtime grande sin ser ML (Playwright/browser, deps de sistema, >15 min,
│  proceso no-ml, o imagen con necesidades especiales)
│   → Fargate on-demand (tarea ECS)
│       - Solo se corre cuando se dispara, luego la tarea termina.
│       - Para iterar sin rebuild: patrón entrypoint genérico + código en S3
│         (la imagen lleva solo el runtime; el .py/.sh se baja y ejecuta).
│
└─ Notebooks interactivos de exploración (no producción)
    → SageMaker Studio Space (ver dev-environment/blueprint.md)
```

## Elegir la instancia / tamaño (no asumir)

- **Tabular (sklearn/XGBoost)**: arrancar por CPU (`ml.m5.large` de pruebas);
  subir solo si el dataset y la métrica lo justifican. XGBoost escala bien a
  CPU multi-core y muchas veces no necesita GPU.
- **PyTorch / TensorFlow**: si hay red neuronal, GPU desde el inicio
  (`ml.g4dn.xlarge` como punto de partida razonable), pero **medir** — un
  dataset chico puede entrenar en CPU sin pagar GPU.
- **Validar siempre con una corrida chica antes de escalar**: un subset del
  dataset, una epoch — el patrón de arranque es "corre chico y mide", no
  "lanza grande por si acaso".

## Patrón: entrypoint genérico + código en S3 (Fargate / Training Job)

Para iterar sobre lógica de negocio sin reconstruir la imagen cada vez:

- La imagen (ECR) contiene solo el **runtime**: imagen base + librerías (por
  framework, ej. `tensorflow/tensorflow`, `pytorch/pytorch`, o base Python +
  sklearn/xgboost/pandas).
- Un `entrypoint.py` genérico vive en la imagen y, al arrancar, descarga el
  script real desde S3 (`S3_BUCKET` + `SCRIPT_KEY` como env vars) y lo ejecuta.
- **Resultado**: cambiar la lógica = subir el `.py` a S3 y re-correr; no hay
  rebuild ni repush a ECR. Rebuild solo cuando cambian las dependencias del
  runtime.
- Si `S3_BUCKET`/`SCRIPT_KEY` faltan, el entrypoint SHALL fallar con error
  claro; si la descarga falla, SHALL propagar la excepción (no silenciar).
- El mismo patrón aplica a SageMaker Training Job: en vez de empaquetar el
  código en cada imagen, usar `SourceDir`/`entry_point` apuntando a un
  `pipeline/` versionado del repo (o a S3), con la imagen conteniendo solo el
  framework runtime.

## Cómputo híbrido (dos piezas del mismo pipeline)

Un pipeline puede combinar varios tipos: ej. Lambda para disparar + SageMaker
Training Job para entrenar + Fargate para el post-procesamiento. No forzar un
solo servicio para todo — elegir por etapa. Lo que sí debe ser consistente es
el almacenamiento intermedio (S3) y la orquestación (ver `cicd/blueprint.md`).

## IAM — mínimo privilegio

- Un rol de ejecución por tipo de cómputo, acotado a:
  - S3: solo los prefijos exactos que lee/escribe, nunca el bucket completo.
  - ECR (si Fargate): pull de imagen (`ecr:GetDownloadUrlForLayer`,
    `BatchGetImage`, `GetAuthorizationToken`).
  - Logs: `logs:CreateLogStream`/`PutLogEvents` acotado al log group propio.
  - Secrets Manager: `GetSecretValue` acotado al secret específico si aplica.
- Nunca managed policies amplias (`*FullAccess`). Nunca permisos IAM en el rol
  de cómputo.

## Parámetros a definir por proyecto (en `project-decisions.md`)

- Tipo(s) de cómputo por etapa del pipeline, con justificación.
- Framework y familia de instancia, y el criterio para escalarla.
- Nombre del repo ECR (si aplica) y estrategia de versionado de imagen.
- Prefijos S3 exactos a los que cada rol tiene acceso.

## Notas de este proyecto

- **Fargate exige dos roles**: un *rol de ejecución* (pull de ECR + logs) y un
  *rol de task* (lo que usa el contenedor, ej. S3). El rol de ejecución no se
  expone al contenedor y el task role no puede hacer pull de ECR; son dos roles
  separados, no uno.
- **Fargate necesita piezas de red/observabilidad**: además de la task
  definition, hacen falta un cluster ECS, un log group de CloudWatch, y un
  security group que habilite el puerto de la app (sin él `run-task` no tiene
  red y la UI no es alcanzable).
