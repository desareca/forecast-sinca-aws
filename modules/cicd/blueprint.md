# Blueprint: cicd

**Cuándo usar este blueprint**: siempre (transversal). Define buenas prácticas
de CI/CD **desde el inicio** del proyecto — no como algo a agregar después. Es
el blueprint más denso y el que fija las reglas de IaC, testing, despliegue y
autenticación.

## 1. Estructura de repo (separación clara)

```
<repo>/
├── infra/                # Terraform (IAM, SageMaker, buckets, red, registry)
├── pipeline/             # scripts .py del pipeline ML — NUNCA notebooks en producción
├── tests/                # unit + integration
├── notebooks/            # SOLO exploración (no se ejecutan en producción)
├── .github/workflows/    # ci.yml, infra-cd.yml, pipeline-cd.yml
└── project-decisions.md, PREREQUISITES.md, ... (docs de la plantilla)
```

Reglas:
- **Los notebooks van en `notebooks/` y son solo exploración.** El código que
  corre en producción vive en `pipeline/` como `.py` versionable y testeable.
  Nunca se despliega un `.ipynb` al pipeline.
- `infra/` y `pipeline/` son codebases separados en preocupación pero
  versionados juntos en el mismo repo.

## 2. Separación dev/prod desde el día uno

- **Nunca** una sola versión "de producción" sin ambiente de prueba.
- Opción A: **SageMaker Pipelines** con ambientes separados (pipelines y
  buckets distintos para `dev` y `prod`).
- Opción B: parametrización por ambiente (variables/env para bucket, IAM,
  nombres) aplicada al mismo pipeline.

Cualquiera sea el mecanismo, `dev` existe antes de que exista `prod`, y el
flujo de promoción es explícito (ver Model Registry abajo). No se "pasa a
producción" editando el código en caliente.

## 3. Todo como código vía Terraform

- Todo es IaC: dominio SageMaker, Space, Lifecycle Config, buckets, Model
  Registry, roles IAM, OIDC, etc. **Nada se configura a mano en consola.**
- **Remote state desde el inicio** (regla fija de esta plantilla):
  - Backend S3 para el state (`terraform { backend "s3" {...} }`).
  - DynamoDB para locking (evita applies concurrentes).
  - No usar `local` state ni siquiera para arrancar — el bucket + tabla se
    crean como parte del primer setup de la cuenta (ver `PREREQUISITES.md`).

## 4. Pirámide de testing

```
        /\
       /  \        E2E (sparse, opcional)
      /----\       Integration: dataset sintético mínimo en dev
     /------\      Validación de datos (data-quality) — paso obligatorio
    /--------\     Unit tests (pytest) sobre funciones puras
   /----------\
```

- **Unit (pytest)**: sobre funciones puras (transformaciones, features,
  métricas). No requiere AWS.
- **Integration**: con un **dataset sintético mínimo** en el ambiente `dev`,
  verificando que el pipeline corre de punta a punta sin tocar producción.
- **Validación de datos**: ver `data-quality/blueprint.md`. Es paso
  **obligatorio antes de entrenar**, dentro del pipeline (no solo en CI).

## 5. Tres workflows de GitHub Actions

### `ci.yml` — lint + tests en cada PR
Corre en cada PR: lint (ruff/flake8), formatter check, y `pytest`. No toca
AWS ni despliega nada. Bloquea el merge si falla.

### `infra-cd.yml` — infraestructura (plan/apply)
- **En PR**: `terraform plan` (no apply).
- **Al mergear a `main`**: `terraform apply`.
- El plan debe ser revisable antes del merge.

### `pipeline-cd.yml` — actualiza la definición del pipeline ML
- **Al mergear cambios en `pipeline/`**: actualiza la **definición** de la
  SageMaker Pipeline (sube el código versionado), **sin ejecutarla**.
- **La ejecución real NO la dispara un push de código.** La dispara un
  **EventBridge Schedule** por separado (o ejecución manual). Separar
  "actualizar la definición" de "disparar la corrida" evita corridas
  accidentales en cada push.

## 6. Model Registry con gate MANUAL al inicio

- Cada corrida que produce un modelo registra una **versión candidata** en el
  Model Registry (SageMaker Registry o MLflow self-hosted — ver
  `tracking-experimentos/blueprint.md`, es decisión por proyecto) con sus
  **métricas adjuntas**.
- Las métricas incluyen **tanto métricas de modelo como métricas de calidad
  de datos** (ver `data-quality/blueprint.md`), no solo accuracy/loss.
- **Gate de aprobación MANUAL al inicio**: cada versión entra como
  `PendingManualApproval`; el operador la aprueba/rechaza a mano revisando las
  métricas.
- **Automatización recién después de varios ciclos de confianza**: solo cuando
  el proceso demuestra ser fiable, se reemplaza el gate manual por un
  **umbral automático** de promoción. No se automatiza de entrada.

## 7. Autenticación GitHub Actions → AWS vía OIDC

Sin Access Keys de larga duración en GitHub Secrets. Se usa OpenID Connect:

- **Terraform — Identity Provider**:
  `aws_iam_openid_connect_provider` apuntando a
  `token.actions.githubusercontent.com`.
- **Terraform — rol con trust policy restringida a `sub`**: la condición
  restringe `sub` al repo y rama exactos (ej.
  `repo:<github-user-or-org>/<nombre-repo>:ref:refs/heads/main`), **nunca abierto a cualquier
  repo**. Ejemplo del bloque de condición:

  ```hcl
  condition {
    test     = "StringLike"
    variable = "token.actions.githubusercontent.com:sub"
    values   = ["repo:<github-user-or-org>/<nombre-repo>:ref:refs/heads/main"]
  }
  ```

- **Dos roles separados de mínimo privilegio** (nunca `AdministratorAccess`):
  - **Rol de infra/Terraform**: permisos para `plan`/`apply` de los recursos
    que Terraform gestiona (IAM, S3, SageMaker, etc.), acotados por prefijo y
    servicio.
  - **Rol de pipeline de modelo**: permisos para actualizar la definición del
    pipeline, escribir artefactos en S3, registrar modelos — sin permisos de
    infraestructura de red/IAM.
- **Workflow YAML**:
  - `permissions: { id-token: write, contents: read }`.
  - Paso `aws-actions/configure-aws-credentials@v4` pasando `role-to-assume`
    (el ARN del rol correspondiente al workflow) y `aws-region`.

```yaml
permissions:
  id-token: write
  contents: read

steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::<account>:role/<rol-infra-o-pipeline>
      aws-region: <region>
```

- **Se repite una vez POR CUENTA AWS** (dev y prod si están en cuentas
  separadas): cada cuenta tiene su propio Identity Provider y sus propios
  roles, de modo que un workflow **nunca puede cruzar accidentalmente a otra
  cuenta** (ver `PREREQUISITES.md` — profile ajeno prohibido).

## 8. Secretos

- Si el proyecto tiene secretos, van en **AWS Secrets Manager**, nunca en
  `.env` commiteado ni en GitHub Secrets (que es para OIDC, no para guardar
  credenciales de aplicación).
- `.env` local está gitignored y sirve solo para desarrollo no-AWS (ver
  `AGENTS.md` regla de credenciales).

## 9. Observabilidad desde el inicio

- **CloudWatch Alarms** sobre fallos de Training Job / pipeline (job falló,
  pasó el timeout, error de infra).
- **Logging estructurado de cada corrida**: versión de datos, métricas de
  calidad, métricas de modelo, y la decisión de Registry — todo hacia el
  MLflow self-hosted (ver `tracking-experimentos/blueprint.md`). Así cada
  corrida deja un registro completo y comparable.
- No spamear alertas: reservar email para condiciones que necesitan acción
  humana (un Training Job que falla), no para todo evento (ver el principio
  de alerta selectiva; `data-quality/blueprint.md` complementa).

## Parámetros a definir por proyecto (en `project-decisions.md`)

- Estrategia de ramas y nombres de ambientes dev/prod.
- Decisión de Model Registry (SageMaker vs MLflow) y umbral de auto-promoción
  (cuando se llegue a él).
- Nombre del repo en el `sub` de OIDC y los ARN de los dos roles.
- Región, bucket de terraform state, y tabla DynamoDB.
