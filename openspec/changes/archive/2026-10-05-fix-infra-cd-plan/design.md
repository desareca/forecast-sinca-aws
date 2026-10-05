## Context

Tras el fix del `sub` OIDC (`fix-oidc-subject-format`), los roles vuelven a ser asumibles, pero el `terraform plan` del workflow `infra-cd.yml` sigue fallando por dos causas preexistentes:

1. El provider AWS v5.100.0 migró de las APIs viejas de tags de CloudWatch Logs (`ListTagsLogGroup`, `TagLogGroup`, `UntagLogGroup`) a las nuevas (`ListTagsForResource`, `TagResource`, `UntagResource`). `sinca-github-infra-role` solo tiene las viejas → `AccessDeniedException` al refrescar.
2. `infra-cd.yml` corre `terraform plan -var="aws_profile="` sin `terraform.tfvars` (gitignored), usando los `default` de `variables.tf`. Dos variables difieren del estado real y producen diffs destructivos.

## Goals / Non-Goals

**Goals:**
- Que `terraform plan` del workflow `infra-cd.yml` corra sin `AccessDeniedException`.
- Que el plan no proponga cambios destructivos (reemplazar el user profile, abrir el SG de MLflow).
- Que el `apply` automático en `main` sea seguro (no-op cuando el estado está alineado).

**Non-Goals:**
- No se cambia la estrategia de `apply` automático en `main` (sigue gateado por `if: github.ref == 'refs/heads/main'`).
- No se migra el backend ni se tocan los otros workflows (`ci.yml`, `pipeline-cd.yml`).
- No se aborda el `pipeline-cd.yml` skeleton desactualizado (change aparte).

## Decisions

### 1. Reemplazar las APIs viejas de tags de logs por las nuevas

En `infra/iam-oidc.tf`, statement `CloudWatchLogs` de `github_infra_policy`, reemplazar:

```hcl
"logs:TagLogGroup",
"logs:UntagLogGroup",
"logs:ListTagsLogGroup",
```

por:

```hcl
"logs:TagResource",
"logs:UntagResource",
"logs:ListTagsForResource",
```

**Alternativa considerada (agregar las nuevas sin quitar las viejas)**: mantiene ambas, pero deja permisos deprecados que el provider ya no usa → viola mínimo privilegio. Descartada.

**Alternativa considerada (solo `logs:ListTagsForResource`)**: el error reportado es solo ese, pero el provider también usa `TagResource`/`UntagResource` cuando hay tags que aplicar. Agregar las tres es lo correcto y evita un segundo round-trip.

### 2. Permiso de EventBridge Scheduler

`github_infra_policy` no tenía ningún permiso de EventBridge Scheduler, y Terraform gestiona `aws_scheduler_schedule.scraper_daily` (`infra/schedule.tf`). El provider AWS, al refrescar, llama `scheduler:GetSchedule` → `AccessDeniedException`. Se agrega un statement `Scheduler` de mínimo privilegio:

```hcl
statement {
  sid    = "Scheduler"
  effect = "Allow"
  actions = [
    "scheduler:GetSchedule",
    "scheduler:CreateSchedule",
    "scheduler:UpdateSchedule",
    "scheduler:DeleteSchedule",
    "scheduler:TagResource",
    "scheduler:UntagResource",
  ]
  resources = [
    "arn:aws:scheduler:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:schedule/default/sinca-scraper-daily",
  ]
}

statement {
  sid    = "SchedulerList"
  effect = "Allow"
  actions = ["scheduler:ListSchedules"]
  resources = ["*"]
}
```

`ListSchedules` es una acción de listado no restringible por recurso (mismo patrón que `s3:ListAllMyBuckets`), por eso va con `resources = ["*"]`.

**Alternativa considerada (un solo statement con `resources = ["*"]`)**: más simple, pero afloja el mínimo privilegio (daría CRUD de Scheduler sobre cualquier schedule). Descartada; se acota el CRUD al ARN del schedule y solo el listado queda `*`.

### 3. Pasar las variables reales vía variables de repo

En `.github/workflows/infra-cd.yml`, los steps `plan` y `apply` pasan las dos variables que difieren de los defaults:

```yaml
-var="user_profile_name=${{ vars.USER_PROFILE_NAME }}" \
-var="mlflow_allowed_cidr=${{ vars.MLFLOW_ALLOWED_CIDR }}"
```

Y se configuran en el repo (Settings → Variables): `USER_PROFILE_NAME=desareca-admin`, `MLFLOW_ALLOWED_CIDR=<ip-del-operador>/32`.

**Alternativa considerada (cambiar los defaults en `variables.tf`)**: haría que el default coincida con el valor real, pero los defaults deben ser genéricos (no específicos del operador), y `mlflow_allowed_cidr` es un valor por-operador. Descartada.

**Alternativa considerada (subir `terraform.tfvars` como Secret)**: el `.tfvars` completo es frágil (cambia con cada variable nueva) y mezcla valores sensibles con no sensibles. Descartada; se pasan solo las dos variables que difieren.

### 4. Orden de aplicación (riesgo de destrucción)

El `apply` automático en `main` es peligroso si el plan tiene diffs destructivos. Por eso el orden es:

1. Aplicar **ambos** fixes (permiso + variables) en el mismo commit.
2. `terraform apply` **local** (profile `AWS_PROFILE`) para actualizar la policy del rol.
3. Configurar las variables de repo.
4. Recién ahí pushear a `main` y verificar que el plan del CI da `No changes`.

Si se aplicara solo el permiso (sin las variables), el plan correría y el `apply` de `main` **destruiría** el user profile y abriría el SG. Por eso ambos fixes van juntos.

## Risks / Trade-offs

- **`mlflow_allowed_cidr` en una variable de repo** → es la IP del operador (semi-sensible). Las variables de Actions no son visibles públicamente (solo colaboradores), y el valor ya está en el state de Terraform. Aceptable.
- **Variables de repo faltantes** → si `USER_PROFILE_NAME`/`MLFLOW_ALLOWED_CIDR` no están configuradas, el plan vuelve a usar defaults y propone diffs destructivos. Mitigación: la tarea de verificación confirma que el plan da `No changes` antes de dar el change por cerrado.
- **Drift perpetuo de SageMaker** → el plan siempre muestra el update in-place de `studio_web_portal_settings {}` (bloque vacío en el state, no declarado en el config; el provider lo re-propone en cada plan incluso tras cada apply). No destructivo, pero hace que el plan nunca sea `No changes` ni el apply `0/0/0`. Aceptable para este change; eliminar el drift (agregar el bloque al config o `ignore_changes`) es un change aparte.

## Migration Plan

1. Editar `infra/iam-oidc.tf` (permisos de logs) y `.github/workflows/infra-cd.yml` (variables).
2. `terraform -chdir=infra plan` local (profile `AWS_PROFILE`) — revisar que solo cambie la policy del rol (+ drift SageMaker).
3. `terraform -chdir=infra apply` local.
4. Configurar `USER_PROFILE_NAME` y `MLFLOW_ALLOWED_CIDR` en Settings → Variables.
5. Commitear y pushear a `main`; re-ejecutar `infra-cd` y confirmar `No changes`.
6. Rollback: revertir el commit y re-aplicar localmente (vuelve al estado anterior, sin pérdida de recursos).

## Open Questions

_(ninguna — el diagnóstico está confirmado con el error real del provider y los diffs del plan.)_
