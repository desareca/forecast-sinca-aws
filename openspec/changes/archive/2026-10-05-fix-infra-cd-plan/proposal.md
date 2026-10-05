## Why

El `terraform plan` del workflow `infra-cd.yml` falla (y, peor, si llegara a correr el `apply` automático en `main`, aplicaría cambios destructivos) por dos causas preexistentes ajenas al fix del `sub` OIDC:

1. **Permiso IAM faltante**: el provider AWS v5.100.0, al refrescar el estado, llama `logs:ListTagsForResource` (API nueva) sobre los log groups, y `sinca-github-infra-role` solo tiene `logs:ListTagsLogGroup` (API vieja) → `AccessDeniedException`.
2. **El CI no tiene los valores reales de variables**: `infra-cd.yml` corre `terraform plan` sin `terraform.tfvars` (gitignored), usando los `default` de `variables.tf`. `user_profile_name` (default `sinca-dev-user` ≠ real `desareca-admin`) y `mlflow_allowed_cidr` (default `0.0.0.0/0` ≠ real IP del operador) hacen que el plan proponga **reemplazar** el user profile y **abrir el SG de MLflow al mundo**.

## What Changes

- Agregar a `sinca-github-infra-role` los permisos de logs nuevos (`logs:ListTagsForResource`, `logs:TagResource`, `logs:UntagResource`) que el provider AWS v5.x usa, reemplazando las APIs viejas deprecadas (`logs:ListTagsLogGroup`, `logs:TagLogGroup`, `logs:UntagLogGroup`).
- Agregar a `sinca-github-infra-role` un statement de EventBridge Scheduler de mínimo privilegio (`scheduler:GetSchedule`, `CreateSchedule`, `UpdateSchedule`, `DeleteSchedule`, `ListSchedules`, `TagResource`, `UntagResource`) sobre el schedule `sinca-scraper-daily`, que Terraform gestiona y el rol no cubría.
- Hacer que `infra-cd.yml` pase los valores reales de las variables que difieren de los defaults (`user_profile_name`, `mlflow_allowed_cidr`) vía variables de repo de GitHub Actions.
- Configurar esas dos variables en el repo (Settings → Variables).

## Capabilities

### New Capabilities

_(ninguna)_

### Modified Capabilities

- `iam`: el requirement "Rol OIDC de infraestructura" amplía sus permisos de CloudWatch Logs a las APIs nuevas de tags, y agrega permisos de EventBridge Scheduler, que el provider AWS v5.x requiere para el refresh del plan.
- `cicd`: el requirement "Workflows de CI/CD" pasa los valores reales de las variables que difieren de los defaults, para que el plan no proponga cambios destructivos.

## Impact

- **Infraestructura AWS**: policy inline de `sinca-github-infra-role` (cambio in-place, sin crear ni destruir recursos).
- **Repo**: `infra/iam-oidc.tf`, `.github/workflows/infra-cd.yml`.
- **Configuración GitHub**: dos variables de repo (`USER_PROFILE_NAME`, `MLFLOW_ALLOWED_CIDR`).
- **Operación**: re-apply local con el profile `AWS_PROFILE`, luego re-ejecutar `infra-cd` y confirmar `No changes`.
- **Sin cambio de costo**.
