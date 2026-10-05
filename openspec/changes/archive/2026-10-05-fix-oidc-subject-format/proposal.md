## Why

Los workflows `infra-cd.yml` y `pipeline-cd.yml` fallan al asumir los roles OIDC con `Not authorized to perform sts:AssumeRoleWithWebIdentity`. La causa es un cambio de GitHub: los repos creados después del 15-jul-2026 usan un **formato inmutable del `sub`** que incluye el owner ID y el repo ID (`repo:desareca@43764566/forecast-sinca-aws@1380017292:...`). Este repo se creó el 21-sep-2026, así que el `sub` real ya no matchea el trust policy actual (`repo:desareca/forecast-sinca-aws:*`).

## What Changes

- Actualizar `infra/iam-oidc.tf`: el `local.github_repo` pasa a incluir los IDs inmutables (`desareca@43764566/forecast-sinca-aws@1380017292`), de modo que las trust policies de `sinca-github-infra-role` y `sinca-github-pipeline-role` matcheen el `sub` real.
- Re-aplicar Terraform **localmente** (el re-apply no puede hacerse vía GitHub Actions: el workflow no puede asumir el rol justamente porque el trust policy está mal — chicken-and-egg).
- Verificar que `infra-cd` y `pipeline-cd` vuelven a asumir los roles correctamente.

## Capabilities

### New Capabilities

_(ninguna)_

### Modified Capabilities

- `cicd`: el requirement "Roles OIDC de mínimo privilegio" cambia el valor del `sub` de la trust policy al formato inmutable con IDs.
- `iam`: los requirements "Rol OIDC de infraestructura" y "Rol OIDC de pipeline" cambian el valor del `sub` de la trust policy al formato inmutable con IDs.

## Impact

- **Infraestructura AWS**: trust policies de `sinca-github-infra-role` y `sinca-github-pipeline-role` (cambio in-place, sin crear ni destruir recursos).
- **Repo**: `infra/iam-oidc.tf`.
- **Operación**: re-apply local con el profile `AWS_PROFILE` (no vía CI), luego re-ejecutar los workflows fallidos en GitHub.
- **Sin cambio de costo** ni de permisos (solo cambia la condición `sub` del trust).
