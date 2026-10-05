## Context

Los workflows `infra-cd.yml` y `pipeline-cd.yml` asumen roles AWS vía OIDC (`aws-actions/configure-aws-credentials@v4`). Desde el 15-jul-2026, GitHub usa un **formato inmutable del `sub`** para repos nuevos: `repo:<owner>@<owner-id>/<repo>@<repo-id>:<ref>`. Este repo (`desareca/forecast-sinca-aws`) se creó el 21-sep-2026, por lo que el `sub` real del token es:

```
repo:desareca@43764566/forecast-sinca-aws@1380017292:ref:refs/heads/main
```

El trust policy actual (en `infra/iam-oidc.tf`) usa el formato antiguo `repo:desareca/forecast-sinca-aws:*`, que no matchea → STS responde `Not authorized to perform sts:AssumeRoleWithWebIdentity`.

Los IDs (`43764566` owner, `1380017292` repo) son públicos (repo público) y no son secretos; se obtuvieron de la API de GitHub.

## Goals / Non-Goals

**Goals:**
- Que `sinca-github-infra-role` y `sinca-github-pipeline-role` vuelvan a ser asumibles por los workflows, matcheando el `sub` inmutable.
- Mantener el mínimo privilegio: el rol infra sigue permitiendo cualquier ref (plan en PR, apply gateado a main por `if:`), y el rol pipeline sigue restringido a `refs/heads/main`.

**Non-Goals:**
- No se cambia el alcance de permisos de los roles (solo la condición `sub`).
- No se migra el backend de Terraform ni se tocan los workflows YAML.
- No se aborda el `pipeline-cd.yml` skeleton desactualizado (eso es un change aparte).

## Decisions

### 1. Actualizar `local.github_repo` con el formato inmutable

En `infra/iam-oidc.tf`, cambiar:

```hcl
locals {
  github_repo = "desareca/forecast-sinca-aws"
  oidc_host   = "token.actions.githubusercontent.com"
}
```

por:

```hcl
locals {
  github_repo = "desareca@43764566/forecast-sinca-aws@1380017292"
  oidc_host   = "token.actions.githubusercontent.com"
}
```

Con esto, las dos trust policies quedan correctas sin tocar su estructura:

- **Rol infra** (`StringLike`): `repo:desareca@43764566/forecast-sinca-aws@1380017292:*` → matchea cualquier ref (push a main y PR).
- **Rol pipeline** (`StringEquals`): `repo:desareca@43764566/forecast-sinca-aws@1380017292:ref:refs/heads/main` → solo main.

**Alternativa considerada (opt-out al formato antiguo vía UI/API de GitHub)**: GitHub permite volver al formato anterior, pero es menos robusto (el formato inmutable es el default y el recomendado; además un rename/transfer futuro lo re-activaría). Descartada.

**Alternativa considerada (patrón `StringLike` con `@*`)**: `repo:desareca@*/forecast-sinca-aws@*:*` matchearía ambos formatos, pero afloja la restricción (aceptaría cualquier owner/repo ID). Descartada por seguridad: mejor el valor exacto.

### 2. Re-apply local (no vía CI)

El re-apply **no puede** hacerse vía GitHub Actions: el workflow no puede asumir el rol justamente porque el trust policy está mal (chicken-and-egg). Se aplica localmente con el profile `AWS_PROFILE` (`terraform -chdir=infra apply`), que ya tiene permisos IAM para actualizar el trust policy.

### 3. Verificación funcional

Tras el apply local, re-ejecutar `infra-cd` (y `pipeline-cd`) en GitHub y confirmar que `configure-aws-credentials` asume el rol sin error, y que `terraform plan` devuelve `No changes` (el state ya está alineado).

## Risks / Trade-offs

- **IDs hardcodeados en el `.tf`** → son públicos y estables (no cambian con rename/transfer, que es justamente la ventaja del formato inmutable). Si el repo se transfiere a otra org, el `sub` cambia y habría que actualizar el local — aceptable.
- **Re-apply local con drift** → el plan puede incluir el drift preexistente del dominio SageMaker (`studio_web_portal_settings {}`). No es destructivo; revisar el plan antes de aplicar.
- **`terraform.tfvars`/`backend.hcl` locales incompletos** → si se recrearon mal, el plan podría proponer destruir recursos. Verificar con `terraform plan` antes de aplicar (lección ya registrada en el blueprint cicd).

## Migration Plan

1. Editar `infra/iam-oidc.tf` (cambio del `local.github_repo`).
2. `terraform -chdir=infra plan` local (profile `AWS_PROFILE`) — revisar que solo cambie el trust policy (+ drift SageMaker conocido).
3. `terraform -chdir=infra apply` local.
4. Re-ejecutar `infra-cd` y `pipeline-cd` en GitHub y confirmar asunción exitosa.
5. Rollback: revertir el `local.github_repo` y re-aplicar (vuelve al estado roto anterior, sin pérdida de recursos).

## Open Questions

_(ninguna — el diagnóstico está confirmado con el `sub` real y los IDs públicos.)_
