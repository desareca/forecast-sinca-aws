## MODIFIED Requirements

### Requirement: Roles OIDC de mínimo privilegio

El sistema SHALL definir dos roles OIDC separados: uno de infraestructura (Terraform plan/apply) y uno de pipeline (actualizar definición y escribir artefactos), ambos con trust policy restringida al repo `desareca/forecast-sinca-aws` mediante el formato inmutable del `sub` (`repo:desareca@43764566/forecast-sinca-aws@1380017292:*`), y sin policies `*FullAccess` ni `AdministratorAccess`.

#### Scenario: Trust restringida al repo
- **WHEN** un workflow de un repo distinto a `desareca/forecast-sinca-aws` intenta asumir un rol OIDC
- **THEN** la asunción es denegada (la condición `sub` no coincide)

#### Scenario: Rol de infra permite plan/apply
- **WHEN** el workflow `infra-cd.yml` asume `sinca-github-infra-role`
- **THEN** puede ejecutar `terraform plan`/`apply` sobre los recursos gestionados, sin permisos de pipeline ni IAM amplios

#### Scenario: Rol de pipeline restringido a main
- **WHEN** un workflow intenta asumir `sinca-github-pipeline-role` desde una rama distinta a `main`
- **THEN** la asunción es denegada (la condición `sub` exige `refs/heads/main`)
