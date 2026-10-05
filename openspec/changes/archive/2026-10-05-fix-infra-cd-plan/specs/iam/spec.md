## MODIFIED Requirements

### Requirement: Rol OIDC de infraestructura

El sistema SHALL definir un rol IAM `sinca-github-infra-role` asumible por GitHub Actions vía OIDC, con permisos de mínimo privilegio para ejecutar `terraform plan`/`apply` sobre los recursos que Terraform gestiona, y trust policy restringida al repo `desareca/forecast-sinca-aws` mediante el formato inmutable del `sub` (`repo:desareca@43764566/forecast-sinca-aws@1380017292:*`).

#### Scenario: Asunción desde el repo correcto
- **WHEN** un workflow del repo `desareca/forecast-sinca-aws` asume `sinca-github-infra-role`
- **THEN** la asunción es permitida y el rol puede operar sobre los recursos gestionados por Terraform

#### Scenario: Sin políticas amplias
- **WHEN** se inspecciona la política de `sinca-github-infra-role`
- **THEN** no contiene `AdministratorAccess` ni managed policies `*FullAccess`

#### Scenario: Permisos de logs para el refresh del plan
- **WHEN** el provider AWS refresca el estado durante `terraform plan`
- **THEN** el rol puede llamar `logs:ListTagsForResource` (y las APIs nuevas de tags) sobre los log groups gestionados

#### Scenario: Permisos de Scheduler para el refresh del plan
- **WHEN** el provider AWS refresca el estado durante `terraform plan`
- **THEN** el rol puede llamar `scheduler:GetSchedule` (y las acciones de CRUD/tags) sobre el schedule `sinca-scraper-daily` gestionado
