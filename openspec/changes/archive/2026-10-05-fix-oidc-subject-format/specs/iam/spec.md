## MODIFIED Requirements

### Requirement: Rol OIDC de infraestructura

El sistema SHALL definir un rol IAM `sinca-github-infra-role` asumible por GitHub Actions vía OIDC, con permisos de mínimo privilegio para ejecutar `terraform plan`/`apply` sobre los recursos que Terraform gestiona, y trust policy restringida al repo `desareca/forecast-sinca-aws` mediante el formato inmutable del `sub` (`repo:desareca@43764566/forecast-sinca-aws@1380017292:*`).

#### Scenario: Asunción desde el repo correcto
- **WHEN** un workflow del repo `desareca/forecast-sinca-aws` asume `sinca-github-infra-role`
- **THEN** la asunción es permitida y el rol puede operar sobre los recursos gestionados por Terraform

#### Scenario: Sin políticas amplias
- **WHEN** se inspecciona la política de `sinca-github-infra-role`
- **THEN** no contiene `AdministratorAccess` ni managed policies `*FullAccess`

### Requirement: Rol OIDC de pipeline

El sistema SHALL definir un rol IAM `sinca-github-pipeline-role` asumible por GitHub Actions vía OIDC, con permisos para actualizar la definición del pipeline y escribir artefactos, y trust policy restringida a `refs/heads/main` del repo `desareca/forecast-sinca-aws` mediante el formato inmutable del `sub` (`repo:desareca@43764566/forecast-sinca-aws@1380017292:ref:refs/heads/main`).

#### Scenario: Asunción solo desde main
- **WHEN** un workflow intenta asumir `sinca-github-pipeline-role` desde una rama distinta a `main`
- **THEN** la asunción es denegada

#### Scenario: Permisos acotados a artefactos y pipeline
- **WHEN** `sinca-github-pipeline-role` opera
- **THEN** solo puede actualizar la definición del pipeline y escribir artefactos en los prefijos acotados, sin permisos de infraestructura de red ni IAM
