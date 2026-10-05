## MODIFIED Requirements

### Requirement: Workflows de CI/CD

El sistema SHALL definir tres workflows de GitHub Actions: `ci.yml` (lint/validate en PR, sin tocar AWS), `infra-cd.yml` (plan en PR / apply en main), y `pipeline-cd.yml` (actualizar definición del pipeline al mergear a main).

#### Scenario: CI en PR
- **WHEN** se abre un PR
- **THEN** `ci.yml` corre `terraform fmt -check` y `terraform validate` sin asumir ningún rol de AWS

#### Scenario: Plan en PR y apply en main
- **WHEN** se abre un PR con cambios en `infra/`
- **THEN** `infra-cd.yml` ejecuta `terraform plan`, y solo al mergear a `main` ejecuta `terraform apply`

#### Scenario: Variables reales en el plan
- **WHEN** `infra-cd.yml` ejecuta `terraform plan`/`apply`
- **THEN** pasa los valores reales de las variables que difieren de los defaults (`user_profile_name`, `mlflow_allowed_cidr`) vía variables de repo, para que el plan no proponga cambios destructivos
