# cicd Specification

## Purpose
TBD - created by archiving change setup-mlflow-ci-cd. Update Purpose after archive.
## Requirements
### Requirement: Autenticación OIDC de GitHub Actions

El sistema SHALL configurar un Identity Provider OIDC de AWS apuntando a `token.actions.githubusercontent.com` para que los workflows de GitHub Actions asuman roles de AWS sin Access Keys de larga duración.

#### Scenario: Identity Provider configurado
- **WHEN** se inspecciona la configuración de IAM de la cuenta
- **THEN** existe un `aws_iam_openid_connect_provider` con URL `token.actions.githubusercontent.com` y audiencia `sts.amazonaws.com`

### Requirement: Roles OIDC de mínimo privilegio

El sistema SHALL definir dos roles OIDC separados: uno de infraestructura (Terraform plan/apply) y uno de pipeline (actualizar definición y escribir artefactos), ambos con trust policy restringida al repo `desareca/forecast-sinca-aws` y sin policies `*FullAccess` ni `AdministratorAccess`.

#### Scenario: Trust restringida al repo
- **WHEN** un workflow de un repo distinto a `desareca/forecast-sinca-aws` intenta asumir un rol OIDC
- **THEN** la asunción es denegada (la condición `sub` no coincide)

#### Scenario: Rol de infra permite plan/apply
- **WHEN** el workflow `infra-cd.yml` asume `sinca-github-infra-role`
- **THEN** puede ejecutar `terraform plan`/`apply` sobre los recursos gestionados, sin permisos de pipeline ni IAM amplios

#### Scenario: Rol de pipeline restringido a main
- **WHEN** un workflow intenta asumir `sinca-github-pipeline-role` desde una rama distinta a `main`
- **THEN** la asunción es denegada (la condición `sub` exige `refs/heads/main`)

### Requirement: Workflows de CI/CD

El sistema SHALL definir tres workflows de GitHub Actions: `ci.yml` (lint/validate en PR, sin tocar AWS), `infra-cd.yml` (plan en PR / apply en main), y `pipeline-cd.yml` (actualizar definición del pipeline al mergear a main).

#### Scenario: CI en PR
- **WHEN** se abre un PR
- **THEN** `ci.yml` corre `terraform fmt -check` y `terraform validate` sin asumir ningún rol de AWS

#### Scenario: Plan en PR y apply en main
- **WHEN** se abre un PR con cambios en `infra/`
- **THEN** `infra-cd.yml` ejecuta `terraform plan`, y solo al mergear a `main` ejecuta `terraform apply`
