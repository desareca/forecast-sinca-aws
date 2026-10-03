## Why

El proyecto `forecast-sinca-aws` ya tiene la infraestructura base (buckets S3, SageMaker Studio Space, roles IAM de base) del change `setup-infra-base`. Falta cerrar la Fase 1 (Setup) con las dos piezas que habilitan el trabajo de ML de las fases siguientes: el **tracking de experimentos** (MLflow self-hosted) y el **CI/CD** (OIDC + workflows de GitHub Actions). Sin esto, no hay historial de experimentos ni automatización de infra/pipeline, y las Fases 2+ (pipeline de datos, baseline, experimentos) arrancarían sin base de tracking ni despliegue.

## What Changes

- **MLflow self-hosted**: repo ECR para la imagen, imagen custom (Dockerfile `FROM mlflow` + entrypoint que descarga `mlflow.sqlite` de S3 al arrancar, corre `mlflow server`, y lo sube de vuelta al apagar), y una task Fargate **on-demand** (no un service 24/7) para levantar la UI de MLflow solo cuando se la necesita.
- **OIDC de GitHub Actions**: Identity Provider (`token.actions.githubusercontent.com`) + 2 roles de mínimo privilegio (`sinca-github-infra-role` para Terraform plan/apply, `sinca-github-pipeline-role` para actualizar la definición del pipeline y escribir artefactos).
- **3 workflows de GitHub Actions**: `ci.yml` (terraform fmt/validate), `infra-cd.yml` (plan en PR / apply en main), `pipeline-cd.yml` (skeleton, apunta a `pipeline/` que aún no existe).
- **Script de operación** del servidor MLflow (prender/apagar la task Fargate).

No incluye (queda para Fases 2+): el pipeline de datos, el scraper SINCA, ni lógica de negocio. El `pipeline-cd.yml` queda como skeleton funcional hasta que exista `pipeline/`.

## Capabilities

### New Capabilities

- `tracking-experimentos`: tracking de experimentos MLflow self-hosted (SQLite+S3), servidor interactivo on-demand sobre Fargate, e imagen custom versionada en ECR.
- `cicd`: integración continua y despliegue vía GitHub Actions con autenticación OIDC a AWS (Identity Provider + roles de mínimo privilegio) y los 3 workflows del proyecto.

### Modified Capabilities

- `iam`: se agregan los roles OIDC de GitHub Actions (`sinca-github-infra-role`, `sinca-github-pipeline-role`) y el rol de ejecución de la task Fargate del servidor MLflow, todos con mínimo privilegio acotado por prefijo/servicio.

## Impact

- **Infraestructura AWS**: recursos nuevos en la cuenta personal (`AWS_ACCOUNT_ID`): repo ECR, task definition Fargate, Identity Provider OIDC, y 3 roles IAM nuevos. Región `us-east-1`. Reutiliza el default VPC (mismo criterio que Studio, sin NAT gateway).
- **Repo**: nuevas carpetas `infra/` (Terraform ampliado), `infra/scripts/` (script del servidor MLflow), `.github/workflows/` (3 workflows), y `docker/` (Dockerfile + entrypoint de la imagen MLflow).
- **Dependencia previa**: requiere el remote state ya operativo (del change anterior) y el bucket `sinca-mlflow` ya creado.
- **Costo mensual estimado**: Fargate on-demand (solo cuando se levanta la UI, ~$0.04/h de vCPU+mem mínima), ECR (despreciable a este volumen), S3 (despreciable). Total por debajo de USD 5/mes; nada queda corriendo 24/7.
