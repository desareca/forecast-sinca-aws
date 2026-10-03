## Context

Proyecto `forecast-sinca-aws` (predicción ICAP zona saturada Temuco/PLC, horizonte 24h). Cuenta AWS personal (`AWS_ACCOUNT_ID`, profile `AWS_PROFILE`, región `us-east-1`). Repo GitHub `desareca/forecast-sinca-aws`.

El change anterior (`setup-infra-base`) dejó: buckets `sinca-data` y `sinca-mlflow`, SageMaker Studio domain + Space (`ml.t3.large`, EFS, SSH over SSM), roles IAM base (`sinca-dev-role`, `sinca-training-role`), y remote state de Terraform operativo. Este change cierra la Fase 1 (Setup) con tracking de experimentos y CI/CD.

Decisiones ya cerradas en `project-decisions.md`: tracking MLflow self-hosted (SQLite+S3, prefijo `sinca-mlflow/_mlflow/`), logging directo sin servidor como modo principal, OIDC con `sub` restringido al repo, feature branches por change, Terraform como código.

## Goals / Non-Goals

**Goals:**
- Servidor MLflow self-hosted on-demand (Fargate) para inspeccionar experimentos, con backend SQLite persistido en S3 y sin nada corriendo 24/7.
- Imagen MLflow custom versionada en ECR, reproducible vía Terraform.
- Autenticación GitHub Actions → AWS vía OIDC, con roles de mínimo privilegio.
- Los 3 workflows (`ci`, `infra-cd`, `pipeline-cd`) definidos y operativos.

**Non-Goals:**
- Pipeline de datos, scraper SINCA, ni lógica de negocio (Fases 2+).
- Model Registry (se reabre en Fase 6, según `project-decisions.md`).
- Servidor MLflow como service 24/7 (es on-demand).
- Secrets Manager (fuentes públicas HTTP, no aplica en esta fase).

## Decisions

### 1. Imagen MLflow custom (Dockerfile + entrypoint)

Se construye una imagen custom en vez de usar la imagen oficial directa. Motivo: la lógica de **descargar `mlflow.sqlite` de S3 al arrancar y subirlo de vuelta al apagar** (con `trap` de SIGTERM) es el punto frágil del patrón, y queda confiable y versionada dentro del entrypoint de la imagen.

- `Dockerfile`: `FROM ghcr.io/mlflow/mlflow` + `COPY entrypoint.sh` + `ENTRYPOINT ["/entrypoint.sh"]`.
- `entrypoint.sh`: (1) descarga `mlflow.sqlite` de `s3://sinca-mlflow/_mlflow/` (si existe), (2) arranca `mlflow server`, (3) `trap` de SIGTERM/SIGINT que sube el `.sqlite` de vuelta a S3 antes de salir.
- Alternativa considerada: imagen oficial + script en S3 (patrón `compute` "entrypoint genérico + código en S3") → descartada, agrega una pieza extra que coordinar y el `trap` queda fuera de la imagen.

### 2. Backend SQLite sincronizado + artefactos directo a S3

- `--backend-store-uri sqlite:///mlflow.sqlite`: el `.sqlite` (metadatos de experimentos/runs) se sincroniza S3 ↔ local (descarga al arrancar, sube al apagar).
- `--artifacts-destination s3://sinca-mlflow/_mlflow/mlruns`: los artefactos (modelos, plots) van **directo a S3**, sin sync de vuelta. Solo el `.sqlite` necesita el ciclo download/upload, lo que simplifica y evita mover artefactos grandes.
- Alternativa considerada: sincronizar también `mlruns/` (como el diagrama del blueprint) → descartada, mover artefactos grandes en cada ciclo es más lento y propenso a error; el destino S3 directo es el patrón soportado por MLflow.

### 3. Fargate on-demand (task, no service)

- Task definition Fargate (no un service ECS): se levanta con `run-task` y se detiene con `stop-task`, sin nada corriendo entre usos.
- Reutiliza el **default VPC** (mismo criterio que Studio en el change anterior: sin NAT gateway, sin recursos de red nuevos).
- Rol de ejecución de task con S3 acotado a `sinca-mlflow/_mlflow/*` (GetObject/PutObject/ListBucket) y `logs` en su log group propio.
- Script `infra/scripts/mlflow-server.sh` con subcomandos `up`/`down` que envuelven `aws ecs run-task`/`stop-task`.

### 4. OIDC — Identity Provider + 2 roles

- `aws_iam_openid_connect_provider` apuntando a `token.actions.githubusercontent.com`.
- **`sinca-github-infra-role`** (Terraform plan/apply): trust `sub` = `repo:desareca/forecast-sinca-aws:*` (cualquier ref). Motivo: `terraform plan` corre en PR (refs de pull request / feature branches), no solo en main. El `apply` se restringe a main vía `if: github.ref == 'refs/heads/main'` en el workflow.
- **`sinca-github-pipeline-role`** (pipeline/modelo): trust `sub` = `repo:desareca/forecast-sinca-aws:ref:refs/heads/main` (solo main). El `pipeline-cd.yml` solo corre al mergear a main.
- Esto **actualiza** `project-decisions.md` §9: la restricción "solo main" aplica al rol de pipeline; el rol de infra permite cualquier ref (con apply gateado por `if:`).

### 5. Tres workflows de GitHub Actions

- **`ci.yml`**: en PR, `terraform fmt -check` + `terraform validate` sobre `infra/`. No toca AWS. (Los tests de `pipeline/` se agregan cuando exista código Python, Fase 2+.)
- **`infra-cd.yml`**: en PR → `terraform plan` (asume `sinca-github-infra-role`); al mergear a main → `terraform apply` (mismo rol, gateado por `if:`).
- **`pipeline-cd.yml`**: skeleton. Al mergear cambios en `pipeline/` a main, actualizará la definición del pipeline (asume `sinca-github-pipeline-role`). Queda como no-op hasta que exista `pipeline/`.

## Risks / Trade-offs

- **`sub` amplio en el rol de infra (cualquier ref)** → mitigación: el `apply` está gateado por `if:` a main; el riesgo residual (una rama feature podría planear) es aceptable en un proyecto personal de un solo dev.
- **Pérdida de historia de tracking si el `.sqlite` no se sube al apagar** → mitigación: `trap` de SIGTERM/SIGINT en el entrypoint; el script `down` usa `stop-task` que dispara SIGTERM; verificación funcional confirma que el `.sqlite` vuelve a S3.
- **Concurrencia SQLite** (una sola fuente de escritura) → mitigación: el servidor interactivo es de un solo usuario a la vez; el logging directo (modo principal) usa su propia copia y la sube, sin escribir contra el servidor.
- **Fargate exige VPC** → mitigación: reutiliza el default VPC (sin NAT gateway, sin costo de red nuevo), igual que Studio.
- **Build de imagen manual** → mitigación: el build+push se documenta como tarea de ejecución; se puede automatizar en un workflow posterior si se vuelve recurrente.

## Migration Plan

1. Escribir Terraform (ECR, task Fargate, OIDC, roles) y la imagen (Dockerfile + entrypoint).
2. Build + push de la imagen a ECR.
3. `terraform plan` / `apply` (crea ECR, task, OIDC provider, roles).
4. Verificación funcional: levantar el servidor, ver la UI, apagarlo, confirmar el `.sqlite` en S3.
5. Rollback: `terraform destroy` de los recursos nuevos (no toca los buckets ni el Studio del change anterior).

## Open Questions

- Ninguna pendiente. La restricción `sub` del rol de infra (Opción A) quedó resuelta con el operador.
