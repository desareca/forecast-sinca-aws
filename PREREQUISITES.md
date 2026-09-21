# Cuentas AWS y credenciales (prerequisito, no parte del proyecto)

Estos recursos ya existen y se **reutilizan**, no se vuelven a provisionar por
cada proyecto nuevo. Antes de empezar a planificar, confirmar que los valores
de abajo siguen vigentes — verificar con `aws` CLI real, no asumir.

## Cuentas AWS separadas — regla fija

Estos proyectos personales usan **SIEMPRE** la cuenta personal, y **NUNCA** la
cuenta de la empresa/terceros. La separación se hace por named profile
explícito en la configuración local de AWS (`~/.aws/config` y
`~/.aws/credentials`).

> **Valores sensibles fuera del repo.** El nombre real del profile, el Account
> ID y el usuario IAM son datos privados de la cuenta y **no se versionan acá**.
> En este documento y en el código se usan placeholders: `AWS_PROFILE` (el
> profile de la cuenta personal), `AWS_ACCOUNT_ID` (el Account ID de la cuenta
> personal) e `IAM_ADMIN_USER` (el usuario IAM de trabajo). Los valores reales
> viven solo en tu configuración local (`~/.aws/`, `terraform.tfvars`,
> `backend.hcl`) y nunca se commitean.

| Profile | Cuenta | Uso en estos proyectos |
|---|---|---|
| `AWS_PROFILE` | **Cuenta personal** | Único profile válido para proyectos de esta plantilla. |
| *(otro profile)* | Cuenta de terceros | **PROHIBIDO** usarlo acá. Pertenece a otra cuenta. Si un comando te sugiere apuntar a otro profile, es un error — frenar y preguntar. |

El profile `default` **no se usa nunca**: toda herramienta debe recibir el
profile por nombre explícito, para evitar cruces accidentales entre cuentas.

### Uso explícito del profile por herramienta

- **AWS CLI**: `aws ... --profile $AWS_PROFILE` o `AWS_PROFILE=$AWS_PROFILE aws ...`
- **boto3**: `boto3.Session(profile_name="$AWS_PROFILE")`
- **Terraform**: `provider "aws" { profile = var.aws_profile region = "<region>" }`
  (o vía variable/backend, pero siempre explícito, nunca `~/.aws` default)

### Verificar la cuenta antes de cada sesión

Antes de correr cualquier comando que toque AWS, confirmar identidad:

```
aws sts get-caller-identity --profile $AWS_PROFILE
```

El `Account` devuelto DEBE ser el ID de la cuenta personal (`AWS_ACCOUNT_ID`),
**no** el de ninguna otra cuenta.

**Account ID de la cuenta personal: `AWS_ACCOUNT_ID`** (usuario de trabajo:
`IAM_ADMIN_USER`). Verificado con `aws sts get-caller-identity --profile
$AWS_PROFILE`.

### Configuración local de Terraform (no versionada)

Los valores sensibles se setean localmente, nunca en el repo:

- `infra/terraform.tfvars` (gitignored) — ejemplo en
  `infra/terraform.tfvars.example`: define `aws_profile`.
- `infra/backend.hcl` (gitignored) — ejemplo en
  `infra/backend.hcl.example`: define `bucket`, `profile`, `dynamodb_table`,
  `region` y `key` del backend S3.

Inicializar el backend remoto con:

```
terraform -chdir=infra init -backend-config=backend.hcl
```

## Qué provisionar en la cuenta personal (una sola vez, no por proyecto)

Al crear la cuenta nueva, dejar listo lo siguiente — es la base que todos los
proyectos de esta plantilla asumen:

- **Credenciales del profile `AWS_PROFILE`** en `~/.aws/` (Access Keys).
  Usuario IAM de trabajo: `IAM_ADMIN_USER`. Región default: `us-east-1`.
- **Remote state de Terraform**: un bucket S3 dedicado (ej.
  `<id-cuenta>-tfstate` o similar) + tabla DynamoDB para locking. Ver
  `modules/cicd/blueprint.md` — es obligatorio desde el primer proyecto, no
  optional.
- **IAM del usuario de desarrollo**: el usuario/persona que corre los comandos
  necesita `iam:CreateRole`, `iam:PutRolePolicy`, `iam:AttachRolePolicy` y
  `iam:PassRole` para que el primer `terraform apply` (que crea roles IAM —
  Training Job role, execution roles, roles OIDC de GitHub Actions) no se
  bloquee tarde. Confirmar estos permisos antes del primer apply, no asumirlos.

## Región

Cada proyecto define su región en `project-decisions.md`. Si no hay razón en
contra, `us-east-1` es un default razonable para proyectos personales (precio
y disponibilidad de servicios), pero **no** es regla fija — se decide por
proyecto.

## Qué NO asumir

- No asumir que el profile `AWS_PROFILE` está configurado sin verificarlo con
  `aws sts get-caller-identity`.
- No asumir que el bucket de terraform state existe — se crea la primera vez
  como parte del proyecto que lo necesite (o manualmente de una vez al crear
  la cuenta).
- No asumir regiones, prefijos S3, ni nombres de buckets — se definen
  explícitamente en `project-decisions.md` de cada proyecto.
