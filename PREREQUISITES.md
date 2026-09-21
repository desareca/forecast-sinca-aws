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
> En este documento, en las specs y en el código se usan placeholders fijos:
>
> | Placeholder | Reemplaza a | Vive realmente en |
> |---|---|---|
> | `AWS_PROFILE` | nombre del named profile de la cuenta personal | `~/.aws/`, `terraform.tfvars`, `backend.hcl` |
> | `AWS_ACCOUNT_ID` | Account ID (12 dígitos) de la cuenta personal | `terraform.tfvars`, `backend.hcl` |
> | `IAM_ADMIN_USER` | usuario IAM de trabajo de la cuenta personal | `~/.aws/` |
> | *(otra cuenta)* | nombre de la cuenta de empresa/terceros (jamás se usa acá) | — |
>
> **Regla fija:** ninguna spec, comando, `.tf` ni `.md` vuelve a escribir estos
> valores reales. Si llegan a aparecer (p. ej. un nombre de cuenta ajeno, una
> ruta local con tu usuario de Windows, o un username personal), se reemplazan
> por su placeholder antes de commitear. Los valores reales solo existen en la
> configuración local y nunca se commitean.

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

### Bootstrap local (una sola vez por clon, no versionado)

El repo **no** trae los valores reales (son privados). Al clonar en una máquina
nueva, crear los dos archivos locales a partir de sus `.example` y completarlos
con los valores de la cuenta:

```powershell
# 1. Profile del provider AWS (variable var.aws_profile)
Copy-Item infra\terraform.tfvars.example infra\terraform.tfvars
#   -> aws_profile = "<nombre-real-del-profile>"

# 2. Backend S3 (bucket, profile, dynamodb_table, región, key)
Copy-Item infra\backend.hcl.example infra\backend.hcl
#   -> bucket  = "<account-id>-tfstate"
#   -> profile = "<nombre-real-del-profile>"
```

Ambos quedan cubiertos por `.gitignore` (`*.tfvars` y `backend.hcl`): **nunca
se commitean**. Los `.example` sí se versionan como referencia.

Inicializar el backend remoto con:

```
terraform -chdir=infra init -backend-config=backend.hcl
```

El resto de los comandos de AWS (`aws`, `boto3`, etc.) usa el profile real vía
`$AWS_PROFILE`, como se documenta más arriba.

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
