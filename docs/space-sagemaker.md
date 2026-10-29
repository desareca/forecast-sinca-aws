# Space de SageMaker — operación y diagnóstico

Runbook del entorno de desarrollo (`sinca-dev`) en SageMaker Studio. Cubre el
ciclo normal (prender / trabajar / apagar) y el diagnóstico paso a paso cuando el
Space **no arranca**. Todo lo que está acá salió de fallos reales y ya está
resuelto en el código; los valores concretos de cuenta/dominio van como
placeholders — los reales viven en los scripts locales (gitignored).

> **Regla de oro**: si el Space no arranca, la causa está en **uno de estos tres
> lados**: la imagen del app, el lifecycle config, o la policy del rol. El cartel
> que muestra la consola no distingue — dice "Permissions not configured
> correctly" para cualquiera de los tres.

## 1. Qué compone el entorno

| Pieza | Dónde se define | Qué es |
|---|---|---|
| Dominio `sinca-studio` | `infra/sagemaker-domain.tf` | Studio con `auth_mode = IAM` |
| User profile `desareca-admin` | idem | dueño del Space |
| Space `sinca-dev` | idem | `ml.t3.large` + home en **EFS** (persiste) |
| App JupyterLab `default` | se crea al prender | **cómputo efímero** (se borra al apagar) |
| Imagen del app | `sagemaker_image_arn` + `_version_alias` en `infra/variables.tf` | `sagemaker-distribution-cpu` (pública de SageMaker, alias `3.2.0`) |
| Lifecycle config `sinca-on-start` | `infra/lifecycle.tf` ← `infra/scripts/on_start.sh` | corre **en cada arranque** |
| Rol del Space | `infra/iam-dev.tf` (`sinca-dev-role`) | mínimo privilegio: S3, logs, SageMaker app |

Lo que **persiste** vs lo que **no**:

- **EFS** (`/home/sagemaker-user`): código, clones, venv, `~/.ssh`, `~/.on_start.log`. Sobrevive apagar.
- **EBS** (disco del cómputo): paquetes instalados a mano (`apt`, `npm`, binarios). **Se pierde** al recrear el app → por eso existe la LCC.
- **App**: el cómputo en sí. Apagado = `list-apps` vacío = no factura.

## 2. Operación normal

### Primera vez (una sola vez por máquina)

```powershell
Copy-Item infra/scripts/start-space.ps1.example infra/scripts/start-space.ps1
Copy-Item infra/scripts/stop-space.ps1.example infra/scripts/stop-space.ps1
# completar AWS_PROFILE, DOMAIN_ID, AWS_ACCOUNT_ID
```

Ambos `.ps1` están en `.gitignore` (llevan valores reales). Solo se versionan los
`.example`.

### Ciclo

```powershell
.\infra\scripts\start-space.ps1   # crea/recrea el app y espera InService
# ... trabajar en JupyterLab o VS Code remoto ...
.\infra\scripts\stop-space.ps1    # borra el cómputo, conserva EFS
```

`start-space.ps1` es **idempotente**: si está `InService` no hace nada; si quedó
en mal estado lo borra y recrea; si no existe lo crea. **No es opcional** usarlo:
la consola a veces no recrea el app y tira
`App [default] ... is not found or in unusable state`.

### Git desde el Space

El clone del Space corre con el rol del Space (sin credenciales). El remote por
**SSH** funciona con una key `ed25519` que vive en EFS (se configura una vez):

```bash
ssh-keygen -t ed25519 -C "sinca-dev-space"    # Enter en todo
cat ~/.ssh/id_ed25519.pub                     # GitHub -> Settings -> SSH keys
git remote set-url origin git@github.com:desareca/forecast-sinca-aws.git
ssh -T git@github.com                         # "Hi desareca! ..."
```

Patrón de trabajo: **dos clones**. Commit/push en local, `git pull` en el Space.

Antes de pushear desde el Space, siempre:

```bash
git pull --rebase
git push
```

Sin `--rebase`, `git pull` con historias separadas falla con
*"Need to specify how to reconcile divergent branches"*.

## 3. Cuando no arranca

### Lo que se ve

Consola de Studio → banner:

```
Permissions not configured correctly
Your account may not be configured properly. Please contact your AWS administrator...
```

**No es un problema de permisos.** Es el cartel genérico que Studio muestra
cuando no puede crear el app. Los permisos del rol están bien.

### Diagnóstico, en orden

```powershell
$env:AWS_PROFILE = "desareca_dev"

# 1. ¿Hay app? ¿En qué estado y por qué falló?
aws sagemaker list-apps --region us-east-1 --profile $env:AWS_PROFILE
aws sagemaker describe-app --domain-id DOMAIN_ID --app-type JupyterLab `
  --app-name default --space-name sinca-dev --region us-east-1 --profile $env:AWS_PROFILE `
  --query "{Status:Status,FailureReason:FailureReason}"

# 2. ¿El Space tiene imagen configurada?
aws sagemaker describe-space --domain-id DOMAIN_ID --space-name sinca-dev `
  --region us-east-1 --profile $env:AWS_PROFILE `
  --query "SpaceSettings.JupyterLabAppSettings.DefaultResourceSpec"
```

Interpretación:

| `FailureReason` / síntoma | Causa | Ir a |
|---|---|---|
| `SageMaker Image ARN is required for App with type [JupyterLab]` | falta `sagemaker_image_arn` en el Space | §4.1 |
| `ConfigurationError: LifecycleConfig execution failed with non zero exit code` | falla `on_start.sh` | §4.2 |
| App `Failed` y no se puede borrar (`previously failed and was automatically deleted`) | normal: los apps fallidos se autoborran y solo son visibles 24 h | — |
| `AccessDenied ... s3:ListBucket on arn:aws:s3:::sinca-mlflow` | policy del rol dev | §4.3 |
| `Error acquiring the state lock ... runner@...` | CI está aplicando Terraform | §4.5 |

Si el app **sí** arrancó, el diagnóstico fino está en el log de la LCC (§4.2).

## 4. Causas conocidas (y su fix)

### 4.1 El Space no tiene imagen

`aws_sagemaker_space` en Terraform define instancia + lifecycle config, pero **no**
la imagen. Sin `sagemaker_image_arn`, `create-app` falla siempre (y la consola
también). El ARN correcto **no** es `.../image/sagemaker-distribution` (es inválido);
es la imagen CPU con alias:

```
arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu   # alias 3.2.0
```

Ese ARN pertenece a la cuenta **pública** de SageMaker (`885854791233`), no a la
cuenta del proyecto: por eso en los scripts va literal y no con el placeholder
`AWS_ACCOUNT_ID`.

Fix: `sagemaker_image_arn` + `sagemaker_image_version_alias` en
`infra/variables.tf` y su uso en `default_resource_spec`. Fuente única de verdad:
si cambia, cambia en un solo lugar.

> Cómo se descubre el ARN que funciona: `aws cloudtrail lookup-events
> --lookup-attributes AttributeKey=EventName,AttributeValue=CreateApp` y mirar el
> `requestParameters.resourceSpec` de un `CreateApp` exitoso.

### 4.2 La lifecycle config falla

`on_start.sh` corre con `set -uo pipefail` (**sin** `-e`) y separa dos clases de
paso:

| Función | Semántica | Uso |
|---|---|---|
| `_fail "<msg>"` | requisito: aborta el arranque | solo si el Space no sirve sin eso |
| `_try "<desc>" <func>` | conveniencia: loguea `WARN (no bloqueante)` y sigue | CLIs, clones |

Hoy **no hay ningún `_fail`**: el app arranca sin ninguna CLI (el repo y los venv
están en EFS). Todo es `_try`.

Toda la salida se escribe en **`/home/sagemaker-user/.on_start.log`** (EFS), con
markers por corrida:

```
=== on_start BEGIN (user=..., repo=...) ===
OK: AWS CLI v2
WARN (no bloqueante): 'Terraform' fallo (exit 100) — el Space arranca igual
OK: repo clonado por SSH
=== on_start END (ok) ===
```

**Este log es el motivo por el que el script no usa `-e`**: un `npm install` de un
paquete inexistente (`@opencode-ai/opencode`, 404) tumbaba el app entero, y el
fallo solo se veía como "Permissions not configured correctly". El log convirtió
un fallo opaco en una línea legible.

Para leerlo, en el terminal de JupyterLab: `cat ~/.on_start.log`.

**La LCC se reemplaza, no se actualiza**: cambiar el contenido hace que Terraform
haga `destroy` + `create` del recurso. AWS permite borrar una LCC en uso, pero el
Space queda apuntando a un ARN que se recrea en el mismo apply.

### 4.3 La policy del rol no permite listar `sinca-mlflow`

Síntoma: `AccessDenied ... s3:ListBucket on arn:aws:s3:::sinca-mlflow`, típico al
listar artefactos de MLflow (notebook, `mlflow_log.download_state()`).

**`ListBucket` exige el ARN del bucket, no un prefijo.** Tenerlo sobre
`arn:aws:s3:::sinca-mlflow/_mlflow` no sirve para listar (sí para `GetObject`).
Fix mínimo privilegio:

```hcl
statement {
  effect    = "Allow"
  actions   = ["s3:ListBucket"]
  resources = [aws_s3_bucket.mlflow.arn]
  condition {
    test     = "StringLike"
    variable = "s3:prefix"
    values   = ["_mlflow/", "_mlflow/*"]
  }
}
```

Diagnóstico rápido: `GetObject` funciona (baja el `mlflow.sqlite`) pero
`list_objects_v2` rebota → es esto.

### 4.4 El clone del Space divergió

Síntoma: `! [rejected] main -> main (fetch first)` o *"Need to specify how to
reconcile divergent branches"*.

```bash
git status -sb                        # "ahead N, behind M"
git log --oneline HEAD..origin/main   # lo que te falta
git log --oneline origin/main..HEAD   # lo que tenés de más
git log --oneline --graph --decorate --all -12
```

Recuperación (el Space es un clon de consumo, no la fuente de verdad):

```bash
git fetch origin
git reset --hard origin/main
```

> Antes de resetear, si hay cambios sin commitear que valgan: `git stash push -m "..."`.

### 4.5 Terraform: lock y `-backend-config`

- **Lock tomado por CI**: `.github/workflows/infra-cd.yml` corre `plan` **y
  `apply -auto-approve`** en cada push a `main` que toque `infra/**`. Si ves
  `Error acquiring the state lock ... runner@runnervm...`, **esperá**: no es tu
  proceso. El `apply` manual es redundante y compite por el lock.
- **`-backend-config` con `=`**: en el Terraform local (`1.15.7`) el flag con `=`
  se interpreta como argumento posicional →
  `Too many command line arguments. Did you mean to use -chdir?`. Usar:

  ```powershell
  terraform -chdir=infra init -backend-config backend.hcl
  ```

  (el CI usa `1.9.8` y ahí `-backend-config="bucket=..."` sí funciona).

## 5. Recrear el Space desde cero (checklist)

Sirve para validar cualquier cambio en esta parte, y es el criterio de aceptación
real: **no "arrancó una vez", sino "arranca desde cero".**

1. Revisar que `terraform plan` solo toque lo esperado:
   ```powershell
   terraform -chdir=infra plan
   ```
2. Aplicar (`main` + CI lo hace solo; si aplicás local, esperá que CI termine):
   `terraform -chdir=infra apply`
3. `.\infra\scripts\start-space.ps1` → `InService`.
4. En JupyterLab: `cat ~/.on_start.log` → `END (ok)`, sin `ERROR`.
5. `git pull` en el Space y abrir `notebooks/analisis_baseline.ipynb` → **Run All**:
   - 7 runs `FINISHED`
   - 6 `predictions.parquet`
   - gráfico real vs pred
6. `.\infra\scripts\stop-space.ps1`.

## 6. Pendientes / decisiones abiertas

- **`apt-get` sin `sudo`**: la LCC corre como `sagemaker-user`, así que Terraform
  nunca se instala y queda en `WARN`. Mientras el Terraform se corra desde local
  no molesta, pero contradice la fila `dev-environment` de
  `project-decisions.md`. Fix candidato: `sudo` en los `apt-get` (y en las
  escrituras a `/usr/share/keyrings` y `/etc/apt/sources.list.d/`).
- **Notebook con outputs versionados**: el archivo debería commitearse sin
  outputs; hoy hay una corrida commiteada.
- **Costos**: el Space encendido factura ~USD 0.06/h. Apagarlo no es opcional.
