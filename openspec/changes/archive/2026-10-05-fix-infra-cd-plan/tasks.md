## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar que `infra/terraform.tfvars` y `infra/backend.hcl` locales están completos (no recreados a medias)

## 1. Escritura de los fixes

- [x] 1.1 En `infra/iam-oidc.tf`, statement `CloudWatchLogs` de `github_infra_policy`: reemplazar `logs:TagLogGroup`/`logs:UntagLogGroup`/`logs:ListTagsLogGroup` por `logs:TagResource`/`logs:UntagResource`/`logs:ListTagsForResource`
- [x] 1.2 En `.github/workflows/infra-cd.yml`, steps `terraform plan` y `terraform apply`: agregar `-var="user_profile_name=${{ vars.USER_PROFILE_NAME }}"` y `-var="mlflow_allowed_cidr=${{ vars.MLFLOW_ALLOWED_CIDR }}"`
- [x] 1.3 En `infra/iam-oidc.tf`, agregar a `github_infra_policy` un statement `Scheduler` (`scheduler:GetSchedule`/`CreateSchedule`/`UpdateSchedule`/`DeleteSchedule`/`TagResource`/`UntagResource` sobre el ARN del schedule `sinca-scraper-daily`) y un statement `SchedulerList` (`scheduler:ListSchedules` con `resources = ["*"]`)

## 2. Ejecución real (operador corre y pega output)

- [x] 2.1 `terraform -chdir=infra plan` (profile `$AWS_PROFILE`) — revisar que solo cambie la policy del rol `sinca-github-infra-role` (+ drift SageMaker conocido, no destructivo)
- [x] 2.2 `terraform -chdir=infra apply` (profile `$AWS_PROFILE`) — actualiza la policy del rol en AWS
- [x] 2.3 `terraform -chdir=infra plan` y `apply` (profile `$AWS_PROFILE`) — aplica el statement `Scheduler` nuevo (revisar que solo cambie la policy del rol + drift SageMaker)

## 3. Configuración de variables de repo (operador, en GitHub UI)

- [x] 3.1 Crear la variable `USER_PROFILE_NAME` con el valor real del user profile (leer de `infra/terraform.tfvars`)
- [x] 3.2 Crear la variable `MLFLOW_ALLOWED_CIDR` con el valor real del CIDR (leer de `infra/terraform.tfvars`)

## 4. Verificación funcional

- [x] 4.1 Commitear y pushear a `main`; re-ejecutar el workflow `infra-cd` y confirmar que `terraform plan` devuelve `No changes` (o solo el drift SageMaker)
- [x] 4.2 Confirmar que el step `terraform apply` no aplica cambios destructivos (no reemplaza el user profile ni abre el SG); el único change es el drift perpetuo de SageMaker (`studio_web_portal_settings {}`), conocido y no destructivo

## Notas de implementación

- **0.1 — profile:** igual que en el change anterior, el `$AWS_PROFILE` del
  entorno era `evolet` (otra cuenta, `457469184690`). Se usó `desareca_dev`
  (`549024266383`) por indicación del operador. `infra/terraform.tfvars`
  (`aws_profile = "desareca_dev"`) y `infra/backend.hcl` (`profile =
  "desareca_dev"`) ya apuntan al profile correcto, así que el `terraform plan`
  local usa `desareca_dev` sin depender del env var.
- **2.1 — plan confirmado:** `Plan: 0 to add, 2 to change, 0 to destroy`. Solo
  cambia el `policy` de `aws_iam_role_policy.github_infra_inline` (las 3
  acciones de logs) + el drift no destructivo de `aws_sagemaker_domain.studio`
  (`- studio_web_portal_settings {}`). No hay cambios destructivos.
- **Valores reales para las variables de repo (3.1/3.2):** leídos de
  `infra/terraform.tfvars` → `USER_PROFILE_NAME=desareca-admin`,
  `MLFLOW_ALLOWED_CIDR=190.5.58.101/32`.
- **2.2 — `terraform apply` está en el deny list** de `opencode.json`, así que
  lo corre el operador.
- **2.2 / 3.1 / 3.2 — confirmadas por el run del workflow:** en el run de
  `infra-cd` disparado por el push a `main` ya no aparece el error de
  `logs:ListTagsForResource` (la policy nueva está aplicada en AWS) y el plan
  ya no propone reemplazar el user profile ni abrir el SG (las variables de
  repo están seteadas con los valores reales). El plan quedó en
  `0 to add, 1 to change, 0 to destroy` (solo drift SageMaker).
- **4.1 — NUEVO BLOQUEO (fuera de alcance del change):** el `terraform plan`
  del workflow ahora falla con otro permiso faltante en
  `sinca-github-infra-role`:
  ```
  AccessDeniedException: not authorized to perform: scheduler:GetSchedule
  on resource: arn:aws:scheduler:us-east-1:549024266383:schedule/default/sinca-scraper-daily
  ```
  El rol no tiene **ningún** permiso de EventBridge Scheduler, y Terraform
  gestiona `aws_scheduler_schedule.scraper_daily` (`infra/schedule.tf`) → no
  puede refrescarlo. Los tasks/specs de este change solo cubren los permisos
  de logs, no Scheduler. Requiere decisión del Arquitecto: extender este
  change (agregar un statement `Scheduler` de mínimo privilegio:
  `GetSchedule`/`CreateSchedule`/`UpdateSchedule`/`DeleteSchedule`/
  `ListSchedules`/`TagResource`/`UntagResource`) o abrir un change nuevo.
  **No se tocó nada de esto.**
- **1.3 / 2.3 — permiso de Scheduler agregado y plan confirmado:** se agregaron
  a `github_infra_policy` los statements `Scheduler` (CRUD + tags, acotado al
  ARN `arn:aws:scheduler:us-east-1:549024266383:schedule/default/sinca-scraper-daily`)
  y `SchedulerList` (`scheduler:ListSchedules`, `resources = ["*"]`). El
  `terraform plan` local da `Plan: 0 to add, 2 to change, 0 to destroy`: solo
  el `policy` de `github_infra_inline` (agrega los 2 statements) + el drift
  SageMaker. Falta el `apply` (lo corre el operador).
- **2.3 / 4.1 — confirmados por el run del workflow:** el `plan` del CI corrió
  sin `AccessDeniedException` (el permiso de Scheduler quedó aplicado por el
  `apply` local de 2.3: el `apply` del workflow solo tocó SageMaker, no la
  policy). El `plan` devolvió solo el drift SageMaker y el `apply` del workflow
  terminó OK.
- **4.2 — apply = `0 added, 1 changed, 0 destroyed`** (no `0/0/0`): el único
  cambio fue el drift preexistente de `aws_sagemaker_domain.studio`
  (`studio_web_portal_settings {}`), que el workflow aplicó (no destructivo,
  aceptable según design). **No** reemplazó el user profile ni abrió el SG, que
  es el objetivo de seguridad de la tarea. Como el drift ya se aplicó, un
  re-run debería dar `No changes` + `0/0/0`; queda pendiente confirmarlo.
- **4.2 — la drift de SageMaker es PERPETUA (no se resuelve con un apply):** el
  re-run volvió a dar `Plan: 0 to add, 1 to change, 0 to destroy` y
  `Apply complete! Resources: 0 added, 1 changed, 0 destroyed`, **de nuevo solo
  `aws_sagemaker_domain.studio`** (`- studio_web_portal_settings {}`). El
  provider vuelve a proponer la remoción en cada plan tras cada apply, así que
  el plan nunca queda en `No changes` y el apply nunca queda en `0/0/0`. Por
  eso **4.2 tal como está redactada (`0 added, 0 changed, 0 destroyed`) es
  inalcanzable**. Su objetivo de seguridad **sí** se cumple: el apply no
  reemplaza el user profile ni abre el SG, y no hay cambios destructivos (el
  `design.md` ya declara este drift como no destructivo y aceptable). Requiere
  ajuste de la tarea por el Arquitecto (o aceptar el change con el drift
  conocido). **No se tocó nada de esto.**
