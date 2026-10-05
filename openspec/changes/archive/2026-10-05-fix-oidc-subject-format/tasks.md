## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar que `infra/terraform.tfvars` y `infra/backend.hcl` locales están completos (no recreados a medias), para que el plan no proponga destruir recursos

## 1. Escritura del fix

- [x] 1.1 En `infra/iam-oidc.tf`, cambiar `local.github_repo` de `"desareca/forecast-sinca-aws"` a `"desareca@43764566/forecast-sinca-aws@1380017292"` (formato inmutable del `sub`)

## 2. Ejecución real (operador corre y pega output)

- [x] 2.1 `terraform -chdir=infra plan` (profile `$AWS_PROFILE`) — revisar que solo cambie el trust policy de los dos roles OIDC (+ drift SageMaker conocido, no destructivo)
- [x] 2.2 `terraform -chdir=infra apply` (profile `$AWS_PROFILE`) — actualiza el trust policy en AWS

## 3. Verificación funcional

- [x] 3.1 Re-ejecutar el workflow `infra-cd` en GitHub y confirmar que `configure-aws-credentials` asume `sinca-github-infra-role` sin error
- [x] 3.2 Confirmar que el fallo del plan del CI ya no es por el `sub` (el `No changes` completo queda bloqueado por issues preexistentes ajenos al fix — permiso `logs:ListTagsForResource` y variables del workflow — resueltos en el change `fix-infra-cd-plan`)
- [x] 3.3 Re-ejecutar el workflow `pipeline-cd` en GitHub y confirmar que asume `sinca-github-pipeline-role` sin error

## Notas de implementación

- **0.1 — profile correcto vs. `$AWS_PROFILE` del entorno:** el shell tenía
  `AWS_PROFILE=evolet`, que apunta a **otra cuenta** (Account `457469184690`,
  user `wsl-csaquel-dev`), no a la cuenta del proyecto. El profile real del
  proyecto es `desareca_dev` (Account `549024266383`, user `desareca_admin`),
  consistente con `infra/terraform.tfvars` (`aws_profile = "desareca_dev"`) y
  `infra/backend.hcl` (bucket `549024266383-tfstate`, `profile = "desareca_dev"`).
  Se ejecutaron los comandos con `--profile desareca_dev` por indicación del
  operador. **Riesgo:** la tarea 0.1 tal como está escrita usa `$AWS_PROFILE`,
  que en este entorno resolvía a la cuenta equivocada; conviene que el
  operador exporte `AWS_PROFILE=desareca_dev` (o que la tarea lo explicite)
  antes de correr comandos que toquen AWS.
- **2.1 — plan confirmado:** `Plan: 0 to add, 3 to change, 0 to destroy`.
  Solo cambian `assume_role_policy` de `sinca-github-infra-role` y
  `sinca-github-pipeline-role`, más el drift no destructivo conocido de
  `aws_sagemaker_domain.studio` (`- studio_web_portal_settings {}`).
- **2.2 — `terraform apply` está en el deny list** de `opencode.json`
  (`{"resource": "terraform apply*", "effect": "deny"}`), así que lo corre el
  operador, no el agente.
- **3.1 — OIDC confirmado funcionando:** en el run de `infra-cd`,
  `configure-aws-credentials@v4` asumió el rol sin error:
  `Set output authenticated-arn = arn:aws:sts::549024266383:assumed-role/sinca-github-infra-role/GitHubActions`.
  Ya no aparece `Not authorized to perform sts:AssumeRoleWithWebIdentity`. El
  fix del `sub` inmutable es correcto.
- **3.2 — BLOQUEO (fuera de alcance del change):** el `terraform plan` del
  workflow falla, por dos causas **ajenas al fix del `sub`** y preexistentes:
  1. **Falta un permiso IAM en `sinca-github-infra-role`:** el provider AWS
     v5.100.0, al refrescar, llama `logs:ListTagsForResource` sobre los log
     groups y el rol no lo tiene (sí tiene `logs:ListTagsLogGroup`, que es la
     API vieja). Error:
     `... not authorized to perform: logs:ListTagsForResource ... because no
     identity-based policy allows ...`. Agregar ese permiso cambia el alcance
     de la policy → el `design.md` de este change lo declara **Non-Goal**
     ("No se cambia el alcance de permisos de los roles (solo la condición
     `sub`)").
  2. **El CI no tiene los valores reales de variables:** `infra-cd.yml` corre
     `terraform plan -var="aws_profile="` **sin** `terraform.tfvars`
     (gitignored), así que usa los `default` de `variables.tf`:
     - `user_profile_name` default `sinca-dev-user` ≠ real `desareca-admin`
       → el plan quiere **reemplazar** `aws_sagemaker_user_profile.admin`
       (`-/+ must be replaced`, `1 to destroy`).
     - `mlflow_allowed_cidr` default `0.0.0.0/0` ≠ real `190.5.58.101/32`
       → el plan quiere **abrir el SG de MLflow al mundo**.
     Estos diffs hacen que 3.2 ("No changes") sea imposible de cumplir tal
     como está redactada. **Riesgo grave:** si solo se arregla el permiso de
     logs, el paso `terraform apply` (que corre en `main`) aplicaría esos
     cambios destructivos (borrar/recrear el user profile y abrir el SG). No
     se tocó nada de esto: son hallazgos para decisión del Arquitecto.
  3. **El run corrió contra el código viejo (fix sin commitear):** el checkout
     del workflow fue el commit `85b56e0` de `main`, que **no** contiene el
     fix (el edit de 1.1 es local, sin commitear). Por eso el plan del CI
     muestra los roles **revirtiendo** al formato viejo
     (`repo:desareca@43764566/...` -> `repo:desareca/forecast-sinca-aws:*`).
     Es esperable en esta etapa: falta commitear/pushear el fix. Aun así, los
     puntos 1 y 2 seguirían rompiendo el plan del workflow aunque el fix
     estuviera en `main`.
- **3.3 — OIDC de pipeline confirmado funcionando:** en el run de
  `pipeline-cd`, `configure-aws-credentials@v4` asumió el rol sin error:
  `Set output authenticated-arn = arn:aws:sts::549024266383:assumed-role/sinca-github-pipeline-role/GitHubActions`.
  El job terminó OK (paso skeleton: "pipeline/ todavía no existe — no-op por
  ahora.").
