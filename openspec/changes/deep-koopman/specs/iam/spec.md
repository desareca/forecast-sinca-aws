## ADDED Requirements

### Requirement: Rol de orquestación de experimentos

El rol de Step Functions SHALL limitarse a crear, describir y detener Training Jobs con prefijo `sinca-koop-`, pasar solo `sinca-training-role` a SageMaker, gestionar la regla de eventos que usa la integración síncrona con SageMaker, invocar la Lambda de orquestación y publicar en el tópico de avisos.

#### Scenario: Job fuera del prefijo
- **WHEN** el rol intenta crear un Training Job cuyo nombre no empieza con `sinca-koop-`
- **THEN** la llamada se deniega

#### Scenario: PassRole acotado
- **WHEN** se inspecciona la política del rol
- **THEN** `iam:PassRole` solo apunta a `sinca-training-role` con `iam:PassedToService = sagemaker.amazonaws.com`

### Requirement: Rol de la Lambda de orquestación

El rol de la Lambda de orquestación SHALL tener solo lectura (`GetObject`, y `ListBucket` condicionado a ese prefijo) del prefijo de orquestación de la campaña en S3 (marcadores y selección) y escritura en su log group. El rol de training no SHALL necesitar permisos nuevos para selección, marcadores ni checkpoints.

#### Scenario: Lectura acotada
- **WHEN** la Lambda intenta leer un objeto fuera del prefijo de orquestación
- **THEN** la llamada se deniega

### Requirement: Permisos del rol de infra para la orquestación

`sinca-github-infra-role` SHALL incluir acciones explícitas de Step Functions, Lambda, SNS y EventBridge para el `plan` y el `apply` de los recursos de orquestación, sin `*FullAccess` ni `Action: "*"`, y su `iam:PassRole` SHALL permitir además `states.amazonaws.com` y `lambda.amazonaws.com` como servicio destino.

#### Scenario: Plan en CI
- **WHEN** infra-cd corre `terraform plan` en un PR con los recursos de orquestación ya creados
- **THEN** el refresh no falla por permisos

#### Scenario: Sin deriva tras el apply local
- **WHEN** infra-cd corre `terraform plan` en el PR solo de infra después del apply local
- **THEN** el plan muestra 0 cambios, incluido el hash del código de la Lambda

#### Scenario: Email no expuesto
- **WHEN** se inspeccionan los logs del plan de CI
- **THEN** el email del operador no aparece en texto claro
