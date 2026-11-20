## ADDED Requirements

### Requirement: Rol de orquestación de experimentos

El rol de Step Functions SHALL limitarse a crear, describir y detener Training Jobs con prefijo `sinca-koop-`, pasar solo `sinca-training-role` a SageMaker, gestionar la regla de eventos que usa la integración síncrona con SageMaker, invocar la Lambda de decisión y publicar en el tópico de avisos.

#### Scenario: Job fuera del prefijo
- **WHEN** el rol intenta crear un Training Job cuyo nombre no empieza con `sinca-koop-`
- **THEN** la llamada se deniega

#### Scenario: PassRole acotado
- **WHEN** se inspecciona la política del rol
- **THEN** `iam:PassRole` solo apunta a `sinca-training-role` con `iam:PassedToService = sagemaker.amazonaws.com`

### Requirement: Rol de la Lambda de decisión

El rol de la Lambda de decisión SHALL tener solo lectura (`GetObject`, y `ListBucket` condicionado a ese prefijo) del prefijo de orquestación de la campaña en S3 (decisiones y marcadores) y escritura en su log group. El rol de training no SHALL necesitar permisos nuevos para decisiones, marcadores ni checkpoints.

#### Scenario: Lectura acotada
- **WHEN** la Lambda intenta leer un objeto fuera del prefijo de decisiones
- **THEN** la llamada se deniega

### Requirement: Permisos del rol de infra para la orquestación

`sinca-github-infra-role` SHALL incluir acciones explícitas de Step Functions, Lambda, SNS y EventBridge para el `plan` y el `apply` de los recursos de orquestación, sin `*FullAccess` ni `Action: "*"`.

#### Scenario: Plan en CI
- **WHEN** infra-cd corre `terraform plan` en un PR con los recursos de orquestación ya creados
- **THEN** el refresh no falla por permisos
