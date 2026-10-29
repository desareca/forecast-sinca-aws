## ADDED Requirements

### Requirement: Scripts locales de start/stop del Space

El sistema SHALL disponer de scripts locales (no versionados) para arrancar y apagar el app JupyterLab del Space de forma confiable: `start-space.ps1` (idempotente, crea el app si falta o lo recrea si quedó en mal estado, y espera `InService`) y `stop-space.ps1` (`delete-app`, conserva EFS/EBS). Los scripts llevan el profile y account reales, por lo que se versionan solo como `.example` con placeholders (`AWS_PROFILE`, `AWS_ACCOUNT_ID`, `DOMAIN_ID`) y los `.ps1` reales quedan en `.gitignore`.

#### Scenario: Arranque confiable tras un Stop
- **WHEN** el operador ejecuta `start-space.ps1` con el Space `InService` pero sin app (o con el app en mal estado)
- **THEN** el script crea (o recrea) el app JupyterLab y espera hasta que quede `InService`, sin depender del "Open" de la consola

#### Scenario: Apagado que conserva el estado
- **WHEN** el operador ejecuta `stop-space.ps1`
- **THEN** se borra el app (cómputo) y el home en EFS/EBS permanece intacto para el próximo arranque

#### Scenario: Credenciales fuera del repo
- **WHEN** se versiona el repo
- **THEN** los `.ps1` con valores reales no se commitean (gitignored) y solo los `.example` con placeholders quedan versionados

### Requirement: Autenticación git desde el Space

El sistema SHALL permitir `git push` desde el Space de SageMaker Studio mediante una key SSH (`ed25519`) persistida en el home (EFS), con la pública registrada en GitHub y el remote del clone apuntando a SSH (`git@github.com:...`). El lifecycle config SHALL clonar el repo por SSH para que un Space nuevo quede autenticado desde el primer arranque.

#### Scenario: Push desde el Space
- **WHEN** el operador hace `git push` desde el clone del Space
- **THEN** la operación se autentica con la key SSH persistida en EFS, sin credenciales en el repo

#### Scenario: Clonado autenticado en Space nuevo
- **WHEN** un Space nuevo ejecuta el lifecycle config (`on_start.sh`)
- **THEN** el repo se clona por SSH, de modo que `git pull`/`git push` funcionan sin pasos manuales de autenticación

### Requirement: Imagen SageMaker del Space fijada

El `default_resource_spec` del Space SHALL definir `sagemaker_image_arn` (`arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu`) y `sagemaker_image_version_alias` versionado (`3.2.0`), no `latest`, para que un recreate del Space desde cero sea determinístico. Los scripts de start/stop SHALL propagar la misma imagen/alias.

#### Scenario: Recreate sin arqueología
- **WHEN** se recrea el app o el Space desde cero (EBS/home nuevos)
- **THEN** el app arranca con la imagen/alias definidos en Terraform y en `start-space.ps1`, sin recuperar el ARN desde CloudTrail

### Requirement: Lifecycle config robusta y con logs

El lifecycle config (`on_start.sh`) SHALL (1) no fallar el arranque del app por tooling de conveniencia (instalación de CLIs, clonado del repo), (2) fallar ruidosamente solo en pasos declarados requisito, y (3) loggear toda su salida a `~/.on_start.log` (EFS persistente).

#### Scenario: Conveniencia no bloqueante
- **WHEN** un paso de conveniencia falla (p. ej. paquete npm inexistente, clonado sin key SSH)
- **THEN** el app arranca igual y el fallo queda registrado en `~/.on_start.log`, sin matar el script

#### Scenario: Log legible sin el app
- **WHEN** el app quedó en `Failed`
- **THEN** `~/.on_start.log` (EFS) conserva la salida completa de la última corrida para diagnóstico
