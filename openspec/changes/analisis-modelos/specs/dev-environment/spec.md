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
