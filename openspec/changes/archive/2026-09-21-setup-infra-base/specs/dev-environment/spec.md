## ADDED Requirements

### Requirement: Entorno de desarrollo remoto persistente

El sistema SHALL provisionar un entorno de desarrollo remoto basado en un Space de SageMaker Studio, definido como código en Terraform, con home sobre EFS para persistir el código y el contexto aunque el cómputo se apague.

#### Scenario: Persistencia al apagar el Space
- **WHEN** el Space se apaga y se vuelve a prender
- **THEN** el contenido del home (`/home/sagemaker-user`) permanece intacto en EFS

#### Scenario: Reproducibilidad desde Terraform
- **WHEN** se ejecuta `terraform apply` en una cuenta limpia
- **THEN** el dominio de Studio, el Space y el lifecycle config se crean sin pasos manuales en consola

### Requirement: Lifecycle config para reinstalar lo efímero

El lifecycle config (`~/.on_start`) SHALL reinstalar al arrancar todo lo que no persiste en EFS: Node.js + Open Code CLI, AWS CLI, Terraform, y el clon del repo del proyecto.

#### Scenario: Arranque de Space nuevo
- **WHEN** el Space inicia sobre un cómputo efímero nuevo
- **THEN** Node.js, Open Code CLI y las herramientas CLI quedan disponibles tras ejecutar `~/.on_start`

### Requirement: Acceso remoto SSH sobre SSM

El Space SHALL habilitar acceso remoto SSH sobre Systems Manager (SSM) para conectar VS Code local contra el filesystem remoto.

#### Scenario: Conexión de VS Code local
- **WHEN** se habilita remote access y se conecta VS Code local vía Remote-SSH
- **THEN** la sesión abre el filesystem del Space (EFS) con la configuración local del usuario

### Requirement: Tamaño de instancia del Space

El Space SHALL usar una instancia `ml.t3.large` (8 GB RAM) como mínimo útil para edición y pruebas ligeras.

#### Scenario: Edición y pruebas ligeras
- **WHEN** se edita código y se corren comandos de `openspec` en el Space
- **THEN** la instancia `ml.t3.large` es suficiente para esas tareas sin necesidad de cómputo dedicado
