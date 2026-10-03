# Blueprint: dev-environment

**Cuándo usar este blueprint**: siempre (transversal). Define el entorno de
desarrollo persistente y de bajo costo para trabajar en proyectos de esta
plantilla, basado en un **Space de Amazon SageMaker Studio** con VS Code local
conectado remotamente.

## Objetivo

Un entorno de desarrollo que **persista indefinidamente** (código, contexto de
opencode, proyectos) aunque se apague el cómputo, para poder **apagarlo cuando
no se usa** sin perder nada ni gastar. Al prenderlo de nuevo, todo queda igual.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│  SageMaker Studio Space (se prende/apaga)               │
│  ┌───────────────────────────────────────────────┐      │
│  │  Home directory sobre EFS (persiste siempre)  │      │
│  │  - código, .opencode/, proyectos, git repos   │      │
│  ├───────────────────────────────────────────────┤      │
│  │  JupyterLab  o  Code Editor (UI)              │      │
│  │  Lifecycle Config  ~/.on_start (corre AL      │      │
│  │  prender el Space)                            │      │
│  └───────────────────────────────────────────────┘      │
└──────────────┬──────────────────────────────────────────┘
               │ SSH sobre SSM (remote access habilitado)
               ▼
   VS Code LOCAL (tu máquina)
   - AWS Toolkit + Microsoft Remote-SSH
   - tu configuración, extensiones y atajos personales
   - dos terminales: Arquitecto / Implementador
```

## Componentes clave

### 1. Home directory sobre EFS

- Todo el home del Space vive en **EFS** (`/home/sagemaker-user`), que
  **persiste aunque se apague el cómputo**. Apagar el Space no borra el código.
- Acá viven: repos clonados, el contexto `.opencode/` del proyecto, configs,
  entornos virtuales. Prender/apagar el Space no los toca.

### 2. Lifecycle Config (`~/.on_start`)

El cómputo del Space es efímero — lo que solo vive en el sistema de archivos
local (no en EFS) se **borra al apagar**. Por eso, lo que no persiste solo se
reinstala en cada arranque vía un script de lifecycle (`~/.on_start` que corre
al prender el Space):

- Reinstalar **Node.js + Open Code CLI** (no persisten solos en el cómputo).
- Configurar **repos de GitHub** (clonar / setear remotes / credenciales).
- Cualquier tool/CLI que se necesite y no viva en EFS.

Regla: **todo lo reusable vive en EFS; todo lo efímero se reinstala en
`~/.on_start`**. Si algo "desaparece" al apagar, es que falta en el lifecycle
script.

### 3. Conexión preferida: VS Code LOCAL vía SSH-sobre-SSM

No trabajar directo en la interfaz web de JupyterLab/Code Editor. En su lugar,
mantener la experiencia local de VS Code (config, extensiones, atajos):

- Habilitar **remote access** en el Space (SSH sobre Systems Manager / SSM).
- En VS Code local: **AWS Toolkit** (para conectar/gestionar el Space) +
  **Microsoft Remote-SSH** (para abrir la sesión remota sobre SSH-SSM).
- Resultado: VS Code local apunta al filesystem remoto (EFS) del Space, con
  tus extensiones y atajos intactos.

### 4. Tamaño de instancia

- Empezar por la mínima útil: **`ml.t3.large` (8 GB RAM)** como referencia —
  suficiente para editar código, correr `openspec`, y pruebas ligeras.
- El Space escala/decide la instancia según la carga; para entrenamiento pesado
  no se usa el Space, se usa cómputo dedicado (ver `compute/blueprint.md`).

### 5. Dos terminales para los agentes

Dos terminales **locales** (paneles de VS Code) conectadas al Space remoto,
una para el agente **Arquitecto** (`arquitecto`) y otra para el agente
**Implementador** (`implementador`) — ver `WORKFLOW.md` y "Rol de esta ventana" en
`AGENTS.md`. Los agentes corren dentro del Space, con el código/contexto en
EFS.

## Ciclo de uso (el hábito central)

1. **Prender el Space** (con el Lifecycle Config reinstalando lo efímero).
2. Trabajar vía VS Code local remoto; los agentes corren en el Space.
3. **Apagar el Space** al terminar — el código/contexto queda en EFS.
4. Próxima sesión: prender y todo sigue igual.

**No dejar el Space prendido por default** (ver `costos/blueprint.md`): es
cómputo que factura por tiempo encendido.

## Todo como código

El Space, el EFS, el Lifecycle Config y los permisos se definen en **Terraform**
(ver `cicd/blueprint.md`), no a mano en consola: así el entorno es reproducible
y versionado, y no hay "quick setup" que cree recursos de red sin avisar (ver
`costos/blueprint.md` Principio 3).

## Parámetros a definir por proyecto (en `project-decisions.md`)

- Nombre del dominio/espacio SageMaker Studio y del Space.
- Tamaño de instancia (default `ml.t3.large`).
- Qué instala/configura el `~/.on_start` además de Node + Open Code CLI.
- Repo(s) de GitHub a clonar y su estrategia de acceso.
- Si se habilita SSH sobre SSM (default: sí, para VS Code remoto).

## Notas de este proyecto

- **VPC obligatoria en SageMaker Studio**: `aws_sagemaker_domain` exige
  `vpc_id` + `subnet_ids` (no es opcional, aun con
  `app_network_access_type = "PublicInternetOnly"`). En `forecast-sinca-aws` se
  reutilizó el default VPC de la cuenta (sin NAT gateway, sin recursos de red
  nuevos) para preservar la intención de bajo costo. El blueprint original
  asumía "sin VPC"; corregir: un VPC es requerido, la decisión real es *cuál*
  VPC (default vs propia).
- **`space_settings.app_type` obligatorio en Spaces privados**: un Space
  privado exige declarar `app_type` explícito (ej. `"JupyterLab"`); no se
  infiere de `jupyter_lab_app_settings`. Sin él, `terraform apply` falla con
  `AppType [null] is not supported for private spaces`.
- **SSH sobre SSM no es un recurso Terraform directo**: el remote access
  (SSM) de Studio se configura sobre el Space al momento de conectarse (VS
  Code + AWS Toolkit), no como recurso Terraform. No requiere una EC2: el
  Space es el cómputo y SSM es solo el túnel de conexión. El rol de dev no
  necesita permisos `ssm` explícitos: el SSH-over-SSM lo gestiona el backend
  de SageMaker.
