# Protocolo de trabajo — <Nombre del proyecto>

Este documento define cómo se coordinan las tres partes de este proyecto:
**rol Arquitecto**, **rol Implementador** (ambos agentes de opencode, ver
"Rol de esta ventana" en `AGENTS.md`), y el operador humano ejecutando
comandos en terminal. Está pensado para evitar el problema clásico de
trabajar con agentes sin punto de control intermedio: darles instrucciones
sueltas y documentar como "hecho" algo que nunca se verificó.

## Roles

- **Agente `arquitecto` (Arquitecto)** — pensar, decidir, y escribir los artefactos
  de OpenSpec (`proposal.md`, `design.md`, `tasks.md`, vía `/opsx/propose`) y
  documentos de referencia. En modo plan no puede ejecutar Bash ni escribir
  hasta que el operador aprueba explícitamente — cualquier comando que
  necesite lo indica en texto para que el operador lo corra y le pegue el
  resultado.
- **Agente `implementador` (Implementador)** — únicamente implementación:
  `/opsx/apply` sobre un change ya definido, y `/opsx/archive` para cerrarlo.
  No se le piden cambios sueltos fuera de `tasks.md`.
- **Operador (terminal)** — ejecuta los comandos que el agente `arquitecto` o el
  agente `implementador` indiquen (CLI de `openspec`, AWS CLI, git, etc.) y pega el
  resultado de vuelta. Ningún comando destructivo o de creación de recursos
  se da por ejecutado sin ver el output real. Alterna entre los agentes
  `arquitecto` y `implementador` según la fase del bucle de trabajo.

## Regla de oro

**Ninguna tarea que no esté en `tasks.md` se le pide directamente al agente
`implementador` (Implementador).** Si durante la implementación surge algo nuevo (un
paso que faltaba, un ajuste de alcance, algo que se descubre haciendo el
trabajo), el agente `implementador` no lo ejecuta por su cuenta ni lo toma como
instrucción suelta — la pregunta vuelve al agente `arquitecto` (Arquitecto, mismo
mecanismo que "Bloqueos", paso 5 de "Paso a paso por change"). Ahí se
decide, caso por caso, si corresponde:

- **agregar la tarea a `tasks.md` del change activo** (si es parte del
  mismo alcance ya aprobado en `proposal.md`/`design.md`), o
- **generar un change nuevo** (si es un alcance distinto o lo suficientemente
  grande como para merecer su propia planificación) — vía el ciclo completo
  de exploración/propuesta/aprobación del agente `arquitecto`, no scaffoldeado a
  mano.

Recién ahí, con la tarea ya escrita en el `tasks.md` correspondiente, el
agente `implementador` la retoma.

## Regla de oro — ciclo de vida de un change

**El agente `arquitecto` no crea, mueve, ni archiva carpetas de change a mano con
herramientas de filesystem.** El ciclo de vida de un change (scaffold
inicial, archivado) se ejecuta exclusivamente vía comandos `openspec`
(`openspec new change` corrido por `/opsx/propose` en el agente `arquitecto` tras
aprobación, `openspec archive` corrido por el agente `implementador` vía
`/opsx/archive`). El rol del agente `arquitecto` en OpenSpec es a nivel macro:
redactar y revisar el *contenido* de `proposal.md` / `design.md` /
`tasks.md`, actuar como experta y revisora del diseño, y detectar cuándo
algo debería ser un change nuevo — pero siempre a través del CLI, nunca
moviendo carpetas directo. Si en algún caso excepcional pareciera necesario
tocar directamente la estructura de un change por fuera del CLI, **debe
primero preguntar y esperar confirmación explícita** del operador antes de
hacerlo — nunca actuar y avisar después.

## Arranque de un proyecto nuevo con esta plantilla

1. Copiar esta carpeta completa del template a la carpeta del proyecto.
1.5. **Crear el repositorio de GitHub del proyecto nuevo** (en tu cuenta
      personal) y dejarlo como remoto (`git init` + `git remote add
      origin`) — paso fijo, no opcional. No arrancar el bucle de trabajo sin
      esto: los changes de OpenSpec y el código que produce opencode deben
      quedar respaldados en git desde el principio, no agregarse a git recién
      al final.
1.6. **Reemplazar `README.md`** (el copiado de la plantilla describe la
      plantilla, no este proyecto) **por un README específico del proyecto
      nuevo** — qué datos entran, qué hace el pipeline o el modelo, qué
      entrega, estructura del repo, estado actual.
2. Abrir opencode en la carpeta del proyecto y arrancar con el agente `arquitecto`
   (Arquitecto). `AGENTS.md` ya obliga a leer `project-decisions.md`,
   `PREREQUISITES.md` y `WORKFLOW.md` al inicio — no depende de ningún estado
   guardado porque la información real vive en disco y la sesión la relee
   cada vez.
2.5. **Confirmar que los skills compartidos relevantes están registrados en
      la máquina.** opencode no carga un skill compartido solo por estar
      mencionado en un `.md` — tiene que estar en
      `~/.config/opencode/skills/` (skills personales) o en
      `.opencode/skills/` del proyecto mismo. Si no está registrado, copiarlo
      antes de arrancar el bucle de trabajo.
3. Leer `PREREQUISITES.md` y confirmar que la cuenta AWS y el profile
   (`AWS_PROFILE`) están activos y apuntan a la cuenta correcta, no a la de
   terceros.
4. Conversar con el agente `arquitecto` para llenar `project-decisions.md`,
   incluyendo qué blueprints de `modules/` aplican.
5. Recién ahí empieza el bucle de trabajo normal (abajo), un change por
   blueprint seleccionado (o agrupando blueprints relacionados en un mismo
   change, a criterio conjunto).

## El bucle de trabajo

```
┌──────────────┐  comandos openspec    ┌──────────────┐
│  Arquitecto   │ ───────────────────▶  │   Operador    │
│ (agente arquitecto) │ ◀─────────────  │  (terminal)   │
└──────┬────────┘   resultado/output     └──────┬────────┘
       │ aprobado → /opsx/propose                │
       ▼                                         │
 proposal.md / design.md / tasks.md              │
       │                                         │
       │              /opsx/apply                ▼
       └───────────────────────────▶  ┌──────────────┐
                                        │ Implementador │
                                        │ (agente implementador)│
                                        └──────┬────────┘
                                               │ se traba / duda
                                               ▼
                                     vuelve al agente arquitecto
```

## Paso a paso por change

1. **Exploración + Planificación** (agente `arquitecto`): arranca con
   `/opsx/explore` para pensar en voz alta el change — investigar el
   código/infra existente, comparar opciones, aclarar el problema — sin
   presión de llegar a un artefacto formal todavía. Cuando el diseño
   cristaliza, lee el/los `blueprint.md` del dominio y `project-decisions.md`,
   y arma la propuesta de `proposal.md`/`design.md`/`tasks.md` como texto en
   la conversación — todavía sin tocar disco. El operador revisa y pide
   ajustes ahí mismo, las veces que haga falta, antes de aprobar nada.
2. **Aprobación y escritura**: cuando el contenido queda conforme, el
   operador aprueba el plan (el agente sale del modo lectura para esa acción
   puntual) y el agente `arquitecto` corre `/opsx/propose "<nombre>"`, que crea el
   change vía CLI (`openspec new change`) y escribe los 3 artefactos en un
   solo paso. Al terminar, la sesión vuelve a quedar en modo plan — no
   continúa hacia implementación de código aunque técnicamente pudiera.
3. **Validación** (agente `arquitecto` o terminal): correr
   `openspec status --change "<nombre>" --json` para confirmar que todo
   quedó `ready`/`done` según corresponda, antes de pasar al agente `implementador`.
4. **Implementación** (agente `implementador`): se corre `/opsx/apply "<nombre>"`.
   El agente `implementador` ejecuta las tareas de `tasks.md` una por una,
   distinguiendo dos tipos de tarea:
   - **Escritura de código/archivos**: no requiere acceso a AWS.
   - **Ejecución real** (terraform apply, disparar un Training Job, correr el
     pipeline, verificar en AWS): ejecutarla según `PREREQUISITES.md` (via el
      profile `AWS_PROFILE`), siempre con el operador corriendo el comando y
     devolviendo el output real.
   - Al planificar `tasks.md` de cada change, ordenar las tareas para que
     todas las de escritura vayan primero y las de ejecución real queden
     agrupadas al final (o claramente marcadas). **La última tarea del
     grupo de ejecución real SHALL ser siempre una verificación funcional
     explícita** — correr el pipeline una vez de punta a punta y confirmar
     con output real (una query con resultados, un log de ejecución exitosa),
     no dar el change por terminado solo porque el código se desplegó sin
     errores de build/apply.
   - Si durante una tarea el agente `implementador` encuentra algo que no calza con
     el blueprint del dominio (un supuesto incorrecto, un caso no cubierto,
     algo que tuvo que hacer distinto) pero **no es un bloqueo** — no
     necesita una decisión para seguir —, lo registra como una línea en una
     sección `## Notas de implementación` al final de `tasks.md`, con la
     tarea, qué encontró, y qué hizo. Esto no cuenta como salirse de la
     tarea ni edita `modules/` directamente — esa evaluación queda para el
     Cierre.
5. **Bloqueos**: si el agente `implementador` se detiene (ambigüedad, error, decisión
   de diseño) —sea porque se trabó solo o porque el operador le pidió
   frenar—, esa pregunta vuelve al agente `arquitecto` (Arquitecto). Ahí se decide
   y se actualiza `design.md`/`tasks.md` (siguiendo el mismo patrón de
   aprobación del paso 2: plan → aprobación → escritura).
6. **Cierre**: `/opsx/sync` y `/opsx/archive`, desde el agente `implementador`. El
   archivado **siempre** se hace con `openspec archive` (CLI real) — nunca
   moviendo la carpeta del change a mano. Antes de archivar, el agente `arquitecto`
   revisa la sección `## Notas de implementación` de `tasks.md` y cualquier
   hallazgo que el agente `implementador` haya registrado. De ahí, evalúa cuáles
   valen la pena preservar sobre el blueprint usado, y lo que corresponda lo
   agrega como nota al final de la copia **local** de
   `modules/<dominio>/blueprint.md` (sección `## Notas de este proyecto`,
   sin tocar el cuerpo original). **Como parte del mismo Cierre, el agente
   `arquitecto` actualiza el `README.md` del proyecto** para reflejar lo que este
   change agregó o cambió (nueva infra, cambio de estado de "en progreso" a
   "en producción", etc.) — el README no se actualiza solo al final del
   proyecto completo, sino en cada Cierre.

## Revisión en el Cierre

El flujo no tiene hooks automáticos: la revisión del trabajo del agente
`implementador` es responsabilidad del agente `arquitecto` en el Cierre de cada change. El
agente `implementador` registra en `## Notas de implementación` de `tasks.md`
cualquier cosa que no calce con el blueprint del dominio o con lo planificado
(permiso IAM de más, valor hardcodeado, supuesto incorrecto, caso no
cubierto). En el Cierre, el agente `arquitecto` revisa esas notas:

- Para la mayoría de los changes (riesgo bajo/medio) basta con esa revisión
  de notas.
- Para changes de **riesgo alto** (IAM nuevo, cambios de red, acceso a datos
  sensibles, cualquier cosa que toque producción), el agente `arquitecto` hace
  además una revisión manual explícita releyendo el change completo, no solo
  las notas.

La coordinación entre roles es manual: el operador alterna entre el agente
`arquitecto` y el agente `implementador` según la fase (planificación → implementación →
cierre). No hay aviso automático de coordinación — el operador decide cuándo
pasar de un rol al otro y qué tan a fondo revisar.

## Reglas fijas (no negociables por proyecto)

- **Repositorio GitHub**: todo proyecto nuevo con esta plantilla tiene su
  propio repositorio, creado en el arranque (ver paso 1.5 arriba) — no se
  posterga hasta tener código que subir.
- **README.md del proyecto**: refleja el proyecto real, nunca el texto de
  la plantilla sin editar (ver paso 1.6 arriba). Se actualiza al cerrar
  cada change (ver paso 6 de "Paso a paso por change").
- **Credenciales**: nunca en texto plano en ningún archivo, ni en bloques de
  "testing local". Siempre `.env` (gitignored) o Secrets Manager.
- **Comandos que mutan infraestructura o datos reales**: el agente `arquitecto` los
  entrega en bloques explícitos, el operador los corre, y **siempre se pide
  el output real antes de asumir que algo se creó/borró/ejecutó correctamente**
  — no se da nada por hecho solo porque la documentación lo dice. Que una
  tarea ya esté aprobada en `tasks.md` no exime este paso: el deny list de
  `opencode.json` cubre los casos más destructivos, pero no todo lo que muta
  estado.
- **IAM**: mínimo privilegio siempre — nunca policies `*FullAccess` salvo
  que se justifique explícitamente.
- **Perfiles AWS**: siempre named profiles explícitos (`AWS_PROFILE`). Nunca
  el profile `default`, y nunca un profile que pertenezca a otra cuenta.
- **Fuente de verdad de arquitectura**: `project-decisions.md`. Si algo
  cambia durante la implementación, se actualiza ese documento también, no
  solo `design.md` del change activo.

## Retroalimentación hacia la plantilla maestra

Durante el proyecto, la copia **local** de `modules/` (dentro de la carpeta
de este proyecto) puede acumular ajustes: cuando un change revela algo que
el blueprint genérico no cubre bien, el agente `arquitecto` lo agrega como una
sección `## Notas de este proyecto` al final del `blueprint.md` local
correspondiente — sin editar el cuerpo original del patrón. Esto permite
dejar constancia de la lección en el momento, sin arriesgar contaminar la
plantilla maestra con algo que todavía no se sabe si generaliza a otros
proyectos.

**La actualización de la plantilla maestra queda fuera de este flujo
automatizado.** No la hace ninguno de los agentes de este proyecto — la
plantilla maestra es un recurso compartido por todos los proyectos futuros,
y su revisión se hace a propósito, caso por caso, no como parte del cierre de
un proyecto puntual.

## Git, IaC, y detalles operativos de este proyecto

El repositorio de GitHub en sí ya se creó en el arranque (paso 1.5, regla
fija — ver arriba). **IaC es Terraform por defecto** (regla fija de esta
plantilla — ver `modules/cicd/blueprint.md`). Lo que sigue sin default único,
a definir acá por proyecto: la estrategia de ramas (¿solo `main`? ¿feature
branches por change?), y el mecanismo de sincronización entre máquinas si
aplica (dev environment remoto + máquina local). No copiar los detalles de
otro proyecto sin confirmarlos primero.

## Estructura de carpetas de este proyecto

- `project-decisions.md` — arquitectura general (contexto para todos los changes)
- `PREREQUISITES.md` — cuentas AWS y profiles reutilizados
- `WORKFLOW.md` — este documento
- `modules/` — blueprints de dominio por tema (no son specs de OpenSpec)
- `opencode.json` — reglas de permisos de comandos (allow/deny) y
  configuración de agentes
- `.opencode/commands/opsx/` — comandos custom (`/opsx/*`) que orquestan el
  ciclo de OpenSpec (propose, apply, archive, explore, sync)
- `.opencode/skills/` — skills de OpenSpec (misma lógica que los comandos,
  invocables por nombre)
- `openspec/changes/` — changes activos y archivados
- `openspec/specs/` — especificaciones consolidadas (fuente de verdad post-archive)
