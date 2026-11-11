# Protocolo de trabajo

Tres partes: **Arquitecto** y **Implementador** (agentes de opencode, ver `AGENTS.md`) y el **operador**, que decide en última instancia.

## Roles

- **`arquitecto`**: piensa y redacta `proposal.md`, `design.md`, `tasks.md` (`/opsx/propose`) y documentos de referencia. En modo plan no edita ni corre comandos hasta que el operador aprueba; los comandos que necesite van en un bloque de código.
- **`implementador`**: solo ejecuta `tasks.md` (`/opsx/apply`) y archiva (`/opsx/archive`).
- **Operador**: corre los comandos que se le indican, alterna entre agentes y tiene la decisión final.

## Reglas de oro

- **Nada fuera de `tasks.md` se le pide al implementador.** Si surge algo nuevo, vuelve al arquitecto, que lo agrega a `tasks.md` (mismo alcance) o abre un change nuevo (alcance distinto).
- **El ciclo de vida de un change va solo por CLI** (`openspec new change`, `openspec archive`). Nadie crea, mueve ni archiva carpetas a mano. Si pareciera necesario, se pregunta antes.
- **La evidencia se ajusta al contexto.** Si el operador dice que algo está hecho o que cierta evidencia basta, se acepta y no se insiste. Una verificación se pide una sola vez.

## Paso a paso por change

1. **Explorar y planificar** (arquitecto): `/opsx/explore`; se lee `project-decisions.md` y los blueprints; la propuesta se discute como texto, sin tocar disco.
2. **Aprobar y escribir**: con la aprobación del operador, `/opsx/propose "<nombre>"` crea el change y escribe los 3 artefactos. La sesión vuelve a modo plan.
3. **Validar**: `openspec status --change "<nombre>" --json`.
4. **Implementar** (implementador): `/opsx/apply "<nombre>"`, tarea por tarea. Tareas de escritura primero, de ejecución real (terraform, Training Jobs, verificar en AWS) al final; la última es una verificación funcional de punta a punta. Si algo no calza con el blueprint pero no bloquea, se anota en `## Notas de implementación` de `tasks.md`.
5. **Bloqueos**: vuelven al arquitecto, que actualiza `design.md`/`tasks.md` con la aprobación del operador.
6. **Cierre**: `/opsx/sync` y `/opsx/archive` (implementador, con `openspec archive`). Antes, el arquitecto revisa las notas de implementación, pasa lo útil como `## Notas de este proyecto` al final del `modules/<dominio>/blueprint.md` local (sin tocar el cuerpo) y actualiza el `README.md`. Riesgo bajo o medio: basta con las notas. Riesgo alto (IAM nuevo, red, datos sensibles, producción): además se relee el change completo.

## Reglas fijas

- **Repo en GitHub** desde el arranque.
- **`README.md`** refleja el proyecto real y se actualiza en cada cierre.
- **Credenciales** nunca en texto plano: `.env` (gitignored) o Secrets Manager.
- **Perfiles AWS** siempre explícitos; nunca `default` ni el de otra cuenta.
- **IAM** de mínimo privilegio; nunca `*FullAccess` sin justificación explícita.
- **Comandos destructivos o que mutan datos/infra reales** (borrar, `apply`): el arquitecto los entrega en bloque, el operador los corre y se mira el output antes de seguir. Para el resto, vale lo que el operador diga.
- **Fuente de verdad de arquitectura**: `project-decisions.md`. Si algo cambia al implementar, se actualiza ahí, no solo en el `design.md`.

## Plantilla y `modules/`

Los `blueprint.md` son guías genéricas. Lo que un change revela se agrega al final del blueprint local como `## Notas de este proyecto`. Actualizar la plantilla maestra queda fuera de este flujo.

## Arranque de un proyecto nuevo

Copiar la plantilla, crear el repo de GitHub y el `README.md` propio, registrar los skills compartidos, verificar el profile AWS (`PREREQUISITES.md`) y llenar `project-decisions.md` con el arquitecto.

## Estructura del repo

- `project-decisions.md` — arquitectura y decisiones cerradas
- `PREREQUISITES.md` — cuentas y profiles AWS
- `modules/` — blueprints por dominio (guías, no specs)
- `opencode.json` — permisos y agentes
- `.opencode/` — comandos `/opsx/*` y skills
- `openspec/changes/`, `openspec/specs/` — changes y specs consolidadas
