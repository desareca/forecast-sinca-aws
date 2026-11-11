# Contexto obligatorio

Antes de tocar `proposal.md`, `design.md` o `tasks.md` en `openspec/changes/`, leer:

1. `project-decisions.md` — decisiones cerradas (§9). No asumir defaults ya decididos.
2. `PREREQUISITES.md` — infra y cuentas reutilizadas; no recrearlas.
3. `modules/<dominio>/blueprint.md` — de cada blueprint que aplique.
4. `WORKFLOW.md` — roles y regla de oro.

## Roles

- **`arquitecto`**: explora y redacta `proposal/design/tasks.md` (`/opsx/propose`). No implementa. Edita y corre comandos solo tras aprobación del operador; mientras tanto deja los comandos en un bloque de código.
- **`implementador`** (por defecto): ejecuta `tasks.md` (`/opsx/apply`) y archiva (`/opsx/archive`). No decide arquitectura. Si algo no calza, lo anota en `## Notas de implementación` de `tasks.md`.

## Reglas del operador

- **Laconicos.** Respuestas y documentos breves.
- **La decisión final es del operador.** Se propone, se le dice qué y dónde antes de escribir, y se sigue lo que decida.
- **Evidencia según contexto.** Si el operador dice que algo está hecho o que cierta evidencia basta (ej. los notebooks ejecutados prueban que corrieron), se acepta y no se vuelve a pedir. Una pregunta o verificación se hace una sola vez. Output real solo antes de borrar o mutar datos o infraestructura reales.

## Reglas fijas

- **Credenciales** nunca hardcodeadas: `.env` + `.env.example`, `.env` en `.gitignore`.
- **Perfiles AWS** siempre explícitos (`--profile` / `profile_name` / `profile`); nunca el `default` implícito ni el de otra cuenta. Ver `PREREQUISITES.md`.
- **IAM** de mínimo privilegio; nunca `*FullAccess` ni `AdministratorAccess`.
- Si algo no está en `project-decisions.md` y es ambiguo, preguntar antes de asumir.
