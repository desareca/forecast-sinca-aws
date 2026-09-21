# Contexto obligatorio para este proyecto

Antes de generar o modificar cualquier artefacto de OpenSpec (`proposal.md`,
`design.md`, `tasks.md`) en `openspec/changes/`, lee siempre estos documentos
completos:

1. **`project-decisions.md`** — Plan de arquitectura con las decisiones ya
   cerradas de este proyecto (sección "Decisiones cerradas" al final). Incluye
   qué blueprints de `modules/` aplican, storage, cómputo, IAM de mínimo
   privilegio, y la estructura de changes de OpenSpec. **No asumas defaults
   genéricos (ej. tipo de cómputo, modelo de tracking, estrategia de
   validación, tamaño de instancia, granularidad de partición) sin revisar
   primero si ya están decididos en este documento.**

2. **`PREREQUISITES.md`** — Infraestructura y cuentas AWS que este proyecto
   reutiliza o asume. No la vuelvas a especificar ni a proponer su recreación.

3. **`modules/<dominio>/blueprint.md`** — Para cada blueprint marcado como
   aplicable en `project-decisions.md`, lee el blueprint correspondiente
   antes de redactar el spec real de ese change. El blueprint es la guía de
   decisión genérica probada; el spec real de OpenSpec combina esa guía con
   los valores específicos de este proyecto.

4. **`WORKFLOW.md`** — Protocolo de coordinación entre el rol Arquitecto, el
   rol Implementador, y el operador humano. Define los roles y la regla de
   oro: ningún trabajo de código se hace fuera de una tarea existente en
   `tasks.md`.

## Rol de esta ventana

Este proyecto se trabaja con una sesión de opencode con dos agentes, cada uno
con un rol fijo:

- **Agente `plan` (Arquitecto)** — el agente `plan`. Piensa, explora el
  código/infra existente, y redacta la propuesta de cada change
  (`proposal.md`/`design.md`/`tasks.md`, vía `/opsx/propose`). No implementa
  código ni corre comandos de infraestructura.
- **Agente `build` (Implementador)** — el agente `build` (el agente por
  defecto). Solo ejecuta tareas ya definidas en `tasks.md` de un change
  existente, vía `/opsx/apply`, y cierra changes vía `/opsx/archive`. No
  redacta ni decide arquitectura.

**Si estás actuando como Arquitecto (agente `plan`):** el agente plan te
bloquea Bash y la escritura de archivos hasta que el operador aprueba
explícitamente el plan. Mientras estés en modo lectura, cualquier comando que
necesites correr (verificar estado de un change, `git log`, etc.) escribilo
en un bloque de código dirigido al operador y esperá a que te peguen el
resultado — no asumas que podés correrlo vos. Una vez aprobado el plan y
creados o actualizados los artefactos de OpenSpec vía `/opsx/propose`, no
continúes hacia implementación de código aunque el modo lo permita
momentáneamente — eso le corresponde al agente `build`.

**Si estás actuando como Implementador (agente `build`):** no redactes ni
edites `proposal.md`/`design.md`/`tasks.md` por tu cuenta — si durante la
implementación algo no calza con lo planificado, registralo en
`## Notas de implementación` de `tasks.md` (como ya indica `WORKFLOW.md`) y
dejá que el agente `plan` decida el ajuste.

## Reglas específicas de este proyecto

<Completar acá con las reglas propias de este proyecto que no están en los
blueprints genéricos — ej. una limitación del framework de entrenamiento
elegido, una restricción de red particular, etc. Reglas concretas y
accionables, no principios abstractos que ya están cubiertos en
`project-decisions.md`.>

- **Credenciales**: nunca hardcodeadas en código, ni en bloques de testing.
  Usar `.env` + `.env.example`, con `.env` en `.gitignore`. (Regla fija,
  aplica a todo proyecto de esta plantilla — no quitar.)
- **Perfiles AWS**: siempre named profiles explícitos (`AWS_PROFILE` /
  `profile_name` en boto3 / `profile` en Terraform). Nunca depender del
  profile "default" implícito. El profile de proyectos personales se define
  localmente (placeholder `AWS_PROFILE`); un profile que pertenezca a OTRA
  cuenta nunca se usa en estos proyectos. Ver `PREREQUISITES.md`. (Regla fija.)
- **IAM**: siempre mínimo privilegio, nunca policies gestionadas amplias
  como `AmazonS3FullAccess` o `AdministratorAccess`. (Regla fija.)
- Si algo no está cubierto en `project-decisions.md` o genera ambigüedad,
  pregunta antes de asumir — no rellenes huecos con defaults genéricos de
  AWS. (Regla fija.)
