## ADDED Requirements

### Requirement: Rol de ejecución de la task Fargate del scraper

El sistema SHALL definir un rol IAM para la task Fargate del scraper (`sinca-scraper-task-role`) con acceso S3 acotado a `sinca-data/*` (GetObject/PutObject/ListBucket) y logs en su log group propio, sin más permisos.

#### Scenario: Acceso acotado al bucket de datos
- **WHEN** la task Fargate del scraper accede a S3
- **THEN** solo puede leer/escribir en `sinca-data/*`, no en otros buckets ni prefijos

#### Scenario: Sin permisos IAM ni políticas amplias
- **WHEN** se inspecciona la política del rol del scraper
- **THEN** no contiene permisos IAM ni managed policies `*FullAccess` / `AdministratorAccess`
