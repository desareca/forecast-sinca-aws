# Blueprint: costos

**Cuándo usar este blueprint**: siempre (transversal). Estos criterios se
repiten en cualquier proyecto personal en AWS. No es una calculadora — es un
checklist de hábitos que evitan sorpresas en la factura a fin de mes.

## Principio 1: evitar servicios que quedan corriendo 24/7 por defecto

Algunos servicios facturan por **tiempo encendido**, no por uso. Si un
proyecto personal los deja corriendo, la factura crece aunque no se esté
trabajando en nada:

- **RDS** (y cualquier base de datos serverless *de pago por instancia*):
  una instancia de RDS cobra por hora todo el mes. Para bajo volumen, preferir
  **SQLite + S3**: el archivo se descarga al cómputo, se opera, se sube de
  vuelta. Cero costo de "estar encendido". RDS solo si hay un requisito real
  de concurrencia/red que SQLite no puede cumplir.
- **Endpoints de hosting** (SageMaker endpoints, cualquier servicio de
  inferencia alojado): un SageMaker endpoint factura por la instancia 24/7.
  Para proyectos personales, preferir **batch transform** o inferencia
  on-demand (una tarea que corre y termina) en vez de un endpoint siempre
  desplegado. Solo desplegar un endpoint si de verdad hay un consumidor
  continuo.
- **Tracking servers gestionados / servicios de red que se levantan de
  "quick setup"**: ver Principio 3.
- **Espacios/cómputo de notebook**: ver `dev-environment/blueprint.md` — el
  Space debe **apagarse** cuando no se usa.

## Principio 2: preferir opciones que se prenden/apagan solas

Para cargas esporádicas (que es el caso típico de un proyecto personal de
datos):

| En vez de… | Preferir… |
|---|---|
| Instancia EC2 fija / RDS siempre encendido | SQLite+S3, Lambda, Fargate on-demand |
| Endpoint de SageMaker 24/7 | Batch transform o Training Job (corre y termina) |
| Instancia de entrenamiento fija | SageMaker Training Job (paga solo el tiempo de tren) |
| Notebook server siempre levantado | Space que se apaga cuando no se usa |

La regla mental: **el cómputo se cobra por el tiempo que está prendido; si no
lo apagás, pagás por dormir.** Toda pieza de infra debería tener una respuesta
a "¿quién lo apaga cuando nadie lo usa?".

## Principio 3: evitar "quick setup" que crea recursos de red sin avisar

Ciertos asistentes de AWS crean recursos adicionales de red (VPCs, subnets,
NAT Gateways, security groups, endpoints) como parte del "quick setup" sin
advertirlo claramente. Caso conocido: **SageMaker Unified Studio / Studio
classic** al hacer un setup rápido puede levantar infraestructura de red
(VPC + NAT gateway) que factura por hora aunque no se use.

Contramedida:
- Preferir Terraform declarado como código (ver `cicd/blueprint.md`) antes que
  asistentes de consola — así lo que se crea es explícito y versionado, no
  "lo que el wizard decidió por mí".
- Si se usa un asistente, revisar después en **VPC > NAT Gateways** y en el
  detalle de la cuenta si no quedó algo facturando.
- Un NAT Gateway (\~$0.045/h + tráfico) dejado encendido un mes es un costo
  silencioso clásico. Sospechar de él si la factura sube sin trabajo aparente.

## Principio 4: cost-aware por diseño, no después

La decisión de costo no es algo que se "revisa al final" — se incorpora en el
momento de elegir cómputo/alojamiento (ver `compute/blueprint.md`), y
`project-decisions.md` exige una **estimación de costo mensual aproximado**
por pieza. No hace falta precisión contable; un número de referencia alcanza
para detectar opciones desproporcionadas.

## Checklist final (responder por cada servicio que se introduce)

- [ ] ¿Queda corriendo si no lo apago? ¿Quién/qué lo apaga?
- [ ] ¿Hay una alternativa on-demand equivalente?
- [ ] ¿El "quick setup" creó VPC/NAT/endpoints extra que no pedí?
- [ ] ¿Está declarado en Terraform (y por tanto visible/revisable)?
- [ ] ¿Tengo una estimación de costo mensual en `project-decisions.md`?

## Herramientas de control

- **AWS Budgets** / **Cost Anomaly Detection**: configurar un budget en la
  cuenta personal que alerte por email al superar un umbral. Un proyecto
  personal debería apuntar a costos de un solo dígito de USD/mes salvo que
  el entrenamiento lo justifique.
- **Cost Explorer**: revisar periódicamente, no solo cuando llega la factura.
