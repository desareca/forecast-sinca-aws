# Proposal: `deep-koopman` (Fase 4, parte 2, change 7.2, exploratorio)

## Why

`dmd-operador-global` (7.1) dejó la precisión en el nivel de la mejor referencia (test: razón 1.030
[0.975, 1.176]), pero sus modos no son identificables entre semillas: solo el diario se repite
(`EXPERIMENT-LOG.md`, 2026-11-20). Sin modos estables no tiene sentido localizar la dinámica, así que
`deep-koopman` va antes que `dmd-localizado` (`project-decisions.md` §8/§9). La pregunta de este
change: ¿un Koopman entrenado de extremo a extremo logra modos identificables entre semillas sin
perder la precisión de 7.1? Razonamiento base: `docs/ae-dmd-propuesta-experimentos.md` §7.

## What Changes

- **Paquete `pipeline/koopman/`**: encoder sobre MP + meteo hasta `t` (GRU, y conv1d somero de la
  misma profundidad), latente único `d = 10`, decoder lineal (principal) o GRU (comparación),
  K global horario diagonal por bloques (diario de 24 h fijo, 2 bloques libres, 4 reales) con
  decaimiento en escala log y techo, y control `B·u` con el calendario conocido.
- **Receta incremental** F1 (reconstrucción + linealidad a 1 paso) → F2 (+ multi-paso 24 h con
  currículo) → F3 (+ RMSE ponderado del promedio 24 h en µg/m³, y variante sobre ICAP_zona con máximo suave), con estabilización desde el inicio.
- **Campaña completa sin ramas** (enmienda del operador tras el re-humo, campaña ~USD 1.5): F1 con
  GRU, conv1d somero, decoder GRU y variantes de K; remedios B1–B4 siempre, como variantes de F1;
  F2, F3 y F3 `icap` para ambos encoders. Criterio de identificabilidad preregistrado como regla de
  selección y reporte, no como compuerta.
- **Selección preregistrada** sobre `sel` en un job propio, registrada antes del cierre; test una
  sola vez sobre la elegida.
- **Evaluación** con el arnés de `pipeline/operador` (mismas filas, bootstrap emparejado, criterio
  D12, diagnósticos, interpretabilidad), contra GRU+DMD+Duan de 7.1, LightGBM y persistencia.
- **Todo el cómputo de experimentos en SageMaker Training Jobs** (CPU, imagen `tf-cpu-v2`); local
  solo tests y edición. Publicación al sqlite compartido desde el job, con los resguardos de 7.1.
- **Orquestación con AWS Step Functions, en serie**: lista fija de jobs, sin ramas ni pausas,
  reanudable por marcador y con checkpoints; avisos por SNS/email y fallos por EventBridge. Infra en Terraform; prueba en modo seco antes del
  lanzamiento real. El humo se lanza a mano.
- Cierre con reajuste a junio y una sola evaluación en test.

## Capabilities

### New Capabilities

- `koopman-entrenado`: Koopman entrenado de extremo a extremo sobre las 6 series de MP: modelo,
  operador estructurado, pérdidas, entrenamiento estable, criterio de identificabilidad, campaña
  completa, selección preregistrada,
  evaluación, confirmación en test e interpretabilidad.
- `orquestacion-experimentos`: ejecución en serie de una campaña de Training Jobs con lista fija,
  selección antes de test, reanudación por job, checkpoints, avisos y modo seco.

### Modified Capabilities

- `tracking-experimentos`: convención de runs del Koopman entrenado y publicación desde Training
  Jobs con control de concurrencia y respaldo.
- `iam`: roles de Step Functions y de la Lambda de decisión; permisos del rol OIDC de infra para
  los recursos nuevos.

## Impact

- Código nuevo: `pipeline/koopman/` (modelo, pérdidas, entrenamiento, inicialización,
  identificabilidad, selección, evaluación, CLI, README), `scripts/koopman/preparar_ejecucion.py`,
  `tests/koopman/`. Infra: `infra/orquestacion-koopman.tf`, ASL, Lambda, ampliación de
  `infra/iam-oidc.tf`. Cambios acotados: `pipeline/train/launch.py` (entry point nuevo),
  `pipeline/train/mlflow_log.py` (publicación con resguardos reutilizable por los jobs y por
  `scripts/mlflow/publish_state.py`). Reutiliza `pipeline/ae/`, `pipeline/operador/` sin cambiar su
  comportamiento.
- AWS: 48 Training Jobs `ml.m5.large`, ~12 h, escritura serializada en
  `sinca-mlflow/_mlflow/`; Step Functions, Lambda, SNS y EventBridge nuevos (IAM nuevo: riesgo alto);
  sin red ni imágenes nuevas. Costo estimado ~USD 1,5
  (orquestación ≈ USD 0; ejecución seca ~USD 0,1; `design.md` D17).
