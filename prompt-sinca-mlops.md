# Prompt para el planificador — Proyecto Predicción Calidad del Aire (SINCA)

Quiero iniciar un proyecto personal de MLOps completo, usando como base el template del
proyecto de plantilla (solo lectura, no modificar).
Cuenta AWS: la personal ya creada (perfil explícito, nunca `default`).

Este es un proyecto más largo y complejo que mis proyectos anteriores porque incluye varios
experimentos de modelado, así que necesito que el plan quede bien definido desde el inicio,
pero sabiendo que algunas decisiones van a cambiar en el camino — cualquier ajuste debe pasar
por el flujo normal de OpenSpec (propuesta de cambio, no reescritura libre).

## 1. Dominio y objetivo

Predicción de calidad del aire (MP2.5/MP10) en la zona saturada de Temuco/Padre Las Casas
(Región de La Araucanía), con horizonte de **24 horas** — mismo horizonte que usa el sistema
oficial del Ministerio del Medio Ambiente (MMA), para poder comparar contra ese benchmark
implícito.

Target derivado: **ICAP** (Índice de Calidad del Aire referido a Partículas), calculado a
partir del promedio móvil de 24h de MP2.5/MP10 aplicando la fórmula oficial (D.S. 12/2011 y
equivalente MP2,5) — no hay que scrapear el estado declarado del MMA, se calcula con los
mismos datos de entrada.

La declaración oficial de alerta/preemergencia/emergencia para la zona saturada
Temuco/Padre Las Casas se hace por el **máximo entre las 3 estaciones representativas**
(Ñielol, Las Encinas, Padre Las Casas II), no por una sola estación ni por promedio:

```
ICAP_zona(t) = max( ICAP_Ñielol(t), ICAP_Las_Encinas(t), ICAP_PLC_II(t) )
```

calculado por separado para MP10 y MP2.5, tomando el peor de los dos contaminantes. Por eso
el pipeline de datos trae las 3 estaciones desde el inicio (ver punto 2) — sin las 3 series
sincronizadas no se puede reproducir ni comparar contra el episodio crítico oficial.

## 2. Alcance de estaciones

**Pipeline de datos: las 3 estaciones activas de la zona saturada desde el inicio** —
Padre Las Casas II (ID SINCA 263, recepción "en línea", mide MP10/MP2.5/gases/meteorología
incl. viento), Ñielol y Las Encinas (ambas en Temuco). Esto es necesario para poder calcular
el ICAP_zona real (máximo entre las 3, ver punto 1) y comparar contra el episodio crítico
oficial — con una sola estación no se puede reproducir ese cálculo.

**Modelado**: puede seguir partiendo simple (ej. primeros experimentos solo con la serie de
Padre Las Casas II), pero se plantea como punto a explorar si las series de las otras 2
estaciones aportan información cruzada útil como features adicionales (ej. un aumento en
Ñielol puede anticipar lo que llega a Padre Las Casas II según el patrón de viento, o
simplemente correlacionar porque comparten la misma fuente de emisión regional) — esto es
una hipótesis a probar en los experimentos, no una decisión cerrada de diseño.

## 3. Fuentes de datos — todas deben ser datasets vivos (obtención 100% automática)

| Variable | Fuente | Acceso | Periodicidad |
|---|---|---|---|
| MP2.5, MP10, y gases donde estén disponibles (NO2, CO) | SINCA — Padre Las Casas II, Ñielol, Las Encinas | Sin API REST — scraping de `apub.htmlindico2.cgi`, mismo mecanismo parametrizado por ID de estación para las 3 | Horaria |
| Temperatura, humedad, presión, precipitación, viento (dir/vel), radiación | SINCA (las 3 estaciones, cobertura de parámetros meteorológicos puede variar entre ellas) | Mismo mecanismo | Horaria |
| Altura de capa límite (boundary layer height) | Open-Meteo API | Gratuita, sin key, JSON | Horaria |
| Feriados | apis.digital.gob.cl/fl/feriados | Gratuita, sin key, JSON | Se consulta 1 vez por corrida |
| Día de semana / ventana GEC (1 abr–15 sep) | Calculado, sin fuente externa | — | Cada corrida |
| ICAP / estado (alerta/preemergencia/emergencia) | Derivado de MP2.5/MP10 propios | Cálculo propio, fórmula oficial | Cada corrida |

Explícitamente **fuera de scope** por no ser automatizable de forma confiable: calendario
escolar Mineduc (no cambia semana a semana, se descartó), y cualquier scraping de comunicados
oficiales del MMA para el estado declarado (se deriva, no se scrapea).

Existe un paquete en CRAN (`AtmChile`) que ya automatiza descargas de SINCA — revisar su
código como referencia antes de construir el scraper desde cero, aunque esté en R.

## 4. Fases (gated — no avanzar sin cerrar la anterior)

1. **Setup**: proyecto desde el template, infra Terraform (Space, MLflow self-hosted en
   SQLite+S3, buckets), sin lógica de negocio.
2. **Pipeline de datos**: scraper SINCA parametrizado + Open-Meteo + feriados + cálculo ICAP,
   con validación de esquema/rangos (pandera) antes de persistir. Smoke test manual antes de
   automatizar el schedule.
3. **Baseline**: persistencia (predicción = último valor) + modelo simple (ej. LightGBM con
   lags + meteorología). Define el piso a superar.
4. **Experimentos avanzados**: incluye la línea de autoencoder + DMD/Koopman. Convención de
   nombres de experimento en MLflow: `sinca-{estacion}-{modelo}-v{n}`.
5. **Evaluación**: comparar contra baseline y contra la referencia pública (~70% ICAP RM,
   WRFChem 2017-2022 — la única vara pública, aunque sea de otra región).
6. **Productivización**: reentrenamiento semanal, inferencia diaria a 24h de horizonte,
   versión mínima (sin autoscaling ni endpoints persistentes).

## 5. Estrategia de evaluación y métricas (idea inicial, no definitiva — puede ajustarse en el proceso)

**Target del pipeline: 6 series continuas** (MP2.5 y MP10 para cada una de las 3 estaciones),
no ICAP directamente. El `ICAP_zona` se deriva siempre en post-proceso:

```
ICAP_final_estación(t) = max( ICAP(MP25_estación(t)), ICAP(MP10_estación(t)) )
ICAP_zona(t)            = max( ICAP_final_Ñielol(t), ICAP_final_LasEncinas(t), ICAP_final_PLC_II(t) )
```

La transformación PM→ICAP es piecewise-lineal (continua, diferenciable salvo en los
breakpoints), así que en principio permitiría usarse como objetivo de entrenamiento — pero el
criterio adoptado es no hacerlo por defecto (ver más abajo), para mantener un criterio común
entre todos los modelos de la escalera.

**Criterio de loss por tipo de modelo:**

- **Baseline**: sin entrenamiento, no aplica ninguna decisión de loss.
- **Holt-Winters / SARIMA**: se ajustan de forma independiente por serie (método estadístico
  clásico, no admite una loss custom basada en el máximo de varias series).
- **LightGBM**: mismo criterio — 6 modelos independientes (uno por serie de MP2.5/MP10 por
  estación), cada uno minimizando su propio RMSE ponderado. No se explora loss conjunta acá:
  requeriría boosting coordinado entre los 6 boosters, complejidad alta para ganancia incierta.
- **AE+DMD**:
  - **Variante base** — mismo criterio que LightGBM: RMSE ponderado independiente por serie.
    Es la que se compara 1 a 1 contra el resto de la escalera.
  - **Variante experimental (loss conjunta)** — a probar como experimento adicional dentro de
    la línea AE+DMD, no como reemplazo de la variante base:
    ```
    loss_total = Σ RMSE_ponderado(serie_i) + λ · loss(max_predicho, ICAP_zona_real)
    ```
    El objetivo es ver si guiar el entrenamiento hacia el agregado (ICAP_zona) mejora el
    recall en las clases críticas, aunque sacrifique algo de precisión en las series
    individuales. Es posible porque AE+DMD es una red neuronal con loss arbitraria — no
    aplica a LightGBM ni a Holt-Winters/SARIMA por las limitaciones de cada uno.

**RMSE ponderado por serie** (loss de las variantes base), no MSE plano — porque las
categorías críticas (Alerta/Preemergencia/Emergencia) están desbalanceadas frente a
Bueno/Regular, y un MSE plano deja que el modelo optimice bien los días "normales" e ignore
los episodios raros que son justamente el objetivo del proyecto. Se aplica igual a cada una
de las 6 series:

```
RMSE_ponderado = sqrt( Σ w_i · (y_i − ŷ_i)² / Σ w_i )
```

Idea de peso continuo (a evaluar, no cerrado): `w_i = 1 + α · max(0, y_i − umbral_regular)` —
preferible a un peso fijo por categoría ICAP porque Emergencia probablemente tenga muy pocos
casos históricos para estimar bien un peso por frecuencia.

**Métricas a loguear por experimento en MLflow (planteamiento inicial):**
- `train_loss` — RMSE ponderado por serie (o loss conjunta, en la variante experimental de
  AE+DMD), lo que realmente optimiza el modelo
- `mse`, `rmse` por serie — sin ponderar, como diagnóstico de error absoluto
- Bloque ICAP oficial, calculado sobre `ICAP_zona` en post-proceso:
  `recall_alerta`, `recall_preemergencia`, `recall_emergencia` — para comparar contra el
  sistema del MMA, aunque se sepa que puede haber pocos casos de algunas categorías en el
  histórico
- Bloque percentil propio (top 5%/10% de días más críticos según `ICAP_zona` histórico, no
  umbrales ICAP nacionales): `recall_top10pct`, `recall_top5pct`, `pr_auc_top10pct` — más
  robusto al desbalance porque no depende de que ICAP tenga suficientes casos por nivel

Criterio de selección tentativo entre runs: RMSE ponderado (promedio o peor serie, a definir)
+ bloque de percentiles propios como principales; bloque ICAP oficial como reporte
comparativo, no como decisor. Esto es un punto de partida para discutir con el planificador,
no una regla fija — ajustar según lo que muestren los primeros experimentos (ej. si el
histórico resulta tener muy pocos días extremos, puede que ni el enfoque de percentiles sea
estable y haya que repensarlo).

## 6. Convenciones de trabajo

- Un `EXPERIMENT-LOG.md` en el repo (qué se probó, hipótesis, resultado, siguiente paso) —
  no depender solo de MLflow para el "por qué".
- Cambios de alcance o diseño durante el proyecto (agregar estaciones, cambiar horizonte,
  nuevas variables) se manejan como propuestas de cambio OpenSpec, no como edición directa
  del plan inicial.

## 7. Pedido al planificador

Antes de generar el plan/spec inicial: explorá el template de referencia para confirmar
estructura y blueprints disponibles, y preguntame cualquier cosa específica de este dominio
(SINCA, ICAP, MLflow, autoencoder+DMD) que no puedas inferir del template genérico.
