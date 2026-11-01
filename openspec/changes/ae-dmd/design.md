> **Estado: BORRADOR EN DISCUSIÓN.** La sección "Decisiones heredadas" está
> cerrada (viene de `project-decisions.md` y de Fase 3/3.5). La sección
> "Puntos de ajuste" (PA-n) está **abierta**: cada punto tiene opciones y una
> recomendación del Arquitecto, pendiente de confirmación del operador. Cuando
> un PA se cierre, se mueve a "Decisions" con su racional y se marca
> `CERRADO`. `specs/` y `tasks.md` se generan recién con todos los PA cerrados.

## Context

Fase 4 del proyecto `forecast-sinca-aws`. Lo que hay hoy:

- **Datos**: 6 series horarias validadas (MP2.5/MP10 × Ñielol, Las Encinas,
  Padre Las Casas II) + meteorología SINCA por estación (temp, rhum, pres,
  wspd; `glob` falta en Las Encinas; `wdir` en ninguna) + BLH de Open-Meteo +
  feriados. Histórico 2021-01-01 → 2026-10-04 (~50 mil horas por serie).
  Dataset chico para estándares de redes neuronales.
- **Baseline (Fase 3)**: persistencia + 6 LightGBM independientes, target
  `mean(MP(t+1 … t+24h))`, RMSE ponderado (`α=1`, umbral ICAP 100), ventana
  deslizante 2a/1m/1m (42 ventanas) + test held-out `2026-07-01 → 2026-10-04`.
  Resultado: LightGBM gana en 4/6 series, **pierde en PLC II** (pm10/pm25).
- **Análisis (Fase 3.5)**: cada run persiste `predictions.parquet`,
  `feature_importance.parquet`, `zone_icap.parquet`, `rolling_scores.parquet`;
  el dashboard Streamlit ya agrupa por familia de modelo (`model_family`,
  lee tag/param `modelo`) y por versión.
- **Código reutilizable**: `features/build.py` (carga, grilla horaria,
  ventanas, held-out), `evaluate/evaluate.py` (métricas, `zone_icap`,
  `icap_array`), `train/mlflow_log.py` (sync SQLite+S3), `train/launch.py`
  (Estimator SDK v2).

## Goals / Non-Goals

**Goals:**
- Implementar la línea AE+DMD/Koopman y compararla **1 a 1** con persistencia
  y LightGBM: mismo test held-out, mismo target evaluado, mismas métricas.
- Entender **de dónde viene** la ganancia (o la falta de ella) mediante una
  escalera interna corta (PA-1).
- Probar la hipótesis de **información cruzada entre estaciones** (prompt §2)
  de forma natural: el estado latente es regional.
- Variante experimental de **loss conjunta** orientada a `ICAP_zona`.
- **Interpretabilidad de primer orden** (definido por el operador,
  2026-10-30): no es un "plus" sino un objetivo al nivel de la precisión. Los
  modos de la dinámica deben ser **representativos del momento** de cada
  predicción, para poder visualizar el efecto de cada modo y usarlo para
  interpretar predicciones concretas (ver "Interpretabilidad" y PA-14).
- Mantener el patrón operativo: Training Job on-demand, disparo manual,
  logging directo a MLflow, artefactos de diagnóstico compatibles con el
  dashboard.

**Non-Goals:**
- Comparación contra la referencia pública MMA/WRFChem (Fase 5 `evaluacion`).
- SageMaker Pipelines / orquestación automática del entrenamiento (Fase 6).
- Nuevas fuentes de datos (p. ej. pronósticos meteorológicos históricos):
  si se decide que hacen falta (PA-2), van en un change de `data-pipeline`.
- Inferencia diaria / reentrenamiento (Fase 6).
- Re-entrenar el baseline LightGBM (salvo lo que decida PA-11).

## Decisiones heredadas (cerradas, no se re-discuten)

1. **Target evaluado**: `mean(MP(t+1 … t+24h))` por serie; `ICAP_zona` siempre
   en post-proceso con `evaluate.zone_icap`. Aunque el modelo produzca una
   trayectoria horaria, se evalúa sobre ese promedio.
2. **Test held-out** `2026-07-01 → 2026-10-04`, nunca tocado en tuning. Es el
   número "oficial" de la comparación.
3. **Métricas**: `train_loss`, `mse`/`rmse`/`weighted_rmse` por serie, bloque
   ICAP oficial (reporte), bloque percentil propio (decisor junto al RMSE
   ponderado). Se reutiliza `evaluate_predictions` sin cambios.
4. **RMSE ponderado** `w_i = 1 + α·max(0, y_i − umbral_regular)`, `α=1`,
   umbral ICAP 100 — mismo que el baseline para la variante base.
5. **Cómputo**: SageMaker Training Job on-demand, disparo manual, código por
   `source_dir`/`entry_point` desde `pipeline/`, imagen ECR solo-runtime.
6. **Tracking**: MLflow logging directo (SQLite+S3, `sinca-mlflow/_mlflow/`),
   una corrida a la vez (SQLite sin concurrencia), convención
   `sinca-{estacion}-{modelo}-v{n}` + tag `pollutant`.
7. **IAM**: se reutiliza `sinca-training-role` (S3 read `validated/*`, r/w
   `_mlflow/*` y `model-artifacts/*`). Sin managed policies amplias.
8. **Región/profile**: `us-east-1`, `AWS_PROFILE` explícito.

## Puntos de ajuste (abiertos)

### PA-1 — Formulación y escalera interna de la línea AE+DMD

**Pregunta**: ¿qué variantes de "autoencoder + DMD/Koopman" se implementan, y
en qué orden?

| Opción | Qué es | Pros | Contras |
|---|---|---|---|
| **A. Hankel-DMD (sin red)** | Embedding de retardos de las 6 series (matriz de Hankel), DMD exacto con truncamiento de rango; rollout lineal 24 pasos | Barato (numpy, segundos), interpretable, da el "piso" de la línea; si ya compite con LightGBM, es un hallazgo | Lineal: no captura saturaciones ni no linealidades meteo |
| **B. AE + DMD en dos etapas** | Entrenar AE por reconstrucción y luego ajustar DMD (mínimos cuadrados) sobre el latente | Simple de implementar | El latente no se optimiza para ser lineal → suele rendir mal; aporta poco como peldaño |
| **C. Deep Koopman end-to-end** | Encoder φ → latente z; operador lineal K entrenable (`z_{t+1} = K z_t`, opcionalmente `+ B u_t`); decoder ψ. Loss = predicción multi-step + reconstrucción + linealidad del latente (estilo Lusch et al. 2018) | Es la formulación "de verdad" de la línea; latente diseñado para dinámica lineal; multi-step nativo | Más hiperparámetros, más costo, riesgo de overfit con ~50k horas |

**Recomendación inicial (2026-10-30, pre-discusión de interpretabilidad)**:
**A → C**, saltando B. A es casi gratis y diagnostica si la dinámica lineal en
retardos ya explica la señal; C es la variante que se compara oficialmente
(`koopman` como familia de modelo). B solo si C no converge.

**Sub-decisiones de C propuestas (sin confirmar)**:

| Pieza | Opciones | Propuesta |
|---|---|---|
| Encoder (historia L h → z) | MLP sobre ventana aplanada / GRU / Conv1D | MLP primero; GRU si se queda corto |
| Operador K | Denso libre / **bloques 2×2 de rotación con radio ≤ 1** / red auxiliar de autovalores (Lusch) | Bloques 2×2 acotados (estable, período y decaimiento legibles por bloque) |
| Control `B·u_t` | Sin / con | Con (depende de PA-2) |
| Decoder (z → observables) | No lineal / **lineal** | Ver abajo |
| Loss | Predicción multi-step + linealidad latente + reconstrucción | Las tres (ponderación en PA-12) |

**Variante agregada: C con decoder lineal** (encoder no lineal, K lineal,
decoder lineal). Cada modo recupera una forma φ_j en el espacio de las 6
series (magnitud por estación/contaminante y fase = desfase entre estaciones)
y la descomposición del pronóstico por modo vuelve a ser **exacta** (en
espacio `log1p`). La no linealidad queda en *cómo se detecta el estado*. Con la
interpretabilidad como objetivo de primer orden, es candidata seria a modelo
oficial.

**K entrenable y global, no recalculado por batch**: con K como parámetro
compartido, el batch es solo la muestra del optimizador y K representa todo el
train. La alternativa "K por mínimos cuadrados dentro de cada batch" (capa
DMD, estilo Takeishi 2017) da un K distinto por batch y obligaría a re-estimar
K global al final para interpretar; descartada salvo razón nueva.

**Pendiente**: la discusión de interpretabilidad (sección siguiente) mostró
que **un K global sobre años promedia la dinámica** y los modos pueden no ser
representativos de cada predicción. PA-1 se reformula en función de lo que se
elija en **PA-14**; "A" probablemente se reemplace por una variante local
(p. ej. A1/A2 del catálogo) y "C" por una variante con dinámica condicionada.

**Estado**: abierto — **bloqueado por PA-14**.

### Interpretabilidad — contexto (discusión 2026-10-30, sin decisiones)

Contexto para PA-1, PA-5, PA-10 y PA-14. Nada de esta sección está decidido.

**Qué da un modelo lineal en algún espacio.** `x(t+k) ≈ Σ_j b_j · φ_j · λ_j^k`.
Por modo j:
- **λ_j** (autovalor): ángulo → **período** (24 h = ciclo diario
  calefacción/inversión; 168 h = semanal); módulo → **decaimiento/persistencia**
  (|λ| = 0.97/h ≈ vida media 23 h).
- **φ_j** (forma): magnitud = cuánto participa cada serie; **fase = desfase
  entre estaciones** (p. ej. Ñielol 2 h antes que PLC → evidencia, no prueba,
  de transporte; prueba directa de la hipótesis del prompt §2).
- **b_j(t)** (amplitud): cuánto está excitado el modo en cada hora.
- Permite **descomponer aditivamente** un pronóstico ("de 80 µg/m³, 45 del modo
  lento, 25 del diario, 10 del semanal"); LightGBM solo da qué inputs importan,
  no qué dinámica los conecta.

**Interpretabilidad por variante (con K global):**

| | Ritmos (período, decaimiento) | Forma por estación / desfase | Descomposición del pronóstico | Qué inputs excitan cada modo |
|---|---|---|---|---|
| A. Hankel-DMD | Directo | Directo | Exacta | Indirecto (vía historia) |
| C, decoder lineal | Directo (bloques 2×2) | Directo | Exacta (en log1p) | Attribution sobre encoder |
| C, decoder no lineal | Directo | Aproximado (ablación de modos / Jacobiano del decoder, dependiente del estado) | Aproximada (no suma exacto) | Attribution sobre encoder |
| LightGBM (ref.) | — | — | — | Feature importance global |

Caveats: (1) con `log1p`+z-score los modos son **multiplicativos** en µg/m³;
(2) modos no ortogonales pueden cancelarse (amplitudes aisladas engañan);
(3) rango alto → modos espurios (filtrar por energía y estabilidad);
(4) se interpreta **el modelo, no la física**; (5) en Hankel-DMD, los últimos
modos pueden actuar como **forzante intermitente (HAVOK)** con picos antes de
eventos extremos — hipótesis a mirar. La matriz de control B (si existe) es
lineal y legible: cómo cada input (feriado, hora, GEC, meteo) empuja cada modo.

**Tres escalas de tiempo que no hay que mezclar:**

| Escala | Pregunta | Quién la fija |
|---|---|---|
| 1. Ventana de **estimación** | ¿Con qué datos se estima K (y los modos)? | Esquema de entrenamiento |
| 2. Horizonte de **validez** | ¿Hasta dónde se le exige a K ser correcto? | Rollout (24 pasos) + lookback L |
| 3. Ventana de **lectura** | ¿Sobre qué período miro qué modos se activan? | Libre: hora, episodio, temporada |

- **Estimación**: en A, mínimos cuadrados sobre toda la ventana de train (2
  años). En C con K entrenable global, el batch solo es la muestra del
  optimizador; K representa todo el train. → En ambos, con K global, **los
  modos son la dinámica promedio de 2 años**.
- **Validez**: K entrenado a 24 pasos restringe bien períodos de horas a ~1
  día; el semanal (168 h = 7× el rollout) queda poco restringido y el anual
  fuera de alcance (aparece como modo casi estático o lo absorbe el control de
  calendario). Interpretable como modo: **horas a pocos días**; lo
  semanal/anual se lee vía B. Hacer el semanal un modo genuino exige L o
  rollout ~168 h (costo + inestabilidad). → **L y rollout dejan de ser solo
  hiperparámetros de precisión** (impacta PA-5).
- **Lectura**: b_j(t) existe en cada origen; agregable por episodio, mes,
  categoría ICAP. Independiente de las otras dos.
- **Rolling**: cada ventana re-estima K → 12-42 versiones de los modos.
  Propuesta: interpretación "oficial" = modelo final (train hasta el inicio
  del test); estabilidad de modos entre ventanas = chequeo de realidad.

**Preocupación del operador (motiva PA-14).** Con K global sobre varios años la
dinámica se promedia y **los modos no son representativos** de cada
predicción: solo varían las amplitudes, no el significado de los modos.
Físicamente, una inversión térmica **cambia la tasa de decaimiento** (menos
ventilación → el episodio dura más); un K global no puede expresarlo y lo
disfraza de amplitudes, distorsionando la explicación. La pregunta pasa a ser
**cómo hacer que los modos dependan de las condiciones del momento**.

### PA-14 — Localización de la dinámica (catálogo, lluvia de ideas)

**Estado: abierto — catálogo sin preselección.** Se filtra en la próxima
sesión. Agrupado por *dónde se pone la variabilidad* que hoy se pierde al
promediar.

**Grupo A — Modos locales en el tiempo (re-estimar con datos cercanos)**

| # | Idea | Lectura de una predicción | Costo | Riesgo |
|---|---|---|---|---|
| A1 | DMD en ventana deslizante dura (últimos N días) | Modos del régimen reciente | Bajo | Ruido con N chico; no resuelve semanal |
| A2 | DMD online con factor de olvido (peso exponencial) | Igual, transición suave | Bajo | Elegir factor; lento ante cambios bruscos |
| A3 | Multi-resolution DMD (mrDMD): lentos del registro completo, rápidos en sub-ventanas recursivas | Descomposición **por escala** (estacional global + ciclos de esa semana) | Bajo-medio | Cortes arbitrarios; más de análisis que de pronóstico |
| A4 | DMD con ventana estacional análoga (misma fecha ±15 d de todos los años) | "Modos típicos de esta época" | Bajo | No ve lo específico del año/día |
| A5 | DMD por análogos (k trozos más parecidos al estado actual) | "Situaciones parecidas evolucionaron con estos modos" | Medio | Definir "parecido"; pocos análogos extremos |
| A6 | DMD ponderado por similitud meteorológica (kernel) | Modos de "días con esta meteo" | Medio | Ancho de banda; costo por origen |

**Grupo B — Regímenes discretos (pocos K, se elige cuál aplica)**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| B1 | Regla de calendario (GEC/no-GEC, mes, laboral/finde) | "Modos de invierno, laboral" | Muy bajo | No sigue la física; hay días limpios en GEC |
| B2 | Regla meteorológica física (umbrales BLH/viento/temp: inversión / ventilado / frontal) | "Hoy es día de inversión" | Bajo | Umbrales a mano; transiciones abruptas |
| B3 | Clustering de estados (k-means sobre Hankel o meteo) + DMD por cluster | Regímenes descubiertos | Bajo-medio | Clusters sin sentido físico |
| B4 | Switching linear dynamical system (SLDS/HMM) | Probabilidad de régimen por hora + sus modos | Medio-alto | Identificabilidad; entrenamiento delicado |
| B5 | Recurrent SLDS (switch depende del estado) | Además **cuándo/por qué** se entra a un régimen | Alto | Complejidad; pocos episodios |
| B6 | Mezcla blanda de operadores con gating: `K(t) = Σ g_m(t)·K_m` | Pesos de mezcla + K_m interpretables | Medio | Si M crece o el gating es opaco, se pierde |
| B7 | Operadores por fase de episodio (formación / meseta / disipación) | "Estamos en formación" | Medio | Etiquetar fases; pocos episodios |

**Grupo C — Dinámica que varía continuamente con las condiciones**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| C1 | Koopman con autovalores condicionados por meteo (red auxiliar sobre BLH/viento/temp/rain) | "Con BLH 300 m la vida media es 45 h"; curvas autovalor vs variable | Alto | Red auxiliar caja negra; estabilidad; meteo futura (PA-2) |
| C2 | Koopman bilineal / LPV-DMD: `K(u) = K₀ + Σ u_i·K_i` | Cada variable meteo tiene **su matriz** que modifica la dinámica | **Bajo** (lineal en parámetros, mínimos cuadrados, sin red) | Muchos parámetros; requiere regularización |
| C3 | DMD paramétrico por interpolación (DMD por tramo de BLH, interpolar autovalores) | Curva autovalor vs BLH empírica | Bajo | Pocos tramos; interpolar autovalores complejos |
| C4 | Autovalores dependientes del estado latente (Lusch original) | Dinámica no lineal "continua" | Alto | Menos legible: el latente no tiene nombre |

**Grupo D — K global, variabilidad explicada por forzantes**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| D1 | DMDc con meteo como input: `z' = Kz + Bu` | Contribuciones por input ("+18 µg/m³ por BLH bajo") | Bajo | Modos siguen promediados |
| D2 | HAVOK: lineal + forzante intermitente | El forzante marca **inicio de episodios** | Bajo | Puede no aparecer un forzante limpio |
| D3 | Lineal + residuo no lineal acotado (gray-box) | "85% de la predicción es la parte lineal" | Medio | El residuo puede absorberlo todo |

**Grupo E — Separar escalas antes de modelar**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| E1 | Descomposición previa (STL/wavelets/EMD) + DMD por escala con ventanas distintas | Lento con ventanas largas, rápido con cortas | Bajo-medio | Artefactos en los bordes, justo donde se pronostica |
| E2 | Análisis tiempo-frecuencia (SPOD, espectrograma wavelet) — solo diagnóstico | Qué ritmos dominan en cada época | Bajo | No es modelo |

**Grupo F — K variable en el tiempo, estimado de forma conjunta**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| F1 | DMD variable en el tiempo con penalización de suavidad (K_t ≈ K_{t−1}) | Evolución suave de modos sin elegir ventana | Medio | Muchos parámetros; elegir suavidad |
| F2 | DMD con filtro de Kalman (K como estado con ruido) | Igual + **incertidumbre** por autovalor | Medio | Tuning de ruidos |
| F3 | DMD bayesiano / BOP-DMD (bagging) | Intervalos de confianza por modo; combinable con todo | Bajo-medio | No localiza por sí solo |

**Grupo G — Interpretabilidad como capa separada del modelo**

| # | Idea | Lectura | Costo | Riesgo |
|---|---|---|---|---|
| G1 | Linealización local de un modelo no lineal (Jacobiano por estado → K(t) local) | Modos locales para cualquier modelo | Medio | Describe el modelo en un punto, no el horizonte |
| G2 | DMD sobre trayectorias predichas en una ventana alrededor de t | "El modelo se comporta con estos modos" | Bajo | Interpreta comportamiento, no mecanismo |
| G3 | Contrafactuales: perturbar meteo de entrada y ver cambio de autovalores/predicción | "Con viento 2 m/s el episodio duraba 20 h menos" | Bajo | Fuera de distribución |

**Cómo se combinan.** No son excluyentes; un diseño elige en cuatro ejes:
(1) qué hace variar la dinámica — tiempo reciente (A, F), regímenes (B) o meteo
continua (C); (2) qué tan lineal es el núcleo — DMD puro, Koopman con red,
lineal + residuo (D3); (3) capa de incertidumbre (F3 o nada); (4) diagnósticos
que no son el modelo (E2, G3, estabilidad entre ventanas).

Combinaciones naturales mencionadas:
- **A2 + F3**: DMD online con intervalos — peldaño lineal barato y honesto.
- **C2 + D1**: bilineal en meteo, sin redes — mismo espíritu que C1 por
  mínimos cuadrados.
- **C1 + decoder lineal + G3**: Koopman con autovalores condicionados +
  contrafactuales.
- **B2/B3 + DMD por régimen**: lo más comunicable.
- **E2 como diagnóstico previo en todos los casos.**

**Propuesta de primer paso (a confirmar): medir antes de elegir.** La
preocupación del promediado es una hipótesis medible y barata: un espectrograma
(E2) y/o A1 corrido como diagnóstico muestran cuánto varían períodos y
decaimientos entre invierno y verano / entre días de inversión y ventilados.
Si varían mucho → se justifica B/C/F; si varían poco → alcanza algo simple.
Candidata a **primera tarea exploratoria del change**, antes de fijar la
arquitectura (PA-1).

**Impacto en otros PA** (cuando PA-14 se cierre):
- PA-1: se reformula según la opción elegida (ver nota en PA-1).
- PA-2: las opciones C1/C2/A6/B2 usan meteo para *decidir la dinámica*; si la
  meteo futura no se conoce, en el horizonte de 24 h se fija la de `t` o se
  asume persistencia meteo — hay que explicitarlo.
- PA-5: L y rollout definen qué períodos son interpretables.
- PA-10: pasa de "diagnósticos opcionales" a **entregables**: espectro,
  formas de modo, descomposición de episodios por modo, línea de tiempo
  espectral (si hay localización), estabilidad entre ventanas, matriz B.

### PA-2 — Entradas exógenas (control) en el rollout

El MP en Temuco está dominado por meteorología (inversión térmica, BLH,
viento) y emisiones de calefacción (hora del día, temperatura, estación). Un
sistema autónomo `z_{t+1} = K z_t` no "sabe" qué tiempo hará en t+1…t+24.

| Opción | Entradas | Fuga de información |
|---|---|---|
| a. Autónomo | Solo historia de MP | Ninguna |
| b. Control con futuro conocido | `u_t` = calendario (Fourier, día de semana, feriado, GEC) en el rollout | Ninguna (determinista) |
| c. Meteo como historia | Meteo/BLH observados hasta `t` entran al **encoder** (no al rollout) | Ninguna (igual que LightGBM con lag) |
| d. Meteo futura | Pronóstico meteo t+1…t+24 en el rollout | Requiere pronósticos históricos (nueva fuente) — **fuera de este change** |

**Recomendación**: **b + c** (Koopman con control, `z_{t+1} = K z_t + B u_t`).
Mantiene el mismo conjunto de información que tiene LightGBM, así la
comparación es justa. Dejar d anotada como hipótesis para un change de datos
posterior (Open-Meteo tiene API de pronósticos históricos; habría que
evaluarla).

**Nota lateral (riesgo para Fase 6)**: el baseline usa `blh` en `t` sin lag.
Si Open-Meteo entrega reanálisis (no disponible en tiempo real a las 01:00),
hay una fuga leve también en LightGBM. No se corrige acá; se anota para
`productivizacion`.

**Estado**: abierto.

### PA-3 — Cómputo: CPU vs GPU

`project-decisions.md` §3 anticipa GPU `ml.g4dn.xlarge` para AE+DMD, pero el
blueprint `compute` pide **medir antes de pagar GPU**. Con ~50k horas × ~30
canales y redes chicas (MLP/GRU de decenas de miles de parámetros), una epoch
en CPU debería tomar segundos.

**Recomendación**: corrida de humo en **CPU `ml.m5.large`** (1 ventana, pocas
epochs), medir tiempo por epoch, y subir a GPU solo si una corrida completa
(search + rolling + test) supera ~2 h en CPU. Si se queda en CPU, se actualiza
`project-decisions.md` §3 (desviación registrada). Si se va a GPU, pedir con
anticipación la cuota `ml.g4dn.xlarge for training job usage` (en cuenta nueva
arranca en 0; revisión manual de 1-2 días hábiles).

**Verificado (2026-10-30, `service-quotas list-service-quotas`)**: la cuenta
ya tiene `ml.g4dn.xlarge for training job usage` = **1** y `ml.g5.xlarge` = 1
(on-demand y spot); `ml.m5.large` = 20. No hace falta pedir cuota: la GPU está
disponible si la medición la justifica.

**Estado**: abierto (falta decidir CPU-primero vs GPU directo; la cuota ya no
bloquea).

### PA-4 — Huecos en los datos (NaN)

LightGBM tolera `NaN`; una red no. Hay que decidir:

- **Imputación de entrada**: interpolación lineal de huecos cortos (≤ N horas,
  p. ej. N=3) en MP y meteo; huecos más largos → la ventana de entrada se
  descarta para entrenamiento. Variables ausentes de forma estructural
  (`glob` en Las Encinas) → canal en cero + **máscara binaria** como canal
  extra, o se excluye `glob` del modelo conjunto.
- **Targets faltantes**: loss **enmascarada** (no se penaliza donde no hay
  verdad).
- **Evaluación**: se evalúa sobre las **mismas filas** que LightGBM (la
  intersección de timestamps con target válido), para que el test sea 1 a 1.

**Datos medidos (2026-10-30, `validated/sinca/`, grilla horaria completa
2021-01-01 → 2026-10-04):**

| Estación | MP faltante | Huecos MP p50 / p90 / max | Meteo utilizable (faltante) | Meteo inutilizable |
|---|---|---|---|---|
| Ñielol | 1.4-1.5% | 1 / 7 / 49 h | temp, pres, wspd (~1.4%); rain (7.6%); rhum (10%) | glob (75%) |
| Las Encinas | 3.1-3.4% | 1-2 / 9 / **~700 h** (44% de lo faltante en cortes > 7 días) | temp, pres, wspd, rain (~2.5-3.4%) | **rhum (96%)**, glob (100%) |
| PLC II | 4.7-5.0% | 1 / 6-7 / 75 h | pres, wspd, rain (~5%) | **temp (63%), rhum (63%)**, glob (99.8%) |

- Las 6 series MP están presentes a la vez en el **90.2%** de las horas.
- Los cortes de MP son mayormente cortos en Ñielol/PLC (≤ 24 h cubre ~75% de
  las horas faltantes); Las Encinas tiene cortes largos (~1 mes) que no se
  deben imputar.
- **Hallazgo colateral**: en PLC II el baseline LightGBM ve `temp`/`rhum`
  faltantes el 63% del tiempo — probable causa (parcial) de que pierda contra
  persistencia ahí. Alimenta PA-11.
- `rain` existe en las 3 estaciones con buena cobertura y el baseline no la usa.

**Recomendación (actualizada con los datos)**:
- Interpolación lineal de huecos de **≤ 6 h** (cubre ~45-50% de lo faltante en
  PLC/Ñielol sin inventar episodios); huecos mayores → ventanas de entrada
  que los toquen se descartan del entrenamiento.
- Loss **enmascarada** en targets faltantes.
- **Meteo del modelo conjunto**: solo canales con cobertura buena en todas las
  estaciones donde se usen: `pres`, `wspd`, `rain` × 3 estaciones +
  `temp` de Ñielol y Las Encinas (no de PLC). Se excluyen `rhum` y `glob`.
  Alternativa: un "temp regional" = media de las disponibles.
- Evaluar sobre la **intersección** de filas con LightGBM.

**Estado**: abierto (umbral 6 h y set de canales meteo a confirmar).

### PA-5 — Representación: ventanas, horizonte y loss multi-step

- **Lookback L** (historia que ve el encoder): candidatos 24 / 72 / 168 h.
- **Salida**: trayectoria horaria de **24 pasos** (t+1…t+24). Verificado
  contra `build.py`: el target es `shift(-24).rolling(24).mean()` =
  `mean(MP(t+1 … t+24))`, así que 24 pasos alcanzan para formar el promedio.
- **Transformación**: `log1p` (MP muy asimétrico) + z-score por canal,
  ajustados **solo con el train de cada ventana**.
- **Loss** — tres opciones:
  1. Solo sobre el promedio 24h (idéntico criterio al baseline).
  2. Solo sobre la trayectoria horaria (natural para Koopman).
  3. **Combinada**: trayectoria horaria (necesaria para que K aprenda
     dinámica) + RMSE ponderado sobre el promedio 24h (el criterio común de
     la escalera, que es el que se reporta como `train_loss` y se usa para
     selección).

**Recomendación**: L como hiperparámetro de la búsqueda (PA-12), 24 pasos de
salida, `log1p`+z-score, **loss combinada (3)**. El peso `w_i` se calcula
siempre en unidades originales (µg/m³), igual que el baseline.

**Estado**: abierto.

### PA-6 — Protocolo de evaluación (costo de 42 ventanas)

Re-entrenar una red 42 veces × N trials de búsqueda es caro y lento. El test
held-out es innegociable (heredado); lo ajustable es el rolling.

| Opción | Search | Rolling reportado | Comparabilidad con LightGBM |
|---|---|---|---|
| a. Igual al baseline | 12 ventanas | 42 ventanas | Total, pero costo alto |
| b. Reducido | 4-6 ventanas | **12 ventanas equiespaciadas** (`_select_windows`) | Se recalcula el rolling de LightGBM sobre esas mismas 12 ventanas **desde su `predictions.parquet`** (sin re-entrenar) |
| c. Mínimo | 1 ventana | Ninguno (solo test) | Solo test; pierde la estimación de varianza |

**Recomendación**: **b**. Mantiene una estimación de varianza honesta y
comparable sin re-entrenar el baseline. Implica una utilidad de evaluación
que filtre `predictions.parquet` por ventana (cambio menor en
`analysis/` o `evaluate/`).

**Estado**: abierto.

### PA-7 — Modelo conjunto vs por serie, y su tracking

Koopman sobre el estado regional (las 6 series juntas) es la formulación
natural y la que prueba la hipótesis cruzada. Pero el baseline es por serie.

- **Modelo**: **una red conjunta** con 6 salidas y loss base
  `Σ_i RMSE_ponderado_i` (sin término cruzado → cumple el criterio
  "independiente por serie" del prompt §5). Alternativa: 6 redes separadas
  (idéntico al baseline, pero pierde la información cruzada y cuesta 6×).
  Ablación opcional: modelo conjunto **por estación** (2 series) para medir
  cuánto aporta lo cruzado.
- **Tracking** — el dashboard trabaja por serie. Opciones:
  1. **1 run "padre" de zona** (`sinca-zona-koopman-v1`, métricas de zona,
     modelo, espectro) **+ 6 runs hijos** por serie (`sinca-{estacion}-koopman-v1`
     + tag `pollutant`, con su porción de `predictions.parquet`), vinculados
     por `mlflow.parentRunId`.
  2. Solo 1 run conjunto con métricas sufijadas por serie (rompe las vistas
     por serie del dashboard).
- **Experimento MLflow**: hoy todo vive en `sinca-baseline`. Opciones: seguir
  en el mismo experimento (comparación directa, nombre engañoso) o crear
  `sinca-advanced` y que el dashboard cargue ambos.

**Recomendación**: red conjunta; tracking opción **1** (padre + 6 hijos);
**mismo experimento** renombrando el concepto a "experimento del proyecto"
(sin renombrar el experimento existente para no romper nada) — o, si se
prefiere orden, `sinca-advanced` + dashboard multi-experimento. Preferencia
débil por mismo experimento por simplicidad.

**Estado**: abierto.

### PA-8 — Framework e imagen de entrenamiento — **CERRADO**

- **Framework**: **TensorFlow + Keras** (decisión del operador, 2026-10-30).
  Keras 3 sobre backend TF. El operador de Koopman y las losses custom se
  implementan como capas / `Model` subclass con `train_step` propio.
- **Imagen**: Dockerfile nuevo `docker/train-tf.Dockerfile` (mismo patrón
  solo-runtime que `train.Dockerfile`), en el **mismo repo ECR `sinca-train`
  con tags explícitos** (p. ej. `lgbm-v1`, `tf-cpu-v1`, `tf-gpu-v1`). Se deja
  de usar `:latest` para no pisar la imagen de LightGBM. Sin cambios de IAM.
- CPU: `tensorflow-cpu`; GPU (si PA-3 lo justifica): base
  `tensorflow/tensorflow:<ver>-gpu` o `tensorflow[and-cuda]`.
- Desarrollo local: TF en Windows nativo es solo CPU y sensible a la versión de
  Python → la iteración local corre **en Docker** con la misma imagen.

### PA-9 — Variante experimental de loss conjunta

`loss_total = Σ RMSE_ponderado(serie_i) + λ · loss(max_predicho, ICAP_zona_real)`

Problemas técnicos a resolver:
- **ICAP en TF**: la transformación MP→ICAP es lineal por tramos;
  implementarla con ops de TF (interpolación por tramos, diferenciable casi en
  todas partes) reutilizando `ICAP_ANCHORS`.
- **max no suave**: el `max` sobre 3 estaciones × 2 contaminantes da gradiente
  solo a un argumento → usar **softmax suave** (`τ·logsumexp(x/τ)`) en
  entrenamiento y `max` exacto en evaluación.
- **`loss(·)`**: RMSE en escala ICAP (simple) vs penalización asimétrica que
  castigue más subestimar sobre umbral 100 (alineado a recall).
- **λ**: grilla corta {0.1, 0.3, 1.0}.

**Recomendación**: incluir en este change, **después** de cerrar la variante
base; RMSE en escala ICAP + `τ` fijo; 3 valores de λ. Se loguea como
`koopman-joint` (familia distinta) para no mezclarla con la base.

**Estado**: abierto.

### PA-10 — Diagnósticos propios del modelo

`feature_importance.parquet` no aplica directamente. Propuesta:
- `koopman_spectrum.parquet`: autovalores de K → período (h) y tasa de
  decaimiento por modo. Chequeo de cordura esperado: modos cerca de 24 h y
  168 h. Es el "diagnóstico interpretativo" de la línea.
- Curvas de entrenamiento (loss por epoch) como métricas con `step` en MLflow.
- Dashboard: tolerar ausencia de `feature_importance` (ya degrada) y,
  **opcional**, un panel del espectro.

**Recomendación**: espectro + curvas sí; panel del dashboard opcional (puede
quedar para Fase 5).

**Nota 2026-10-30**: con la interpretabilidad como objetivo de primer orden,
este PA deja de ser "opcional": los diagnósticos interpretativos pasan a ser
entregables (lista en PA-14, "Impacto en otros PA"). El panel del dashboard
probablemente también entre en este change. Se redefine al cerrar PA-14.

**Estado**: abierto.

### PA-11 — Alcance: ¿qué más entra en Fase 4?

Pendientes que hoy no tienen change asignado:
- **LightGBM v2** con features cruzadas entre estaciones y `rain`/viento de
  Open-Meteo para PLC II (sugerido en el README tras el baseline).
- **Holt-Winters / SARIMA** (aparecen en la escalera del prompt §5 pero no en
  ningún change de `project-decisions.md` §8).

**Recomendación**: **fuera** de este change. LightGBM v2 es barato y muy
informativo (aisla si la ganancia de Koopman viene de lo cruzado o del
modelo) → change propio chico `lightgbm-v2`, idealmente antes o en paralelo.
Holt-Winters/SARIMA: proponer descartarlos o moverlos a `evaluacion` como
referencias adicionales; decisión del operador.

**Estado**: abierto.

### PA-12 — Búsqueda de hiperparámetros y semillas

- Random search (mismo criterio que el baseline), budget ~15 trials para C,
  sobre: `latent_dim` {8,16,32}, `lookback` {24,72,168}, `hidden` {64,128},
  `lr` {1e-3, 3e-4}, pesos de recon/linealidad {0.1, 1.0}, dropout {0, 0.1}.
  Early stopping sobre la valid de cada ventana.
- Para A (Hankel-DMD): búsqueda trivial sobre número de retardos y rango.
- **Semillas**: las redes tienen varianza entre inicializaciones → el modelo
  final se entrena con **3 semillas** y se reporta media ± desviación (el run
  reporta el ensemble o la media; a decidir).

**Recomendación**: lo anterior; final con 3 semillas y reportar el
**promedio de predicciones** (ensemble) como número oficial.

**Estado**: abierto.

### PA-13 — Escalas mezcladas y criterio de selección

`Σ_i RMSE_ponderado_i` en µg/m³ queda dominado por MP10 (valores ~2-3× los de
MP2.5) y por la estación más contaminada. Además, el prompt dejó abierto
"promedio o peor serie" como criterio de selección.

- **Loss**: calcular cada término en escala estandarizada (dividido por la
  desviación de la serie en train) manteniendo los pesos `w_i` en unidades
  originales.
- **Selección entre trials/variantes**: **skill relativo a persistencia**
  promediado sobre las 6 series (`1 − RMSE_pond_modelo / RMSE_pond_persistencia`),
  con el peor caso reportado al lado; más el bloque percentil.

**Recomendación**: lo anterior. Esto también aplicaría retroactivamente a la
lectura del baseline (sin re-entrenar).

**Estado**: parcialmente cerrado — el operador confirma que **hay
normalización** de datos (principio cerrado: `log1p` + z-score ajustado en
train, ver PA-5). Queda abierto el criterio de selección (skill vs
persistencia).

## Risks / Trade-offs

- **Dataset chico para una red** → overfit. Mitigación: redes chicas,
  dropout, early stopping, comparación contra Hankel-DMD (A) como control.
- **Koopman no converge / K inestable** (autovalores |λ|>1 explotan en 24
  pasos) → Mitigación: parametrizar K con radio espectral acotado o
  penalizarlo; rollout de entrenamiento con la misma longitud que el de
  evaluación.
- **Comparación injusta por información distinta** → Mitigación: PA-2 (mismo
  set de información que LightGBM) y PA-4 (mismas filas de evaluación).
- **Costo de búsqueda × ventanas** → Mitigación: PA-6 opción b, humo en CPU
  primero (PA-3).
- **Pisar la imagen LightGBM** al pushear `:latest` → Mitigación: tags
  explícitos (PA-8).
- **SQLite sin concurrencia**: no lanzar dos Training Jobs a la vez (padre +
  variantes) → una corrida por vez, como en Fase 3.
- **Pocos episodios extremos en el test (3 meses, julio-octubre)** → el bloque
  ICAP oficial puede quedar con n muy bajo; ya conocido desde Fase 3.

## Costos (estimación preliminar, a confirmar con PA-3/PA-6)

| Ítem | Supuesto | Estimación |
|---|---|---|
| Training Jobs CPU `ml.m5.large` (~USD 0.115/h) | ~15-25 corridas × 0.5-2 h durante la fase | USD 2-5 (único, no mensual) |
| Alternativa GPU `ml.g4dn.xlarge` (~USD 0.74/h) | mismas corridas | USD 10-30 |
| ECR (imagen TF CPU ~1.5 GB) | storage | < USD 0.1/mes |
| S3 artefactos MLflow adicionales | decenas de MB | despreciable |

Nada queda corriendo: Training Jobs terminan solos; sin endpoints.

## Migration Plan (borrador)

1. Preparación de datos para red (huecos, normalización, ventanas) + tests
   locales con fixture.
2. Variante A (Hankel-DMD) local → primera comparación sin AWS.
3. Variante C (Deep Koopman) local en subset → humo.
4. Imagen TF + push con tag explícito; Training Job de humo en CPU (medir).
5. Corrida completa variante base → comparación en dashboard.
6. Variante loss conjunta.
7. `EXPERIMENT-LOG.md` + verificación funcional (runs en MLflow con
   diagnósticos, visibles en el dashboard).
8. Rollback: código aditivo; runs nuevos identificables por tag `modelo`.

## Open Questions

- ~~Estadística de huecos~~ → resuelto, ver PA-4 (medido 2026-10-30).
- ~~Cuota GPU~~ → resuelto, ver PA-3 (`ml.g4dn.xlarge` = 1 ya disponible).
- **Experimento MLflow** (PA-7): mismo `sinca-baseline` o `sinca-advanced`.
- **PA-14**: filtrar el catálogo; ¿se hace primero la medición exploratoria
  (E2 / A1 como diagnóstico) antes de fijar arquitectura?
- **Holt-Winters/SARIMA** (PA-11): ¿se descartan o se mueven?

## Bitácora de discusión

### Sesión 2026-10-30

**Cerrado:**
- PA-8: TensorFlow + Keras; mismo repo ECR `sinca-train` con tags explícitos
  (se deja `:latest`).
- PA-13 (parcial): se normalizan los datos (`log1p` + z-score en train). Queda
  abierto el criterio de selección.
- Interpretabilidad = **objetivo de primer orden** (Goals).

**Verificado con datos reales:**
- PA-3: cuota `ml.g4dn.xlarge` training = 1 ya disponible (no hay que pedir).
- PA-4: estadística de huecos medida (tabla en PA-4). Hallazgo: PLC II sin
  `temp`/`rhum` el 63% del tiempo; `rain` disponible y sin usar.

**Discutido sin decidir:**
- Interpretabilidad por variante, tres escalas de tiempo (estimación /
  validez / lectura), y el problema del promediado de un K global
  (sección "Interpretabilidad").
- Catálogo PA-14 (grupos A-G).

**Para retomar:**
1. Filtrar PA-14 (¿qué grupos/opciones siguen en carrera?).
2. Decidir si la primera tarea del change es la **medición exploratoria**
   (E2 / A1 diagnóstico) de cuánto varía la dinámica en el año.
3. Con eso, reformular y cerrar PA-1; luego PA-2 (muy ligado) y el resto.
