# ae-dmd — Propuesta de experimentos (v5, base para el arquitecto)

> Documento de planificación teórica de `ae-dmd`, acordado con el operador (2026-11-12).
> **Lo vinculante está resumido en `project-decisions.md` §9 ("Plan de `ae-dmd`")**; este documento da el razonamiento detrás de cada decisión. Si hay diferencia, prevalece `project-decisions.md`.
> El borrador original (proposal/design con PA-1…PA-14) está en `docs/referencia/ae-dmd-borrador/` y es solo referencia.

---

## 0. Punto de partida (hechos medidos, EXPERIMENT-LOG 11.5)

| Latente (semilla 0) | Error a 1 paso | Error a 24 pasos | RMSE ponderado del pronóstico (test) |
|---|---|---|---|
| conv1d L=24 d=16 (referencia) | 16 % | 86 % | 33.28 |
| GRU L=24 (exploratorio) | 12 % | 81 % | 31.09 |
| LightGBM (ref.) | — | — | 28.17 |

- Errores relativos a la escala de z.
- **Dimensión efectiva** del latente: ~8 de 16. Aun así, **truncar a rango 8 empeora** la dinámica.
- **Picos**: en RMSE simple el DMD empata con LightGBM (20.97 vs 21.25), pero en el ponderado pierde en 5/6 series.
- **La reconstrucción misma ya subestima los picos**: con el latente verdadero, el sesgo en el tramo ≥100 es −22 µg/m³ (pendientes 0.44–0.90, en AE y en PCA). Es un techo que ningún operador puede superar (ver 1.0).
- La meteo aporta al AE (sin `aux_input` falla la compuerta).

## 1. Cómo entran hoy los datos al DMD (código actual)

```
en cada hora t:
  MP   [t−23 … t] × 6 series                 ─► encoder_mp  ─► z_mp  (12)
  aux  [t−23 … t] × 34 canales                ─► encoder_aux ─► z_aux (4)
       = 12 meteo (pres, wspd, rain ×3; temp Ñielol/LE; blh)
       + 12 máscaras de disponibilidad
       + 10 calendario (Fourier diaria/semanal/anual, feriado)

z_t = [z_mp | z_aux]   (16)
K   = Z₁·pinv(Z₀) sobre pares horarios consecutivos de todo el train; global; rango completo
pronóstico: z_{t+24} = K²⁴·z_t → decoder_mp(z) → ventana [t+1 … t+24] → promedio
```

### Consecuencias estructurales

| ID | Hecho | Consecuencia |
|---|---|---|
| S1 | El paso es de 1 h con ventanas solapadas 23/24 | K tiene que aprender el desplazamiento de la ventana dentro de un latente no lineal (H1) |
| S2 | **K también hace evolucionar z_aux** | El modelo "pronostica" meteo y calendario de forma lineal y autónoma. No usa información futura, pero es un pronóstico implícito y pobre |
| S3 | La meteo entra solo por los bloques cruzados z_aux → z_mp de K | Actúa como **forzamiento aditivo**: suma al MP, pero **no puede cambiar la vida media** de los modos de MP |
| S4 | El calendario está comprimido junto con 12 variables meteo en 4 dimensiones no lineales | El ciclo diario se predice con K en vez de usarse como conocido, y **no se puede separar** dentro de z_aux |
| S5 | El decoder es no lineal (conv1d) y recibe z completo | La descomposición por modo es aproximada |
| S6 | K se ajusta una sola vez con todo el train (~5 años) | Promedia más historia que una ventana de rolling de 2 años |
| S7 | La validación actual es solo junio de 2026 y se usó para el early stopping del AE | Es poco para seleccionar por RMSE ponderado y no es del todo ajena al encoder (se corrige en el change 0) |
| S8 | El PCA(16) de referencia se calculó solo con MP | No es comparable con el AE para la dinámica (se corrige en 1.1c) |

---

## 2. Hipótesis de por qué falla el DMD global

| ID | Hipótesis | Evidencia | Mecanismo |
|---|---|---|---|
| H0 | **Techo de la representación** | Sesgo de −22 µg/m³ en picos ya en la reconstrucción | El decoder contrae hacia la media. Si eso sobrevive al promedio de 24 h, el problema no es K |
| H1 | Latente mal condicionado para pasos de 1 h | 1 paso bien, 24 pasos mal; S1 | En coordenadas de retardo el desplazamiento es exactamente lineal; en un latente comprimido no lineal, no. Los errores se componen en K²⁴ |
| H2 | Regresión a la media del estimador | Pierde en picos | Mínimos cuadrados con ruido sesga los autovalores hacia adentro. Con MSE se predice la esperanza condicional, más suave que la realidad |
| H3 | Promediado de la dinámica (PA-14) | No medida; S6 | Una inversión cambia la tasa de decaimiento. Un K global la promedia y la disfraza como amplitud |
| H4 | Acoplamiento meteo solo aditivo | S3 | La meteo puede empujar el nivel, pero no alargar un episodio |
| H5 | Energía ≠ relevancia dinámica | Truncar a 8 empeora | Modos de poca varianza cargan la dinámica lenta |
| H6 | Objetivo mal alineado | — | K se ajusta a 1 paso y sin ponderar; se evalúa a 24 pasos, en promedio y con ponderación |
| H7 | Exógenas tratadas como estado | S2, S4 | K gasta capacidad en predecir meteo y calendario, y esos errores contaminan z_mp |

---

## 3. Principios y criterios acordados

### 3.1 Información permitida y dónde entra

- **Prohibido**: meteo futura, observada o pronosticada.
- **Permitido**: todo lo pasado y todo lo determinista que se derive de t.

| Información | Rol | Cómo entra |
|---|---|---|
| Historia de MP | Estado | Encoder / Hankel |
| Historia de meteo hasta t | Estado | Encoder / Hankel |
| Meteo del horizonte | Supuesto | **Persistencia diurna**: meteo(t+k) = meteo(t+k−24). No se congela el valor de t, porque BLH y temperatura tienen ciclo diario |
| **Régimen meteo del día** | **Condiciona la dinámica** | K(u_t), con u = **agregados de las últimas 24 h** (por ejemplo BLH mínimo, viento medio) |
| Ciclo diario | **Modo** de la dinámica (decidido) | K horario; aparece en el espectro con período y decaimiento |
| Día de semana, feriados, temporada | **Control conocido** (decidido) | Calendario real de t+k en el rollout |

### 3.2 Escalera de representaciones

Cada peldaño agrega una sola cosa:
1. **Hankel** (retardos lineales, sin compresión).
2. **PCA** de la ventana (compresión lineal).
3. **conv1d y GRU** (compresión no lineal; embeddings de retardos aprendidos).

Pasan a las etapas siguientes **a lo más 2 representaciones**. Todas reciben el mismo conjunto de entradas.

### 3.3 Método

- Screening secuencial con arrastre del ganador; no hay factoriales.
- Sin grillas de tamaño: d y rango salen de análisis espectral y selección de modos.
- 3 semillas.
- **Herramientas**: se permite `pydmd` (a prueba) para los estimadores estándar. El DMD exacto actual en numpy queda como control: ambos deben coincidir. `pydmd` no entra al entrenamiento con TensorFlow; trabaja sobre los latentes como arrays.
- Interpretabilidad transversal (§9).

### 3.4 Criterio de éxito (por serie, las dos condiciones)

- **Precisión**: RMSE ponderado ≤ **1.05 × LightGBM** en **≥ 4/6 series**, en las mismas filas. En PLC II, además, **skill > 0 vs persistencia**.
  - Toda razón se reporta con **intervalo por bootstrap de bloques de días**. Una serie "no cumple" solo si todo el intervalo queda sobre 1.05; una diferencia entre variantes cuenta solo si supera el intervalo.
- **Interpretabilidad**: **≤ 10 modos**. Los modos dominantes aparecen en las 3 semillas y en las re-estimaciones por bootstrap (≥ 80 %), con intervalos de período y vida media que se solapan.

---

## 4. Mapa de changes y compuertas

| # | Change | Contenido | Compuerta de salida |
|---|---|---|---|
| 0 | `ae-dmd-prep` | Infraestructura y nueva partición (detalle abajo) | Infra lista y AE reentrenados |
| 1 | `dmd-operador-global` | Oráculo, escalera, picos, exógenas, horizonte, estimador, modos (§5) | Mejor operador global (≤ 2 representaciones). Si cumple el criterio, es candidato oficial |
| 2 | `dmd-localizado` | Medición de variabilidad (siempre); dinámica condicionada solo si la variación supera el ruido (§6) | Mejor variante localizada, o PA-14 cerrado como "global basta" |
| 3 | `deep-koopman` | Koopman entrenado de extremo a extremo (§7) | Entra si el mejor lineal **no cumple** la precisión, o si el oráculo muestra que la representación es el techo y Hankel tampoco alcanza |

### Contenido del change 0

- **Nueva partición** (§8): reentrenar conv1d y GRU, 3 semillas cada uno (6 entrenamientos), con train hasta febrero de 2026, early stopping en marzo y validación de selección en abril–junio.
- `predictions.parquet` de persistencia y de LightGBM sobre la validación de selección.
- Descarga selectiva en `ensure_state`.
- Semilla en el nombre del run; limpieza de runs v99.
- Pase DMD para todas las semillas y para el GRU.
- Probar `pydmd` en la imagen (Python 3.11, TensorFlow 2.20, numpy ≥ 1.26). Plan B: entorno aparte que lee los latentes guardados.

---

## 5. Change 1 — Operador global post-hoc

Todo es álgebra lineal sobre latentes ya entrenados o sobre las series crudas, en CPU.

### 1.0 Oráculo: techo de la representación (H0) — primer experimento

Se codifica la ventana **real** que termina en t+24, se decodifica y se promedia. Es el pronóstico que daría un operador perfecto. Usa información futura, pero solo como diagnóstico.

| Resultado del oráculo (RMSE ponderado) | Lectura | Consecuencia |
|---|---|---|
| Claramente bajo LightGBM | El problema es la dinámica | El plan sigue como está |
| Cerca o sobre LightGBM | El techo es la representación | Los AE actuales quedan fuera. Hankel sin compresión pasa a ser la línea principal y el change 3 (representación entrenada con la métrica) gana prioridad |

Se calcula para conv1d, GRU y PCA. Hankel no tiene techo de reconstrucción.

### 1.1 Escalera de representaciones

| Exp | Estado | Pregunta |
|---|---|---|
| 1.1a Hankel MP | 6 series con h = 24 retardos horarios | Piso autónomo |
| 1.1b Hankel MP + meteo | 1.1a más los 12 canales meteo con retardos | Aporte lineal de la meteo; comparación justa con el AE |
| 1.1c PCA + DMD | **PCA(12) sobre MP + PCA(4) sobre auxiliares**, la misma estructura del AE | ¿Comprimir linealmente aporta? |
| 1.1d conv1d + DMD | Semillas 0/1/2 | Fila base (si pasa el oráculo) |
| 1.1e GRU + DMD | Semillas 0/1/2 | ¿Un encoder recurrente linealiza mejor? Se descarta si el pronóstico varía mucho entre semillas |
| 1.1f Hankel por estación vs conjunto | Ablación | ¿Aporta la información cruzada entre estaciones? |
| 1.1g Hankel h = 48 | Control de memoria | Si supera claramente a h=24, se habilita conv1d L=72 |

En los AE, la fila base se corre con z_aux autónomo, como hoy.

### 1.2 Picos, ponderación y log1p (H2/H6)

Va temprano porque la evidencia dice que se pierde en los picos y puede cambiar qué representación gana.

- **1.2a DMD por mínimos cuadrados ponderados**, con la w de la métrica. Peso de cada par: el **máximo de w entre las 6 series**.
- **1.2b Corrección de retransformación** (smearing de Duan): separa Jensen de contracción.

La que gane queda como decisión global.

### 1.3 Tratamiento de las exógenas en el rollout (H7, S2-S4)

| Variante | Qué hace con la meteo y el calendario | Formulación |
|---|---|---|
| 1.3a Autónoma (actual) | K hace evolucionar z_aux | z_{t+1} = K·z_t |
| 1.3b **Exógena** | z_aux deja de ser estado. En cada paso k se **re-codifica** la ventana auxiliar con **persistencia diurna** de la meteo y **calendario real** de t+k | z_mp,{t+1} = K_mm·z_mp,t + K_ma·z_aux,t |

- Con persistencia diurna, la ventana recodificada es un perfil real (el de ayer), no una constante que el encoder nunca vio.
- En Hankel, el equivalente es tratar los canales meteo como entrada con la misma persistencia.
- Deja armada la estructura del change 2: K_mm es el bloque que se condiciona.

### 1.4 Horizonte del operador

| | K horario (rollout K²⁴) | A₂₄ directo (z_t → z_{t+24}) |
|---|---|---|
| Alineación con el target | Indirecta, con error compuesto | Directa |
| Estabilidad | λ²⁴ amplifica errores | Una sola aplicación |
| Interpretabilidad | Ciclo diario visible como modo | El modo diario se pliega a período ∞; el decaimiento se recupera con \|λ\| = \|μ\|^(1/24) |

**K horario es el principal** (ciclo diario como modo, §3.1). A₂₄ va solo como referencia de precisión. Variante intermedia: K horario ajustado con objetivo multi-paso (1 y 24 h); si resulta cara, pasa al change 3.

### 1.5 Estimador y estabilidad

| Exp | Ataca |
|---|---|
| DMD exacto | Control |
| Clip \|λ\| ≤ 1−ε | Explosión en el rollout |
| fbDMD | Sesgo de autovalores hacia adentro (H2) |

**Incertidumbre por modo**: bootstrap por bloques (días o semanas) del estimador elegido. Es el insumo de §6 y §9.

### 1.6 Rango y modos, sin grilla

1. **Techo de rango**: SVD de la trayectoria y umbral de Gavish–Donoho.
2. **Selección de modos**: **eliminación hacia atrás**. Se quita de a un modo, el que menos empeora el error a 24 h. Selecciona por contribución dinámica y no por energía (H5).
3. **Salida**: curva precisión vs número de modos.

d se queda en 16; el análisis dice si d=8 vale para el change 3.

---

## 6. Change 2 — Dinámica localizada

### Tipos de localización

| Tipo | Qué representa K | Promediado (H3) | Ruido / sesgo (H2) | Ante un cambio de régimen |
|---|---|---|---|---|
| Global | Promedio de todo el train | Alto | Bajo | No reacciona |
| Temporal (ventana deslizante / período contiguo) | "Cómo se comportó estos últimos días" | Bajo | Medio-alto, según la ventana | **Con retraso** (falla al inicio de los episodios) |
| **Condicionada al régimen meteo del día** | "Cómo se comporta en condiciones como las de hoy" | Bajo | Bajo (usa todos los datos parecidos) | **Inmediata**, si la meteo explica el cambio |

La localización temporal se usa solo como diagnóstico. Los modelos son condicionados.

### 2.0 Medición de variabilidad (siempre se hace)

- DMD por ventanas mensuales: trayectoria de períodos y vidas medias.
- DMD por tercil de los agregados diarios (BLH mínimo, viento medio, temperatura).
- Espectrograma wavelet.

**Compuerta**: los intervalos por bootstrap dan la escala del ruido.
- Ejemplo que justifica localizar: invierno 38 ± 4 h vs verano 12 ± 3 h.
- Ejemplo que no: 22 ± 5 vs 19 ± 5.

Si los decaimientos se ordenan por BLH más que por fecha, se salta 2.1. **De aquí salen las variables de la LPV**: las que realmente ordenan los decaimientos.

### Modelos condicionados (sobre la estructura exógena de 1.3b; solo se condiciona K_mm)

| Exp | Formulación | Lectura | Limitación |
|---|---|---|---|
| 2.1 Temporada | Un K_mm para GEC y otro para el resto | "En invierno la vida media es 35 h" | No distingue días ventilados dentro del invierno |
| 2.2 Régimen meteo | K_mm por régimen según los agregados de las últimas 24 h (umbrales salidos de 2.0) | "Hoy hay inversión → 40 h" | Umbrales; transición abrupta; menos datos por K |
| 2.3 **LPV** | K_mm(u_t) = K₀ + Σ uᵢ,t·Kᵢ, con u = agregados diarios en unidades físicas | "Cada 100 m menos de BLH mínimo suman 6 h de vida media" | ~600 parámetros con 3 variables; requiere ridge |

- **Entrenamiento de la LPV**: regresión lineal sobre las features aumentadas [z_mp,t, uᵢ,t·z_mp,t, z_aux,t], con todos los datos. Los parámetros quedan fijos, pero **cada predicción usa su propio operador**.
- **Estabilidad**: u se acota al rango observado y se verifica el radio espectral de K(u) en todo ese rango.
- **Diferencia con 1.3b**: allí la meteo empuja el estado; en la LPV cambia los modos (H4).
- **Limitación común**: supone que el régimen de hoy vale mañana. Un frente no observable en t no se anticipa.
- **Candidata principal: 2.3.**

---

## 7. Change 3 — Deep Koopman (con compuerta)

### Arquitectura

- **Estado**: encoder (conv1d o GRU, según el change 1) sobre la historia de MP y meteo. **El calendario sale del encoder** y entra solo como control B·u_{t+k}.
- **Operador**: K en **bloques 2×2 de rotación con radio ≤ 1**, con período y decaimiento legibles por bloque. Hereda el condicionamiento del change 2 (K₀ y Kᵢ entrenables). Red auxiliar para los autovalores solo si lo lineal se queda corto.
- **Decoder lineal** como variante principal: la descomposición por modo es **exacta** en log1p. Costo: por construcción no puede reconstruir mejor que PCA(d), así que renuncia al ~8 % que ganó el AE. Decoder no lineal como comparación.
- **Arranque en caliente** desde el AE entrenado. d según 1.6.

### Cómo se ajusta K

| Opción | Qué representa K | Decisión |
|---|---|---|
| **K entrenable global** (peso compartido; el batch es solo la muestra del optimizador) | Todo el train | **Elegida**: única compatible con bloques 2×2 y con la forma condicionada |
| K por batch aleatorio | El mismo promedio, con más ruido | Descartada |
| K por período contiguo | Dinámica reciente | Descartada como modelo: llega tarde |

- Al terminar el entrenamiento, los parámetros quedan fijos. Con condicionamiento, el operador efectivo cambia en cada predicción.
- **Diagnóstico de coherencia**: DMD por mínimos cuadrados sobre el latente de todo el train, comparado con el K aprendido.

### Las "regresiones a la media" y cómo se atacan

| Problema | Qué lo resuelve | Qué no lo resuelve |
|---|---|---|
| Techo de la representación (H0) | Entrenar con la pérdida ponderada (F3); o Hankel sin compresión | Cualquier cambio del operador |
| Promediado de la dinámica (H3) | Condicionamiento heredado del change 2 | Un K global fijo, aunque sea entrenable |
| Contracción por MSE (H2) | Pérdida ponderada (F3), multi-paso (F2), localización | El radio ≤ 1 |
| Incertidumbre real (frentes) | Nada sin meteo futura | — |

### Pérdidas: receta incremental

| Paso | Pérdida | Ataca |
|---|---|---|
| F1 | L_rec + L_lin | H1 |
| F2 | \+ L_pred multi-paso (24 h) | H6, H2 |
| F3 | \+ L_tgt: RMSE ponderado del promedio 24 h; variante ICAP_zona con max suave (PA-9) | H0, H2 |

- Son 3 corridas por 3 semillas.
- **Selección de modos**: eliminación hacia atrás sobre la K entrenada.
- **Pesos λ**: normalización de escalas al inicio; a lo más se barre λ_lin, con 3 valores.

---

## 8. Protocolo de evaluación común

### Partición

| Tramo | Fechas | Uso |
|---|---|---|
| Train | 2021-01 → 2026-02 | Entrenar los AE y ajustar K (una sola vez) |
| Early stopping | Marzo de 2026 | Solo para detener el entrenamiento de los AE |
| **Validación de selección** | **Abril–junio de 2026** | Elegir el ganador de cada etapa. Tres meses, incluye el inicio del invierno; ningún modelo la ve al entrenar |
| Test held-out | 2026-07-01 → 2026-10-04 | Solo confirma al final de cada change |

### Reglas

- **Mismo esquema para todos**: Hankel, PCA y AE se seleccionan en la validación, con K ajustado sobre el mismo train.
- **Confirmación en test**: la configuración ganadora se reajusta con datos hasta junio de 2026 y se evalúa una vez. Los números de test ya conocidos (33.28, 31.09) no se usan para elegir.
- **Rolling de 12 ventanas**: solo para Hankel y PCA, como dato adicional. Para un AE, solo en el modelo final, reentrenándolo por ventana.
- **3 semillas**, con la semilla en el nombre del run.
- **Métricas**: RMSE ponderado por serie y skill vs persistencia (promedio y peor caso), en las mismas filas que LightGBM, con intervalo por bootstrap de bloques.
- **Diagnósticos obligatorios**:
  - error vs horizonte (1…24 h)
  - error en episodios (≥ umbral ICAP100)
  - espectro
  - pendiente y sesgo por tramo
- **Cómputo**: CPU, salvo que el change 3 demuestre lo contrario.

## 9. Interpretabilidad (entregable en cada change)

- **Espectro**: período y vida media por modo, con intervalo (bootstrap).
- **Formas de modo**: participación por estación y contaminante; fase = desfase entre estaciones.
- **Descomposición de episodios por modo**: exacta en Hankel, PCA y decoder lineal; aproximada con el decoder conv1d actual (S5).
- **Estabilidad de modos** entre semillas y re-estimaciones.
- **Matrices de forzamiento** (K_ma, B) y, si hay LPV, curvas autovalor vs variable meteo.
- Caveat: con log1p, los modos son multiplicativos en µg/m³.

## 10. Fuera de alcance

- Híbridos DMD + LightGBM: solo como referencia futura.
- Clustering de regímenes latentes (SLDS, HMM, k-means).
- Localización temporal como modelo: solo como diagnóstico.
- Meteo futura.
- Grillas de d, L y lookback. conv1d L=72 solo si lo habilita 1.1g.
- LightGBM v2 y Holt-Winters/SARIMA (PA-11).

## 11. Pendientes que dependen de datos

1. **Variables de la LPV**: se eligen con la medición 2.0.
2. **Compatibilidad de `pydmd`** con la imagen: se confirma al construirla (change 0).

## 12. Mapa a los PA del diseño (referencia)

| PA | Destino |
|---|---|
| PA-1 escalera | Oráculo → Hankel → PCA → conv1d/GRU post-hoc → exógenas → condicionado lineal → Deep Koopman |
| PA-14 localización | Medición en 2.0; temporada / régimen / LPV en el change 2; red auxiliar solo en el change 3 |
| PA-2 exógenas | Sin meteo futura. Persistencia diurna; régimen del día condiciona; calendario como control conocido |
| PA-5 / PA-12 | Grillas reemplazadas por análisis; L = 24 |
| PA-6 evaluación | Validación de selección abril–junio de 2026; rolling de 12 ventanas solo para Hankel/PCA y el modelo final |
| PA-9 loss ICAP | Change 3, paso F3 |
| PA-10 diagnósticos | §9, entregables |
| PA-7, PA-13 | Se mantienen |
| PA-3 cómputo | CPU |
| PA-11 | Fuera |
| Dependencias | `pydmd` permitido a prueba (cambia la decisión de "solo numpy") |
| Pendientes heredados | Change 0 (infraestructura) y 1.2 (log1p) |
