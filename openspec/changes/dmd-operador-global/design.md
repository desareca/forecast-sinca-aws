# Design: `dmd-operador-global`

## Context

- Vinculante: `project-decisions.md` §9 ("Plan de `ae-dmd`", 7.1; partición y presupuesto fijo).
  Razonamiento: `docs/ae-dmd-propuesta-experimentos.md` (v5) §5, §8, §9. Motivo: `proposal.md`.
- Base (`EXPERIMENT-LOG.md` 2026-11-15, v2): runs AE `sinca-zona-ae_{arq}-aedmd-v2-s{seed}` con
  `latent.parquet` (`split ∈ train/es/sel/test`), `encoder.keras`, `decoder_mp.keras`,
  `dmd_operator.npz`; 36 runs DMD por serie v2; `lightgbm-aedmd` v2 (param `source_run_id` →
  `lightgbm-v1` holdout, con `predictions.parquet` de test y train hasta 2026-06) y
  `persistence-aedmd` v2; filas comunes n = 1407 (mínimo por serie).
- Reutilizable sin cambios: `ae/data.py` (grilla, normalizador, `build_windows`, `mp_to_ugm3`),
  `ae/dmd.py` (`fit_dmd`, `split_segments`, `predict_latent`), `ae/models.py` (`encode`,
  `decode_mp`), `features/partition.py` (`row_mask`), `evaluate/evaluate.py` (`sample_weights`,
  `weighted_rmse`), `analysis/sel_summary.py` (`common_rows`), `train/mlflow_log.py`,
  `baseline/persistence.py`.
- El encoder es `z = [enc_mp(x_mp) | enc_aux(x_aux)]`: `z_aux` depende solo de la ventana auxiliar.
- El sqlite de S3 tiene `artifact_location = file:///opt/ml/mlflow/mlruns` (rutas del job).

## Goals / Non-Goals

**Goals:** mejor operador lineal global (≤ 2 representaciones), con decisión registrada por etapa,
confirmado una vez en test, con veredicto del criterio de éxito e interpretabilidad.

**Non-Goals:** Training Jobs y reentrenar AE; Space; conv1d `L = 72`; K horario multi-paso
(1+24 h, diferido a `deep-koopman`); rolling del AE final; grillas de `d`, `L`, lookback o ε;
híbridos con LightGBM; clustering de regímenes; dinámica condicionada o localizada
(`dmd-localizado`); meteo futura; cambios al dashboard más allá de no romperse.

## Decisions

### D1. Dónde corre y cómo escribe

- Solo CPU local (`.venv`: TF 2.20 + `pydmd` 2025.8.1 + scipy).
- Datos validados: copia local de `sinca-data/validated/` (tarea [OPERADOR]); el normalizador del
  AE se reconstruye desde esa copia con el `train` de la partición (como en el job).
- Estado: solo la etapa `base` descarga (`ensure_state(run_ids=…)`, `--profile` obligatorio) a
  `mlflow-op/` los 6 AE v2, LightGBM v2, persistencia v2, los 36 DMD v2 y los 6 `lightgbm-v1`
  holdout (vía `source_run_id`), y guarda `LastModified` del sqlite en `.download.json`. Rechaza
  si ya existe un sqlite local. Las demás etapas corren con `--no-download`, sin `--profile`.
- Artefactos: `registro.py` escribe cada run nuevo con artefactos en
  `<state-dir>/mlruns/<run_id>/artifacts/` (artifact URI explícito por run; el experimento
  descargado apunta a `/opt/ml/...`).
- Publicación (`scripts/mlflow/publish_state.py`, script simple): `plan` lista runs nuevos
  (`stage=dmd-operador-global`) y falla si alguno no tiene artefactos; `apply --yes` compara
  `LastModified` remoto con `.download.json` (aborta si difiere), respalda el sqlite remoto en
  `_mlflow/backups/mlflow-<AAAAMMDD-HHMM>.sqlite`, sube el sqlite y solo `mlruns/` de los runs
  nuevos, y reescribe `.download.json` con el nuevo `LastModified` (`head_object`). Sin borrados.
  Lo corre el operador o el implementador tras su confirmación.
- `output/` no se versiona (`.gitignore`); los resúmenes de etapa van ahí y las cifras al log.

### D2. Módulos (`pipeline/operador/`)

`estado.py` (representaciones), `estimadores.py`, `rollout.py`, `evaluacion.py` (filas, métricas,
bootstrap, diagnósticos), `modos.py` (espectro, Gavish–Donoho, eliminación, emparejamiento),
`registro.py` (runs, resumen markdown), `run.py` (CLI `--etapa`), `decisiones.json` (arrastre) y
`README.md`. Tests en `tests/operador/` con fixtures sintéticos, sin AWS.

### D3. Filas de evaluación fijas

`eval_rows.parquet` (por serie, `timestamp`) = filas de los 36 DMD v2 ∩ LightGBM v2 ∩ persistencia
v2 (base n = 1407), elegibilidad `h = 24`. Se fija en `base`. Toda variante debe cubrirlas (spec:
error, no recorte). 1.1g (`h = 48`) se compara emparejada sobre su intersección con esas filas, con
su `n`, sin recortar las del resto. Target: `features.build` (`mean(MP(t+1…t+24))`).

### D4. Representaciones (mismas ventanas que el AE, `ae.data.build_windows`)

| Rep. | Estado | Decodificación |
|---|---|---|
| `hankel` | ventana MP normalizada 24×6 aplanada (144) | exacta: últimas 24 filas |
| `hankelmet` | + 12 canales meteo × 24 (432) | exacta |
| `hankelest` | un Hankel 24×2 por estación (3 operadores) | exacta |
| `hankel48` | MP 48×6 (288) | exacta |
| `pca` | PCA(12) de la ventana MP + PCA(4) de la ventana aux 34×24, ajustados en train | lineal (PCA MP) |
| `conv1d`, `gru` | `latent.parquet` v2 (16) | `decoder_mp.keras` |

- **Marzo**: sin uso. Normalización, PCA, K, rango, pesos y factores de Duan se ajustan solo con
  `train` (2021-01 → 2026-02); `es` solo se reporta como diagnóstico.
- Rango: Hankel con Gavish–Donoho (σ desconocido, `ω(β)·mediana(σ)`) desde 1.1; PCA y AE con rango
  completo (continuidad con `ae-dmd-prep`) y GD reportado. 1.6 refina.
- `pca` no es el `reference_recon_z("pca")` actual (PCA(d) solo de MP, S8 del doc).

### D5. Estimadores (numpy; `pydmd` solo contrasta el DMD exacto)

- Exacto: `ae.dmd.fit_dmd` (pares dentro de segmentos). Truncado: proyección `U_r`.
- Ponderado (1.2a): `K = (Z₁WZ₀ᵀ)(Z₀WZ₀ᵀ)⁺`, `w` del par `(t, t+1)` = `max_s w_s(target_s(t))`, solo
  `t` con `row_mask(…, "train")`; target faltante → 1. Con `W = I` coincide con el exacto (test).
- **fbDMD**: `K = (K_f·K_b⁻¹)^{1/2}` en coordenadas de rango `r`; de cada raíz de autovalor se elige
  la más cercana a `eig(K_f)` y se reconstruye real. Siempre se evalúa con recorte. Solo numpy (la
  versión ponderada no existe en `pydmd`, y `pydmd.FbDMD` mezcla bases SVD de X e Y: difiere
  ~2e-3). Se valida contra la fórmula en coordenadas completas y contra la verdad de un sintético
  con ruido de medición y de proceso.
- Recorte: `|λ| > 1−ε` → `(1−ε)·λ/|λ|`, reconstrucción real; ε = 1e-3 h⁻¹ fijo (vida media máxima
  ≈ 29 días), sin barrido.
- `A₂₄`: pares `(z_t, z_{t+24})` dentro de segmentos; solo referencia.

### D6. Rollout

- Autónomo: `K²⁴ z_t` con `matrix_power` (o `Ã²⁴` en coordenadas reducidas), como v2. La
  descomposición en modos se usa solo para espectro y eliminación (1.6).
- Exógeno (1.3b): `z_mp,t+1 = K_mm z_mp,t + K_ma z_aux,t`, ajuste conjunto por LS en train.
  - AE/PCA: en cada paso `k` se arma la ventana aux 34×24 que termina en `t+k`: horas `τ ≤ t`
    observadas; `τ > t` con meteo y máscaras de disponibilidad por persistencia diurna (`τ−24`) y
    calendario real de `τ`. AE: `z_aux = encode(x_mp, x_aux')[d_mp:]`; PCA: proyección PCA(4).
  - Hankel (DMDc): entrada `u_{t+k}` = 12 canales meteo por persistencia diurna + 10 de calendario
    de `t+k`; sin máscaras ni ventana aux completa. Rango GD sobre `[X; U]`.
- Test de no-fuga: alterar meteo posterior a `t` no cambia el pronóstico de `t` (spec).

### D7. Picos y retransformación (1.2)

- 1.2a: D5 ponderado. 1.2b: Duan por serie y paso `k`: `S_{s,k} = mean(exp(ℓ − ℓ̂))` sobre rollouts
  de train en horas reales, solo `t` con `t+24` en train (`ℓ` = log1p);
  `ŷ = mean_k((1 + expm1(ℓ̂_k))·S_{s,k} − 1)`.
- Se evalúan 1.2a, 1.2b y ambos; gana uno solo como decisión global.

### D8. Bootstrap

- Métricas: bloques móviles de 7 días de `sel`, B = 2000, percentil 95 %, semilla 0, emparejado
  (mismos índices para variantes y referencias). Razones y skill por serie y media de 6.
- Modos (1.5): bloques de 7 días de `train`, B = 200 re-estimaciones; emparejamiento húngaro
  (`scipy.optimize.linear_sum_assignment`) contra el operador de referencia usando un
  representante por modo (arg λ ≥ 0), en `(log|λ|, arg λ)`. Un modo "aparece" si período y vida
  media difieren ≤ 25 % (fijo); en modos reales se compara solo la vida media. Par conjugado = 1 modo.

### D9. Reglas de selección por etapa

- Puntaje: media de 6 series del RMSE ponderado en `eval_rows`; en AE, media de las 3 semillas (no
  la mejor: evita el optimismo anotado en v2).
- Una variante reemplaza a la vigente solo si la diferencia emparejada tiene intervalo < 0; si no,
  queda la más simple (Hankel < PCA < AE; autónomo < exógeno; sin peso < con peso; exacto < fbDMD).
- Oráculo (1.0): "techo en la representación" si el intervalo de la razón oráculo / mejor referencia
  (D12) alcanza 0.95 o más. Si es así para un AE, ese AE sale y Hankel pasa a línea principal.
- GRU se descarta si el rango entre semillas de su puntaje supera el semiancho del intervalo
  bootstrap del puntaje medio de 6 series.
- Escalera: pasa la mejor y la segunda solo si no es significativamente peor; máximo 2.
- **1.6**: techo de rango GD; se quita el modo de menor empeoramiento del error ponderado a 24 h en
  `sel` mientras ese empeoramiento no supere el intervalo bootstrap emparejado; se detiene en el
  primer empeoramiento significativo. Si quedan > 10 modos, se registra (criterio) y decide el
  operador. `d = 16` fijo.
- **Enmienda del operador, pausa 8.3 (2026-11-19)** — reemplaza la regla 1.6 anterior: (a) PCA y AE
  parten de rango completo (16); GD solo se reporta. (b) La eliminación compara cada candidato contra
  la configuración inicial de 1.6 (no contra el paso previo): se quita el modo de menor empeoramiento
  mientras el intervalo de `candidato − inicial` contenga 0 o quede bajo 0; en el primer paso
  significativamente peor se detiene y conserva el previo. (c) El par de período ≈ 24 h nunca se
  elimina (el target de 24 h lo cancela); sus armónicos (≈ 12 h, ≈ 8 h) solo se protegen si aparecen
  en ≥ 80 % del bootstrap por modo. (d) Candidato oficial = GRU (Duan, autónomo, horario, exacto,
  modos de 1.6), apartándose del desempate "más simple"; PCA va a test como dato secundario, sin
  efecto en el veredicto.
- La lectura es una propuesta: la decisión es del operador en cada pausa y queda en
  `decisiones.json` (`etapa`, ganador, configuración, fecha, nota). La CLI rechaza una etapa si la
  anterior no tiene decisión.

### D10. Runs y versión

| Run | Nombre | Notas |
|---|---|---|
| Evaluación conjunta | `sinca-zona-dmdop-{rep}-{exp}-aedmd-v{n}-s{seed}` | `{exp}`: `e10`, `e11a`…`e16`, `cierre`; params `tipo=operador` (oráculo: `diagnostico`; `A₂₄`: `referencia`), `etapa`, `estimador`, `rollout`, `rank`, `version` |
| Test por serie | `sinca-{slug}-{pollutant}-dmdop-{rep}-aedmd-v{n}-s{seed}` | contrato LightGBM, `split = test`, tag `evaluacion=test` |

- `base` no registra runs: deja `eval_rows.parquet` y la tabla de continuidad en `output/`.
- `{seed}`: semilla del AE; en Hankel/PCA, semilla del bootstrap (0). `version = 1` en todo el
  change; si una etapa se repite tras corregir un error, `version = 2` solo en esa etapa y se anota.
- Artefactos por run: `predictions.parquet` (6 series), `horizon.parquet`, `episodes.parquet`,
  `spectrum.parquet`, `tramos.parquet`, `bootstrap.parquet`, `operator.npz`.

### D11. Cierre y test

- Reajuste hasta 2026-06-30. Hankel/PCA reajustan normalización, PCA y K con train+es+sel. AE: solo
  K (más pesos/Duan) con latentes de train+es+sel del encoder congelado, sin reentrenar (decisión
  del operador). Asimetría asumida: el AE conserva normalización y encoder hasta 2026-02.
- Rango `r` y modos de 1.6 se fijan: el operador reajustado se empareja (húngaro, D8) con los modos
  seleccionados en `sel` y se conservan los emparejados; no se re-selecciona.
- Enmienda del operador (pausa 8.3): se reajustan y evalúan GRU (oficial, 3 semillas; veredicto con la
  media de semillas) y PCA (secundario, sin veredicto).
- Referencias de test: `lightgbm-v1` holdout (mismo corte) y persistencia calculada en el mismo
  proceso; filas comunes. Una sola evaluación; la CLI rechaza repetirla sin `--force-test`.
- Rolling de 12 ventanas solo si gana Hankel/PCA, como dato adicional (no elige).
- Interpretabilidad: `notebooks/dmd_operador_global.ipynb` (matplotlib, con outputs). Descomposición
  por modo exacta en Hankel/PCA; aproximada en AE (decoder no lineal, S5). Caveat log1p: modos
  multiplicativos en µg/m³.

### D12. Criterio de éxito (decisión del operador, reemplaza el pendiente de `ae-dmd-prep`)

- Mejor referencia por serie = la de menor RMSE ponderado entre LightGBM y persistencia (en `sel`
  para la lectura del oráculo; en test para el veredicto).
- Precisión: razón vs mejor referencia ≤ 1.05 en ≥ 4/6 (falla solo si todo el intervalo queda sobre
  1.05); PLC II además skill > 0 vs persistencia; piso en las 6: el intervalo del skill vs
  persistencia no queda entero bajo 0. Interpretabilidad: la de §9. Un solo veredicto.
- `project-decisions.md` §3.4/§9 se actualiza al cierre (Workflow follow-up).
- conv1d `L = 72` queda fuera: si 1.1g gana claro, se registra y se decide después.

## Risks / Trade-offs

- [`sel` corto: 3 meses, ~1400 filas por serie → intervalos anchos] → regla de significancia (D9),
  más simple ante empate, test una sola vez.
- [Muchas comparaciones secuenciales sobre `sel`] → arrastre sin factoriales; cifras de test no
  eligen; modos fijados antes del test.
- [Hankel de alta dimensión sobreajusta] → Gavish–Donoho desde el inicio.
- [Decodificación TF no determinista entre máquinas] → tolerancia relativa en la continuidad.
- [Escritura concurrente del sqlite] → publicación con chequeo de `LastModified`, respaldo y
  `.download.json` actualizado tras cada subida.

## Migration Plan

Grupo 1 (base) → compuertas 2–8 en orden, cada una con publicación y pausa → cierre (9). Rollback:
runs con `stage=dmd-operador-global` se borran con `scripts/mlflow/maintenance.py`; el respaldo del
sqlite queda en `_mlflow/backups/`.
