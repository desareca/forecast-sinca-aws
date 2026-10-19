# EXPERIMENT-LOG

Registro narrativo de los experimentos de modelado de este proyecto
(convención de `prompt-sinca-mlops.md` §6). MLflow guarda las métricas y los
parámetros por corrida; acá va el **por qué**: qué se probó, con qué
hipótesis, qué resultado dio, y qué se decidió después.

Una entrada por experimento o familia de experimentos:

```
## <fecha> — <change / serie o grupo de series>

- **Hipótesis**: qué se esperaba y por qué.
- **Qué se probó**: datos, features, hiperparámetros, partición, loss.
- **Resultado**: métricas clave (las de MLflow) y observaciones.
- **Decisión / siguiente paso**: qué se hace con esto.
```

Reglas rápidas:

- Las corridas de MLflow se nombran `sinca-{estacion}-{modelo}-v{n}` (ver
  `pipeline/train/mlflow_log.py`); acá se referencian por ese nombre.
- Métricas decisoras: **RMSE ponderado** por serie y el **bloque percentil
  propio** (`recall_top10pct`/`recall_top5pct`/`pr_auc_top10pct`). El bloque
  ICAP oficial (`recall_alerta`/`recall_preemergencia`/`recall_emergencia`) es
  reporte comparativo: el histórico tiene pocos casos extremos y puede ser
  inestable.
- Cambios de alcance o de diseño no se anotan acá como hechos consumados: se
  manejan como propuesta de change OpenSpec (mismo §6 del prompt).

---

## 2026-10-19 — baseline (Fase 3): persistencia + LightGBM

- **Hipótesis**: con lags + meteorología + Fourier, un LightGBM por serie debe
  batir claramente al baseline de persistencia (piso trivial = último promedio
  móvil 24h). Las features de calendario (feriados, ventana GEC) y las
  armónicas diaria/semanal/anual deben capturar parte de la estacionalidad que
  los árboles no ven nativamente.
- **Qué se probó**:
  - **Serie objetivo**: promedio móvil de 24h a `t+24h` (`mean(MP(t+1 … t+24h))`
    por serie; 6 modelos independientes MP2.5/MP10 × 3 estaciones).
  - **Loss**: RMSE ponderado, `w_i = 1 + α·max(0, y_i − umbral_regular)` con
    `α = 1` y `umbral_regular` = ancla ICAP 100 (MP10 150 / MP2.5 50).
  - **Partición**: ventana deslizante (train 2 años / valid 1 mes / step 1 mes,
    42 ventanas sobre 2021-01-01 → 2026-06-30) + test final held-out
    `2026-07-01 → 2026-10-04`.
  - **Hiperparámetros**: random search + early stopping, ~20 trials por serie
    (espacio del `design.md` §9), evaluados promediando el RMSE ponderado sobre
    una muestra de ventanas de validación.
- **Resultado**: _(pendiente — se completa tras correr el Training Job
  completo, tareas 8.4/8.5/9.1)._ Se espera loguear en MLflow, por serie:
  `train_loss`, `rolling_weighted_rmse_mean/std`, `test_loss` y el bloque
  completo de métricas; más el run de referencia `sinca-persistence-baseline`.
- **Decisión / siguiente paso**: si el percentil propio (`recall_top10pct`)
  apenas mejora el de persistencia, revisar features (p. ej. lags de
  meteorología más cortos o viento direccional). Si `α`/`umbral_regular`
  resultan mal calibrados (muchos o ningún caso extremo), ajustar y re-correr.
  Con los resultados del baseline se fija la vara para la Fase 4 (autoencoder +
  DMD/Koopman).
