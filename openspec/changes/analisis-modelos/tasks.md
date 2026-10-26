## 0. Pre-work (operador)

- [x] 0.1 Verificar identidad: `aws sts get-caller-identity --profile $AWS_PROFILE` y confirmar `Account = AWS_ACCOUNT_ID`
- [x] 0.2 Confirmar que el change `baseline` está archivado y `openspec/specs/modelado/spec.md` existe (base del delta `modelado` de este change)

## 1. Persistencia de diagnósticos en el training

- [x] 1.1 `pipeline/train/train.py`: al cierre de cada run LightGBM, construir y loguear con `mlflow.log_artifact` (dentro de `_mlflow/mlruns/`, no a `model-artifacts/`):
  - `predictions.parquet` — concat de `rolling_frame` + `test_frame` con columna `split` (`train`/`valid`/`test`), columnas `timestamp, estacion, pollutant, y_true, y_pred, split`
  - `feature_importance.parquet` — gain por feature y serie (`booster.feature_importance("gain")`)
  - `zone_icap.parquet` — `ICAP_zona` real vs pred del test held-out (`evaluate.zone_icap(test_frame)`), columnas `timestamp, icap_true, icap_pred`

## 2. Módulo de análisis (`pipeline/analysis/`)

- [x] 2.1 Crear `pipeline/analysis/__init__.py` y `pipeline/analysis/load.py` — `download_state` a un dir local, listar runs/experimento desde `mlflow.sqlite`, ubicar y leer los 3 parquet por run; soportar `--data-dir` (estado local) vs S3 directo; degradar a "solo métricas" con aviso si falta `predictions.parquet`
- [x] 2.2 Crear `pipeline/analysis/plots.py` — figuras Plotly puras (independientes de Streamlit): series real vs pred con bandas ICAP (alerta 100 / preemergencia 200 / emergencia 300), `ICAP_zona` real vs pred, feature importance horizontal, barra RMSE LightGBM vs persistencia
- [x] 2.3 Crear `pipeline/analysis/streamlit_app.py` — widgets (experimento, run, serie) + pestañas + `@st.cache_data` para no re-descargar

## 3. Tooling local (contenedor) y fixture sintético

- [x] 3.1 Crear `requirements-analysis.txt` (streamlit, plotly, pandas, mlflow, boto3, pyarrow)
- [x] 3.2 Crear `docker/analysis.Dockerfile` — imagen con solo el runtime instalado desde `requirements-analysis.txt`, `ENTRYPOINT ["streamlit", "run"]`
- [x] 3.3 Crear fixture sintético en `pipeline/analysis/fixtures/` (dataset señuelo con predicciones, feature importance e `ICAP_zona` mock) para iterar la estética sin AWS

## 4. Pruebas locales de gráficos (iteración, sin AWS)

- [x] 4.1 Construir la imagen y correr el dashboard en contenedor contra el fixture (montando el código como volumen); afinar la definición de gráficos junto al operador hasta cerrarla

## 5. Documentación

- [x] 5.1 Actualizar `README.md` — sección "Dashboard de análisis": correr local en contenedor (`docker build`/`docker run` + fixture) y en el Space (`pip install -r requirements-analysis.txt` + `streamlit run` + port-forward del 8501)
- [x] 5.2 Actualizar `project-decisions.md` §8 — insertar la fase `analisis-modelos` después de `baseline` y renumerar las siguientes
- [ ] 5.4 Verificar que `notebooks/analisis_baseline.ipynb` (base, provisto por el arquitecto) abre en el Space, y documentar en `README.md` el flujo de notebooks: encender (Studio → Spaces → `sinca-dev` → Open), trabajar (rol del Space, sin credenciales), y apagar (Spaces → `sinca-dev` → Stop) para evitar sobrecostos

## 6. Ejecución real (operador corre y pega output)

- [x] 6.1 Reset del tracking (borrado aprobado, "partir de 0"): `aws s3 rm s3://sinca-mlflow/_mlflow/ --recursive` y `aws s3 rm s3://sinca-mlflow/model-artifacts/ --recursive`; pegar output real
  <!-- 2026-10-26: 17 objetos / 49 MB borrados (7 runs viejos + sqlite + code/ + model-artifacts). Output real del operador pegado en sesión. -->
- [x] 6.2 Re-correr el Training Job completo: `python pipeline/train/launch.py --role-arn <arn> --image-uri <uri>:latest --wait` → 7 runs limpios (`sinca-{estacion}-lightgbm-v1` × 6 + `sinca-persistence-baseline`)
  <!-- 2026-10-26: job `sinca-baseline-train-2026-10-26-15-55-51-226` → Completed (2380 s facturables, ~40 min). 6 series LightGBM + persistencia; `Estado MLflow subido ... 38 objetos`. -->
- [x] 6.3 Verificar en S3 que aparecen `predictions.parquet`, `feature_importance.parquet` y `zone_icap.parquet` bajo `s3://sinca-mlflow/_mlflow/mlruns/.../artifacts/`
  <!-- 2026-10-26: 7 run dirs; 6 runs LightGBM × 4 parquet (6 predictions, 6 feature_importance, 6 zone_icap, 6 rolling_scores) = 24 parquet; 39 objetos totales (sqlite + code/ + modelos). Persistencia sin parquet (esperado). -->
- [x] 6.4 Levantar el dashboard en el Space (`git pull` + `pip install -r requirements-analysis.txt` + `streamlit run pipeline/analysis/streamlit_app.py --server.port 8501`) y port-forward del 8501
  <!-- 2026-10-26: ejecutado con Docker local contra S3 real (opcion A aprobada por el operador), no en el Space: el port-forward depende de la conexion remota de VS Code, bloqueada por aws-toolkit-vscode#7915. `/_stcore/health`=ok en :8501; AppTest sin excepciones. -->

## 7. Verificación funcional

- [ ] 7.1 Ver el dashboard con datos reales: series real vs pred, `ICAP_zona`, RMSE vs persistencia y feature importance, confirmando con output/captura real que coinciden con las métricas de MLflow

## 8. Scripts locales de start/stop del Space (no versionados)

- [x] 8.1 Agregar a `.gitignore` (sección Terraform, junto a `backend.hcl`):
  `infra/scripts/start-space.ps1` y `infra/scripts/stop-space.ps1` — llevan el
  profile y account reales, nunca se commitean
- [x] 8.2 Crear `infra/scripts/start-space.ps1.example` y
  `infra/scripts/stop-space.ps1.example` con placeholders (`AWS_PROFILE`,
  `AWS_ACCOUNT_ID`, `DOMAIN_ID`), siguiendo la convención de
  `terraform.tfvars.example`. Contenido de referencia (implementado en los
  `.example`; el operador copia el `.example` a `.ps1` y completa los
  placeholders):

  `start-space.ps1.example` — arranca el app JupyterLab (idempotente: crea si
  falta, o borra y recrea si quedó en mal estado) y espera `InService`:

  ```powershell
  $env:AWS_PROFILE = "AWS_PROFILE"
  $domain = "DOMAIN_ID"
  $spec   = "InstanceType=ml.t3.large,LifecycleConfigArn=arn:aws:sagemaker:us-east-1:AWS_ACCOUNT_ID:studio-lifecycle-config/sinca-on-start"

  $existing = aws sagemaker list-apps --domain-id-equals $domain --space-name-equals sinca-dev --profile $env:AWS_PROFILE --query "Apps[?AppName=='default'].Status" --output text

  if ($existing -eq "InService") {
      Write-Host "App ya InService."
  } else {
      if ($existing) {
          aws sagemaker delete-app --domain-id $domain --app-type JupyterLab --app-name default --space-name sinca-dev --profile $env:AWS_PROFILE
          Start-Sleep -Seconds 20
      }
      aws sagemaker create-app --domain-id $domain --app-type JupyterLab --app-name default --space-name sinca-dev --resource-spec $spec --profile $env:AWS_PROFILE
  }

  do {
      Start-Sleep -Seconds 15
      $status = aws sagemaker describe-app --domain-id $domain --app-type JupyterLab --app-name default --space-name sinca-dev --profile $env:AWS_PROFILE --query "Status" --output text
      Write-Host "Status: $status"
  } while ($status -ne "InService")

  Write-Host "Listo: abrir desde la consola (Studio -> Spaces -> sinca-dev -> Open)"
  ```

  `stop-space.ps1.example` — apaga el app (borra cómputo, conserva EFS/EBS):

  ```powershell
  $env:AWS_PROFILE = "AWS_PROFILE"
  $domain = "DOMAIN_ID"
  aws sagemaker delete-app --domain-id $domain --app-type JupyterLab --app-name default --space-name sinca-dev --profile $env:AWS_PROFILE
  ```

- [x] 8.3 Documentar en `README.md` (sección "Entorno de desarrollo") el flujo
  start/stop con estos scripts: copiar `.example` → `.ps1`, completar
  placeholders, y usar `.\start-space.ps1` / `.\stop-space.ps1` en vez de la
  consola (evita el quirk del app).

## 9. Autenticación git desde el Space (SSH key)

- [x] 9.1 Cambiar el clone de `infra/scripts/on_start.sh` de HTTPS a SSH:
  `git clone git@github.com:desareca/forecast-sinca-aws.git "$REPO_DIR"` (para
  que un Space nuevo quede autenticado desde el primer arranque)
- [x] 9.2 Documentar en `README.md` (sección "Entorno de desarrollo") el flujo
  de autenticación git desde el Space: generar la key SSH en el Space
  (`ssh-keygen -t ed25519 -C "sinca-dev-space"`), agregar la pública a GitHub
  (Settings → SSH and GPG keys), y cambiar el remote a
  `git@github.com:desareca/forecast-sinca-aws.git`. Aclarar que la key persiste
  en EFS (se configura una sola vez) y el patrón de dos clones (commit/push en
  uno, `git pull` en el otro).

## Notas de implementación

- **1.1 (split de `predictions.parquet`)**: se persisten `rolling_frame` como
  `split="valid"` y `test_frame` como `split="test"`. **No hay filas `train`**:
  el design §1 pide "rolling + test" y no existe un frame de predicciones de
  train (solo se calcula el `train_loss`). Validado localmente con un run
  mínimo (`--no-upload`, Nielol/pm10, 1 trial): `predictions` 28910 filas,
  `feature_importance` 22 filas, `zone_icap` 1897 filas, esquemas correctos.
- **1.1 (`zone_icap.parquet` es por serie, no por zona)**: `evaluate.zone_icap`
  necesita las 6 series para tomar el máximo por timestamp; como se loguea al
  cierre de **cada** run, el `zone_icap.parquet` de un run es el ICAP de **esa**
  serie (una estación/contaminante). El `ICAP_zona` real se calcula en el
  dashboard concatenando las predicciones de test de los 6 runs
  (`load.zone_icap_from_runs`), que es lo que muestra la pestaña "ICAP zona".
  Queda a criterio del arquitecto si además quiere un `zone_icap.parquet` global
  en un run aparte.
- **2.2 (bandas ICAP en el gráfico de concentración)**: el ICAP es un índice, no
  una concentración. En el gráfico de serie las bandas se dibujan en el
  **equivalente en µg/m³** de ICAP 100/200/300 (`icap_to_concentration`:
  pm10 150/195/240; pm25 50/80/110), no literalmente en 100/200/300. En el
  gráfico de `ICAP_zona` sí son 100/200/300 (eje índice).
- **2.3 (`width="stretch"`)**: se usa `width="stretch"` en `plotly_chart`/
  `dataframe` (Streamlit >= 1.50) en vez de `use_container_width` (deprecado).
  `AppTest` (headless, dentro del contenedor) detectó un bug inicial en las
  tuplas de bandas; corregido. Estado final: 4 tabs, 0 excepciones, métricas OK.
- **3.3 (fixture = estado MLflow real)**: `make_fixture.py` genera
  `pipeline/analysis/fixtures/state/` con `mlflow.sqlite` + `mlruns/` reales
  (7 runs: 6 LightGBM + persistencia, cada uno con los 3 parquet). Así el
  dashboard corre el mismo camino que con S3 (`--data-dir`), sin AWS.
- **4.1 (estado)**: imagen `sinca-analysis` construida; contenedor sirve en
  `:8501` (`/_stcore/health` = `ok`) y `AppTest` pasa contra el fixture. Falta
  la revisión visual del operador para cerrar la definición de gráficos.
- **5.4 (estado)**: el flujo encender/trabajar/apagar del notebook quedó
  documentado en `README.md`; falta que el operador confirme que
  `notebooks/analisis_baseline.ipynb` abre en el Space.
- **4.1 (UX cerrada por ahora, iterativa)**: se afinó el dashboard con el
  operador. Cambios respecto del diseño inicial:
  - Flujo **serie primero** + selector único **"Modelos (comparar)"** por
    `(familia, versión)` (p. ej. `lightgbm v1`, `lightgbm v2`, `dmd v1`), que
    alimenta **series e ICAP**. La familia se lee de param/tag `modelo` (o se
    parsea del `run_name`); `train.py` ya setea el tag `modelo`.
  - **ICAP_zona** se agrupa por `(modelo, versión)` (`zone_icap_by_model_version`),
    nunca mezcla familias/versiones. Respeta el **split** (test/valid).
  - **RMSE vs persistencia**: dos gráficos **lado a lado** (test y valid) para la
    **serie seleccionada**, filtrados por el selector de modelos. El de valid
    muestra **media por ventana con barras de error min–max** y tooltip con
    `media/sd/min/max`; la tabla agrega `valid_sd`.
  - El **fixture** se amplió a 19 runs (lightgbm v1/v2 + dmd v1 × 6 series +
    persistencia) para poder comparar familias.
  - El operador aclara que la revisión de gráficos es **continua**; se cierra
    4.1 por ahora y se seguirá iterando.
- **4.1 (4º diagnóstico `rolling_scores.parquet`)**: para el rango min–max del
  CV hacía falta el RMSE por ventana (antes solo media/std). `train.py` ahora
  persiste `rolling_scores.parquet` (`window_index, rmse`) junto a los otros 3
  diagnósticos. Es un agregado respecto de los 3 artefactos que listaba 1.1;
  validado con un run local (se generan los 4 parquet).
- **Feature importance — PENDIENTE Fase 4 (stand by)**: el **gráfico** es
  agnóstico (lee `feature_importance.parquet` con esquema
  `estacion, pollutant, feature, gain`; si falta, el run se omite sin romper).
  El **cálculo** es específico del modelo: LightGBM usa `feature_importance("gain")`;
  para DMD/autoencoder (Fase 4) hay que definir el método (recomendado:
  permutation importance sobre validación, agnóstico) y persistir el mismo
  esquema. Evaluar además renombrar `gain` → `importance` para no atar la
  columna a terminología de árboles. **Decisión del arquitecto**; reabrir al
  encarar Fase 4.
- **PENDIENTE — trabajar desde VS Code local en SageMaker**: el flujo de
  conexión remota (SSH-over-SSM vía AWS Toolkit → "Start and connect") está
  bloqueado por el bug `aws/aws-toolkit-vscode#7915` (el `detached-server` de la
  extensión no encuentra chunks de webpack, p. ej. `130.js`). No es un problema
  de infra ni de IAM: `aws sagemaker start-session` funciona y devuelve sesión
  válida. **Queda pendiente buscar una manera de trabajar desde VS Code local
  contra el Space** (alternativas a evaluar: método SSH CLI de la doc de remote
  access, `sagemaker-ssh-helper`, o esperar el fix del bug). Mientras tanto, el
  Space se usa por la web de JupyterLab.
- **8 (contexto — quirk del app JupyterLab)**: al apagar el Space con "Stop" se
  borra el app (el cómputo) y el Space queda `InService` con `list-apps` vacío.
  La consola a veces no recrea el app al "Open" y tira "App [default] ... is not
  found or in unusable state". Por eso los scripts `start-space.ps1` /
  `stop-space.ps1`: el start es idempotente (crea el app si falta, o lo borra y
  recrea si quedó en mal estado) y espera `InService`. Llevan profile/account
  reales → gitignored, con `.example` de placeholders como referencia.
- **9 (contexto — git auth desde el Space)**: el Space es un clone separado sin
  credenciales (rol `sinca-dev-role`), por lo que `git push` requiere
  autenticación. Se elige SSH key (`ed25519`) persistida en EFS (se configura
  una sola vez) frente a un fine-grained PAT (rotación/expiración). El
  `on_start.sh` pasa a clonar por SSH para que un Space nuevo quede autenticado
  desde el arranque. La generación de la key y el alta en GitHub son pasos del
  operador (no del implementador); el implementador solo deja el clone por SSH y
  la documentación.
- **8.2 (guardas de los scripts `.example`)**: se agregaron dos detalles sobre
  el contenido de referencia de `tasks.md`, sin cambiar el flujo:
  1. el poll de `start-space.ps1` corta también con `Failed`/`Deleting`/
     `Deleted` y termina con `throw` (el `do/while` original quedaba en loop
     infinito si el app fallaba);
  2. `Write-Host` de progreso al borrar/recrear un app en mal estado.
- **9.1 (`known_hosts` en el clone)**: `on_start.sh` clona con
  `GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=accept-new"`. Sin eso, el
  primer clone por SSH en un Space nuevo falla por verificación del host key de
  GitHub (no hay TTY para responder el prompt). El `|| true` se mantiene: si la
  key todavía no existe, el clone falla callado y el operador lo hace a mano
  después de generarla (paso documentado en README).
- **6.4 (desvío aprobado: Docker local, no el Space)**: el port-forward del 8501
  depende de la conexión remota de VS Code, bloqueada por
  `aws/aws-toolkit-vscode#7915`. Con el operador se eligió correr el dashboard
  en **Docker local contra S3 real** (misma app, mismos datos). Se levantó con
  el código montado (`-v ${PWD}:/app`): `/_stcore/health` = `ok` y `AppTest`
  sin excepciones, 4 tabs y 5 figuras. **Queda a criterio del arquitecto** si la
  tarea se reescribe como "Docker local contra S3" o se deja el Space para
  cuando se resuelva el bug.
- **6.4 (hallazgo: montar todo `~/.aws` rompe el contenedor)**: el `config` del
  Space define `ca_bundle = /home/desareca/.aws/desareca-ca-bundle.pem`; al
  montar `~/.aws` completo, boto3 intenta leer esa ruta de Linux y falla con
  `SSLError: [Errno 2] No such file or directory`. Solución documentada en
  README: montar **solo** `~/.aws/credentials` y pasar la región por
  `AWS_DEFAULT_REGION`. (Además, el shell del operador tiene `AWS_PROFILE=evolet`
  de otra cuenta: en este proyecto siempre se pasa `desareca_dev` explícito.)
- **7.1 (verificación programática)**: con el estado real descargado de S3, se
  recalculó el RMSE sobre `predictions.parquet` (split `test`) por serie y se
  comparó contra el `test_rmse` del `mlflow.sqlite`: **delta 0.0 en las 6
  series** (19.6646 / 19.4533 / 17.6652 / 16.5959 / 28.7509 / 27.7482).
  `feature_importance` 22 filas y `zone_icap` ~1.9k filas por run. El run de
  persistencia degrada a "solo métricas" sin fallar (requisito de degradación).
- **4.1 (iteración operador — candidatos del search en Valid)**: se reemplazó el
  gráfico de Valid (barras de media con rango min–max) por **una barra por
  candidato del random search** por modelo seleccionado, con el ganador marcado
  (rombo) y resaltado en el hover. Alcance: **solo visualización**, sin tocar el
  entrenamiento ni re-correr nada; el dato sale del `search_history` del JSON de
  reporte que ya loguea `train.py`.
  - `load.search_candidates` / `load.candidates_by_model` (lee el JSON del run).
  - `plots.rmse_candidates_figure`; `rmse_valid_figure` queda como fallback para
    runs sin reporte y **sin** barras de error.
  - `make_fixture.py` ahora genera el JSON de reporte con 20 candidatos señuelo
    (antes no lo generaba, y el modo fixture no tenía candidatos).
  - **Aclaración importante** (verificado con el operador): el `score` de cada
    candidato es el promedio sobre las **12 ventanas del search**, mientras que
    la tabla y `rolling_scores` usan las **42** de la validación completa. No son
    comparables entre sí; el caption del gráfico lo dice.
  - **Pendiente / decisión del arquitecto**: cajas o violines **por candidato**
    requieren persistir el score por ventana de cada trial
    (p. ej. `search_scores.parquet`, ~240 filas) y re-correr el job (~40 min).
    Hoy `random_search` calcula esos valores y los descarta. También quedó
    pendiente (descartado por el operador por ahora) el peso por antigüedad.
- **3.3 (hallazgo: el fixture está medio trackeado)**: `fixtures/state/mlflow.sqlite`
  está versionado, pero `fixtures/state/mlruns/` cae en la regla genérica
  `mlruns/` del `.gitignore` (90 archivos, ~1,15 MB). Un clone limpio tiene el
  sqlite pero **no** los artefactos → el modo fixture muestra los runs sin
  diagnósticos. No se tocó (es decisión de repo): o se versiona el fixture
  completo (negación en `.gitignore`), o se destrackea el sqlite y el fixture se
  genera siempre con `make_fixture.py` (lo que dice el README).
