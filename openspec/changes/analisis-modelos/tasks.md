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
- [x] 5.4 Verificar que `notebooks/analisis_baseline.ipynb` (base, provisto por el arquitecto) abre en el Space, y documentar en `README.md` el flujo de notebooks: encender (Studio → Spaces → `sinca-dev` → Open), trabajar (rol del Space, sin credenciales), y apagar (Spaces → `sinca-dev` → Stop) para evitar sobrecostos

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

- [x] 7.1 Ver el dashboard con datos reales: series real vs pred, `ICAP_zona`, RMSE vs persistencia y feature importance, confirmando con output/captura real que coinciden con las métricas de MLflow
  <!-- 2026-10-26: verificado. Automático: RMSE recalculado de predictions.parquet == test_rmse de MLflow (delta 0.0 en las 6 series) y AppTest con S3 real (4 tabs, 0 excepciones). Visual: el operador revisó el dashboard contra S3 real e iteró la pestaña RMSE hasta aprobarla. -->

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

## 10. Reparación del Space (imagen SageMaker + lifecycle config robusta)

Motivo: el Space `sinca-dev` no reconstruye el app desde cero: el
`default_resource_spec` no define imagen (`SageMaker Image ARN is required`) y
`on_start.sh` falla por el paquete inexistente `@opencode-ai/opencode` bajo
`set -euo pipefail`. Esto bloqueaba la tarea 5.4. Se corrige acá (es el
prerrequisito directo de "verificar que el notebook abre en el Space").

- [x] 10.1 `infra/variables.tf`: agregar `sagemaker_image_arn` (default
  `arn:aws:sagemaker:us-east-1:885854791233:image/sagemaker-distribution-cpu`)
  y `sagemaker_image_version_alias` (default `"3.2.0"`). Imagen pública de
  SageMaker, no de la cuenta del proyecto.
- [x] 10.2 `infra/sagemaker-domain.tf`: en
  `space_settings.jupyter_lab_app_settings.default_resource_spec` agregar
  `sagemaker_image_arn` + `sagemaker_image_version_alias` referenciando las
  variables de 10.1.
- [x] 10.3 `infra/scripts/start-space.ps1.example`: sumar al `$spec` de
  `create-app` `SageMakerImageArn=...` y `SageMakerImageVersionAlias=3.2.0`
  (propaga la fuente única de verdad).
- [x] 10.4 `infra/scripts/on_start.sh`: reescribir con log + `_try`/`_fail`:
  - `set -uo pipefail` (sacar `-e`); funciones `log`, `_fail` (exit 1), `_try`
    (loguea `WARN (no bloqueante)` y sigue).
  - `tee -a ~/.on_start.log` para toda la salida; marker begin/end por corrida.
  - conveniencia (`aws-cli`, `terraform`, `git-clone`) por `_try`; el clone SSH
    mantiene `StrictHostKeyChecking=accept-new` y la guarda `[ ! -d "$REPO_DIR/.git" ]`.
  - quitar el bloque de Node.js + Open Code CLI (no se usa en el Space; era el
    bloqueante).
  - sin pasos `_fail` hoy (el app arranca sin ninguna CLI).
- [x] 10.5 `modules/dev-environment/blueprint.md`: línea 52 quitar "Node.js +
  Open Code CLI"; en `## Notas de este proyecto` registrar: imagen fijada
  (arn + alias `3.2.0`), LCC `_try`/`_fail` + log EFS, opencode fuera del Space.
- [x] 10.6 `project-decisions.md` §3: fila `dev-environment` — quitar
  "Node+OpenCode" de la descripción de la lifecycle config.

## 11. Ejecución real (operador corre y pega output)

- [x] 11.1 `terraform -chdir=infra init -backend-config=backend.hcl` y
  `terraform -chdir=infra plan` → pegar output (verificar que `default_resource_spec`
  muestra `sagemaker_image_arn` + alias).
- [x] 11.2 `terraform -chdir=infra apply` (deny list → lo corre el operador) →
  pegar output (actualiza la LCC `sinca-on-start` y las `space_settings`).
- [x] 11.3 `Copy-Item infra/scripts/start-space.ps1.example infra/scripts/start-space.ps1`
  + completar placeholders (`AWS_PROFILE`, `DOMAIN_ID`, `AWS_ACCOUNT_ID`) +
  `.\infra\scripts\start-space.ps1` → app `InService`.
- [x] 11.4 Verificación (criterio e = tarea 5.4): abrir
  `notebooks/analisis_baseline.ipynb` en JupyterLab del Space y leer
  `~/.on_start.log` (terminal de JupyterLab): corrida limpia; el `git-clone`
  puede quedar en `WARN` solo si falta la key SSH en EFS nuevo (documentado en
  README). Pegar output real.

Contingencia: si `terraform apply` rechaza el alias `3.2.0`, setear
`sagemaker_image_version_alias = null` (solo arn) y replicar el flag en el
`$spec`; registrarlo en Notas de implementación.

## 12. Cierre (higiene)

- [x] 12.1 `infra/scripts/on_start.sh`: quitar `install_terraform` (función) y su
  línea `_try "Terraform" install_terraform`; ajustar el comentario de cabecera
  si lista terraform. Terraform no se usa en el Space (corre en local/CI); sacar
  el paso elimina el `WARN` recurrente y deja el log `END (ok)` sin ruido.
- [x] 12.2 `project-decisions.md` §3: fila `dev-environment` — quitar "terraform"
  de la lifecycle config (deja "reinstala aws-cli si falta y clona el repo").
- [x] 12.3 `docs/space-sagemaker.md`: §4.2 quitar la línea `WARN ... Terraform`
  del ejemplo de log; §6 cerrar el pendiente `apt-get` sin `sudo` con la decisión
  "terraform fuera del Space (corre en local/CI), ya no se instala en la LCC".
- [x] 12.4 `notebooks/analisis_baseline.ipynb`: limpiar outputs y execution_count
  (`jupyter nbconvert --clear-output --inplace notebooks/analisis_baseline.ipynb`
  o equivalente) + commit atómico.
- [x] 12.5 (operador) `.\infra\scripts\stop-space.ps1` → apagar el Space (no es
  opcional, factura ~USD 0.06/h).
  <!-- 2026-10-29: list-apps muestra el app `default` en `Deleted` → el Space ya
       está apagado (sin cómputo facturable; el EFS queda, no factura). -->
- [x] 12.6 Nota (registrada, no se ejecuta acá): abrir change `cicd` futuro para
  decidir el `apply -auto-approve` de `infra-cd.yml` en cada push a `infra/**`
  (hoy el apply manual compite por el lock). No bloquea archivar.

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
  candidato del random search**, con selector de modelo junto al título de Valid,
  orden de mejor a peor, la mejor resaltada y los hiperparámetros visibles (hover
  + tabla desplegable). La tabla resumen pasó arriba de los gráficos (desplegable)
  y la de hiperparámetros abajo del todo. Alcance: **solo visualización**, sin
  tocar el entrenamiento ni re-correr nada; el dato sale del `search_history` del
  JSON de reporte que ya loguea `train.py`.
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
- **10.4 (log fuera de `~` a propósito)**: el `tee` escribe en
  `/home/sagemaker-user/.on_start.log` (ruta explícita) en vez de
  `~/.on_start.log`: el lifecycle config puede correr como **root**, y en ese
  caso `~` es `/root`, que no está en EFS ni es lo que se lee desde la terminal
  de JupyterLab (tarea 11.4). Se mantiene el criterio de 10.4, cambia solo la
  resolución de la ruta.
- **10.4 (verificación local del patrón, sin AWS)**: con una copia del script en
  `/tmp` (rutas de repo/log redirigidas y las instalaciones reemplazadas por
  no-ops) se confirmó el comportamiento pedido: una conveniencia que falla
  termina `EXIT=0` con `WARN (no bloqueante)` en el log, y un `_fail` termina
  `EXIT=1` **a través del `tee`** (lo propaga `pipefail`). Sin esa prueba, el
  `main | tee` habría tragado el código de salida de `_fail`.
- **10.4 (apt-get sin `sudo`)**: los `apt-get` de `install_terraform` quedaron
  igual que antes (sin `sudo`), asumiendo que la LCC corre como root. Si en
  11.4 el paso queda en `WARN` con error de permisos, es eso: el `_try` lo deja
  pasar sin tumbar el app, y hay que ajustar el script.
- **10.3 (el ARN de imagen no sale de `AWS_ACCOUNT_ID`)**: la imagen es pública
  de SageMaker (cuenta `885854791233`), no de la cuenta del proyecto, así que en
  `start-space.ps1.example` el ARN va literal con un comentario que lo aclara;
  solo el ARN de la lifecycle config usa el placeholder `AWS_ACCOUNT_ID`.
- **10.5 (ajuste de consistencia fuera del texto de la tarea)**: además de la
  línea 52, se actualizó el bullet de "Parámetros a definir por proyecto" que
  decía "además de Node + Open Code CLI" (quedaba contradictorio al sacarlo).
  No cambia ninguna decisión: es el mismo cambio aplicado a otra mención.
- **10.2 (efecto esperado en la recreación del app)**: al fijar
  `sagemaker_image_arn` en el `default_resource_spec` del Space, todo `CreateApp`
  (consola o `start-space.ps1`) hereda imagen + alias. Esto es lo que faltaba:
  `create-app` sin imagen falla con `SageMaker Image ARN is required` y la
  consola muestra el cartel engañoso "Permissions not configured correctly".
  **No verificado contra AWS todavía** (eso es 11.1–11.4, operador).
- **11.1 (plan real, corre el operador)**: `init` falla con
  `-backend-config=backend.hcl` en el Terraform local (`1.15.7`): interpreta el
  flag como argumento posicional → `Too many command line arguments`. Con
  `-backend-config backend.hcl` (espacio, sin `=`) inicializa bien contra el
  backend S3 (`549024266383-tfstate`, `terraform-locks`, `desareca_dev`).
  Plan: **1 add / 3 change / 1 destroy**.
  - `aws_sagemaker_space.dev`: agrega `sagemaker_image_arn` +
    `sagemaker_image_version_alias` (lo buscado) y recalcula
    `lifecycle_config_arn` (known after apply).
  - `aws_sagemaker_studio_lifecycle_config.on_start`: **se reemplaza** (el
    contenido fuerza reemplazo; `-/+`, no update in-place). Riesgo: por defecto
    Terraform destruye la LCC vieja **antes** de crear la nueva, y el Space la
    referencia (`LifecycleConfigArn`). Si AWS rechaza borrar una LCC en uso, el
    apply corta ahí. Además cambia el ARN, que el Space toma después.
  - `aws_sagemaker_domain.studio` y `aws_sagemaker_user_profile.admin`: solo
    remoción de bloques que Terraform no declara (`studio_web_portal_settings`
    vacío en el dominio; `jupyter_lab_app_settings.lifecycle_config_arns` en el
    user profile, que la consola había dejado a nivel usuario). Sin efecto sobre
    la LCC del Space. Se aceptan.
- **11.1 (riesgo abierto, no decidido acá)**: si el `apply` falla al destruir la
  LCC en uso, la salida no es improvisar comandos contra la consola: hay que
  volver al arquitecto (posibles salidas: `create_before_destroy`, renombrar el
  recurso, o desacoplar la LCC del Space antes de reemplazarla).
- **11.2 (`apply` ejecutado por el operador — riesgo 1 NO materializado)**: el
  reemplazo `-/+` de la LCC salió limpio: AWS permitió `Destroying...
  Destruction complete after 0s` y el `Creating...` inmediato con el mismo
  nombre (`1 added, 3 changed, 1 destroyed`). No hizo falta tocar arquitectura.
  Verificado después por API (no por el output del apply):
  - `describe-space` → `SageMakerImageArn =
    ...:885854791233:image/sagemaker-distribution-cpu`,
    `SageMakerImageVersionAlias = 3.2.0`, `InstanceType = ml.t3.large`,
    `LifecycleConfigArn` apuntando a `sinca-on-start`.
  - `describe-studio-lifecycle-config` → el contenido decodificado contiene
    `_try` y no menciona `opencode`: es el script nuevo.
  - Queda un registro de app `default` en `Failed` de los intentos previos (se
    autoborra y solo es visible 24 h en `list-apps`); no impide `create-app`.
    Es justamente el caso que motiva 11.3.
- **11.4 (log real — WARN de Terraform CONFIRMADO, no bloqueante)**: la LCC corre
  como **`sagemaker-user`, no como root** (el header del log lo muestra:
  `user=sagemaker-user`). Resultado real de `~/.on_start.log`:
  ```
  OK: AWS CLI v2
  E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
  E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
  WARN (no bloqueante): 'Terraform' fallo (exit 100) — el Space arranca igual
  OK: repo clonado por SSH
  === on_start END (ok) ===
  ```
  Consecuencias: (a) el criterio de 11.4 ("corrida limpia") **no se cumple al
  pie de la letra**: hay un `WARN` que no es el de `git-clone`; (b) `terraform`
  no queda disponible dentro del Space, aunque la fila `dev-environment` de
  `project-decisions.md` §3 dice que la lifecycle config lo reinstala; (c) el
  `set -uo pipefail` + `_try` cumplieron: el fallo no tumbó el app.
  Arreglo candidato (una línea por comando, no aplicado): prefijar `sudo` en
  los `apt-get`, en el `gpg --dearmor` y en el `echo` a
  `/etc/apt/sources.list.d/`. Requiere otro `apply` + reinicio del Space → **lo
  decide el arquitecto / el operador**, no el implementador.
- **11.4 (hallazgo: la policy del rol dev no permite LISTAR `sinca-mlflow`)**: la
  celda 4 del notebook falla con
  `AccessDenied ... s3:ListBucket on arn:aws:s3:::sinca-mlflow`. La policy
  (`infra/iam-dev.tf`) incluye `s3:ListBucket` sobre
  `arn:aws:s3:::sinca-mlflow/_mlflow` y `.../_mlflow/*`, pero `ListBucket`
  **exige el ARN del bucket**, no un prefijo: esas dos entradas son inútiles
  para listar. Lo que sí funciona es `GetObject` (celda 2 baja
  `_mlflow/mlflow.sqlite` sin problema) y `sinca-data` (tiene el ARN del
  bucket → celda 3 OK). **Alcance real**: además del notebook, rompe
  `mlflow_log.download_state()` (pagina con `list_objects_v2`) → el "dashboard
  en el Space" del README falla igual bajo el rol dev. Arreglo mínimo
  propuesto (no aplicado): `s3:ListBucket` sobre `arn:aws:s3:::sinca-mlflow`
  con condición `StringLike {"s3:prefix": ["_mlflow/", "_mlflow/*"]}`, que es
  exactamente el mínimo privilegio que se quiso expresar. Es IAM → decisión
  del arquitecto + `apply` del operador.
- **11.4 / 5.4 (VERIFICADO con datos reales del notebook en el Space)**: el
  notebook corrió completo en JupyterLab (`Run All`) y los outputs quedaron
  commiteados (`6a3eb83`), así que la evidencia es reproducible desde el repo:
  - celda 2: **7 runs** `FINISHED` (6 LightGBM + `sinca-persistence-baseline`).
  - celda 3: lista `validated/sinca/estacion=.../year=.../month=...` y lee el
    parquet (rol dev → `ListBucket` sobre `sinca-data` OK).
  - celda 4: **`pred_keys` = 6** rutas `_mlflow/mlruns/<run_id>/artifacts/
    predictions.parquet` (seis series), y lee el parquet con el esquema
    `timestamp, estacion, pollutant, y_true, y_pred, split`.
  - celda 5: `<Figure size 1200x400 with 1 Axes>` → el gráfico real vs pred
    se dibuja.
  Con esto se cierra también la 5.4 (el notebook abre y corre en el Space).
- **11.4 (desvío de higiene, decidido por el operador)**: el notebook
  versionado quedó **con outputs y execution_count** (`6a3eb83`, +774 líneas)
  porque el operador lo commiteó como prueba de push desde el Space. No rompe
  nada, pero ensucia el diff: si el arquitecto quiere, corresponde un commit
  que lo devuelva a "sin outputs".
- **11.2/11.4 (CI aplica solo)**: `.github/workflows/infra-cd.yml` corre
  `plan` **y `apply -auto-approve`** en cada push a `main` que toque `infra/**`.
  El `apply` manual del operador es redundante y compite por el lock de DynamoDB
  (`runner@runnervm48cm3`). La policy de IAM terminó aplicada por CI: verificado
  por API (`s3:ListBucket` sobre `sinca-mlflow` con `StringLike s3:prefix =
  ["_mlflow/", "_mlflow/*"]`).
- **11.4 (`-backend-config` con `=` no funciona)**: en el Terraform local
  (`1.15.7`) hay que usar `-backend-config backend.hcl` (separado por espacio).
  El CI usa la versión `1.9.8` y `-backend-config="bucket=..."` sí le funciona:
  no es un problema del workflow, es del binario local.
- **Cierre (runbook)**: todo el diagnóstico de esta sección quedó documentado en
  `docs/space-sagemaker.md` (pedido del operador para futuras sesiones): piezas
  del entorno, ciclo normal, tabla síntoma → causa, las 5 causas conocidas con
  su fix, y el checklist de "recrear desde cero" como criterio de aceptación.
  `README.md` apunta a ese documento desde "Entorno de desarrollo". Los valores
  de cuenta/dominio van como placeholders, igual que en los `.ps1.example`.
- **12.1 (verificación real post-push)**: el push de `baee91d` disparó el CI, que
  hizo `plan` + `apply` y **reemplazó la LCC** en ~40 s. Verificado por API:
  - `describe-studio-lifecycle-config` → el contenido decodificado **no** contiene
    `install_terraform` y es **idéntico byte a byte** al `on_start.sh` del repo
    (salvo el EOL final).
  - El `lifecycle_config_arn` del Space apunta al `sinca-on-start` recreado.
  Consecuencia en el próximo arranque: el log debe quedar `BEGIN → OK: AWS CLI v2
  → OK: repo clonado por SSH → END (ok)`, sin ningún `WARN` (el de Terraform ya no
  existe). Eso no se ejecutó en esta sesión: es 12.5 (operador).
- **12.3 (docs coherentes con la decisión)**: en `docs/space-sagemaker.md` el
  ejemplo de log de §4.2 ya no incluye la línea de `WARN ... Terraform`, y §6
  reemplaza el pendiente "apt-get sin sudo" por una subsección
  "Cerrado: Terraform ya no se instala en el Space" con la decisión tomada.
- **12.4 (notebook sin outputs)**: `jupyter nbconvert --clear-output --inplace`.
  Estado verificado: las 5 celdas de código con `outputs: []` y
  `execution_count: null` (−595 líneas). Commit atómico separado del cambio de
  LCC. Recordar: si se vuelve a correr en el Space y se commitea, vuelven los
  outputs (pasó en `6a3eb83`).
