# Ejecuta en serie los Training Jobs de ae-dmd-prep (change ae-dmd-prep, D9).
#
# - sinca-aedmd-prep-lgbm-v1: reajuste de LightGBM y persistencia en sel (train/train.py --mode refit),
#   con los hiperparametros de los runs fuente (-SourceRunIds). Imagen lgbm-v1.
# - sinca-aedmd-prep-conv1d-s{0,1,2} y sinca-aedmd-prep-gru-s{0,1,2}: AE en la particion ae-dmd
#   (ae/run.py --mode final --partition ae-dmd), con el pase DMD encadenado a cada semilla.
#   Imagen -TfTag (defecto tf-cpu-v2; plan B: tf-cpu-v1).
# - -Smoke: un solo job conv1d semilla 0, 2 epocas, experimento sinca-smoke.
# - Verifica que la cuenta sea la personal antes de lanzar nada.
# - Lanza cada job sin --wait y consulta su estado con describe-training-job.
# - Se detiene en el primer job que no termine en Completed.
# - Un job ya Completed con el mismo nombre se saltea (re-ejecutar es seguro).
# - Un job con el mismo nombre en otro estado detiene el script (revisar antes).
# - -DryRun solo imprime los comandos: no consulta ni lanza jobs (salvo sts get-caller-identity).
#
# Uso:
#   .\scripts\ae\run_ae_dmd_prep.ps1 -Smoke -DryRun
#   .\scripts\ae\run_ae_dmd_prep.ps1 -Version 2 -SourceRunIds "padre_las_casas_ii:pm25=<run>,..."  # serie v2 (defecto)
#   .\scripts\ae\run_ae_dmd_prep.ps1 -SourceRunIds "..." -TfTag tf-cpu-v1                         # plan B
#
# Log: %TEMP%\sinca-aedmd-prep-<fecha>.log

param(
    # v2 (D11): presupuesto fijo sin early stopping. v1 queda como evidencia (no se relanza).
    [ValidateRange(1, 99)][int]$Version = 2,
    [string]$TfTag = 'tf-cpu-v2',
    [string]$SourceRunIds = '',
    [string]$SmokeJobName = 'sinca-aedmd-prep-smoke',
    [switch]$Smoke,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$Region  = 'us-east-1'
$PollSec = 60

if (-not $Smoke -and [string]::IsNullOrWhiteSpace($SourceRunIds)) {
    throw '-SourceRunIds es obligatorio salvo con -Smoke (un run lightgbm fuente por serie: slug:pollutant=run_id,...).'
}
if ($SourceRunIds -and $SourceRunIds -notmatch '^[^:,=\s]+:[^:,=\s]+=[^:,=\s]+(,[^:,=\s]+:[^:,=\s]+=[^:,=\s]+)*$') {
    throw "-SourceRunIds con formato inválido: '$SourceRunIds' (esperado slug:pollutant=run_id,...)."
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
# Profile y cuenta esperada salen de .env (gitignored), nunca del script ni del entorno.
$EnvFile = Join-Path $RepoRoot '.env'
if (-not (Test-Path $EnvFile)) { throw "Falta $EnvFile. Copiar .env.example a .env y completar AWS_PROFILE y AWS_ACCOUNT_ID." }
$cfg = @{}
foreach ($line in Get-Content $EnvFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $cfg[$matches[1]] = $matches[2] }
}
$Profile   = $cfg['AWS_PROFILE']
$AccountId = $cfg['AWS_ACCOUNT_ID']
foreach ($v in @(@('AWS_PROFILE', $Profile), @('AWS_ACCOUNT_ID', $AccountId))) {
    if (-not $v[1] -or $v[1] -match '[<>]') { throw "$($v[0]) sin valor real en $EnvFile." }
}
$RoleArn   = "arn:aws:iam::${AccountId}:role/sinca-training-role"
$ImageLgbm = "${AccountId}.dkr.ecr.us-east-1.amazonaws.com/sinca-train:lgbm-v1"
$ImageTf   = "${AccountId}.dkr.ecr.us-east-1.amazonaws.com/sinca-train:$TfTag"
$Py        = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Launcher  = 'pipeline/train/launch.py'
$LogFile   = Join-Path $env:TEMP ("sinca-aedmd-prep-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmm'))

function Write-Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Invoke-Aws([string[]]$cliArgs) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & aws @cliArgs 2>$null
        return [pscustomobject]@{ Code = $LASTEXITCODE; Out = ($out -join '').Trim() }
    }
    finally {
        $ErrorActionPreference = $prev
    }
}

function Get-JobStatus([string]$name) {
    $r = Invoke-Aws @('sagemaker', 'describe-training-job', '--profile', $Profile, '--region', $Region,
                      '--training-job-name', $name, '--query', 'TrainingJobStatus', '--output', 'text')
    if ($r.Code -ne 0) { return 'NotFound' }
    return $r.Out
}

# Hiperparametros comunes del AE en la particion ae-dmd (D2/D3; misma config que la referencia).
function New-AeHp([string]$arch, [hashtable]$over) {
    $hp = [ordered]@{
        mode = 'final'; partition = 'ae-dmd'; arch = $arch; length = '24'; d = '16'
        hidden = '64'; dropout = '0.1'; lr = '0.003'; batch_size = '128'
        w_aux = '0.1'; aux_input = 'true'; seeds = '0'; epochs = '60'; patience = '8'
        dmd_after_final = 'true'; version = "$Version"; mlflow_runs = 'none'; early_stopping = 'off'
    }
    foreach ($k in $over.Keys) { $hp[$k] = $over[$k] }
    return $hp
}

# Orden: primero el reajuste de LightGBM (barato y sin red), luego los AE por semilla.
if ($Smoke) {
    $jobs = @(
        @{ Name = $SmokeJobName; Image = $ImageTf; Entry = 'ae/run.py'; MaxRun = 3600;
           Hp = (New-AeHp 'conv1d' @{ seeds = '0'; epochs = '2'; experiment = 'sinca-smoke' }) }
    )
} else {
    $jobs = @(
        @{ Name = "sinca-aedmd-prep-lgbm-v$Version"; Image = $ImageLgbm; Entry = 'train/train.py'; MaxRun = 3600;
           Hp = [ordered]@{ mode = 'refit'; partition = 'ae-dmd'; version = "$Version"; mlflow_runs = 'none';
                            source_run_ids = $SourceRunIds; rounds_from_source = 'true' } }
    )
    foreach ($seed in 0, 1, 2) {
        $jobs += @{ Name = "sinca-aedmd-prep-conv1d-s$seed-v$Version"; Image = $ImageTf; Entry = 'ae/run.py'; MaxRun = 7200;
                    Hp = (New-AeHp 'conv1d' @{ seeds = "$seed" }) }
    }
    foreach ($seed in 0, 1, 2) {
        $jobs += @{ Name = "sinca-aedmd-prep-gru-s$seed-v$Version"; Image = $ImageTf; Entry = 'ae/run.py'; MaxRun = 10800;
                    Hp = (New-AeHp 'gru' @{ seeds = "$seed" }) }
    }
}

Set-Location $RepoRoot
Write-Log "Inicio. Log: $LogFile  DryRun=$DryRun  TfTag=$TfTag  Jobs=$($jobs.Count)"

# 1) Cuenta correcta (única llamada a AWS en -DryRun)
$acc = (Invoke-Aws @('sts', 'get-caller-identity', '--profile', $Profile, '--query', 'Account', '--output', 'text')).Out
if ($acc -ne $AccountId) { throw "Cuenta inesperada para profile ${Profile}: '$acc' (esperada $AccountId). Abortado." }
Write-Log "Cuenta verificada: $acc"

if (-not (Test-Path $Py)) { throw "No existe el Python del venv: $Py" }

# 2) Serie
foreach ($job in $jobs) {
    $name = $job.Name

    if ($DryRun) {
        $launchArgs = @($Launcher, '--profile', $Profile, '--region', $Region, '--role-arn', $RoleArn,
                        '--image-uri', $job.Image, '--entry-point', $job.Entry, '--max-run', $job.MaxRun, '--job-name', $name)
        foreach ($kv in $job.Hp.GetEnumerator()) { $launchArgs += @('--hp', ("{0}={1}" -f $kv.Key, $kv.Value)) }
        Write-Log ("DRYRUN {0} {1}" -f $Py, ($launchArgs -join ' '))
        continue
    }

    # SageMaker no permite reutilizar un nombre de job: si el intento anterior fallo o se detuvo,
    # se relanza con sufijo -r2, -r3, ... (el run de MLflow no depende del nombre del job).
    $status = Get-JobStatus $name
    $attempt = 1
    while ($status -in @('Failed', 'Stopped')) {
        $attempt++
        $name = "$($job.Name)-r$attempt"
        $status = Get-JobStatus $name
    }

    if ($status -eq 'Completed') { Write-Log "SKIP $name (ya Completed)"; continue }
    if ($status -ne 'NotFound') { throw "El job $name ya existe con estado '$status'. Revisar antes de relanzar." }

    $launchArgs = @($Launcher, '--profile', $Profile, '--region', $Region, '--role-arn', $RoleArn,
                    '--image-uri', $job.Image, '--entry-point', $job.Entry, '--max-run', $job.MaxRun, '--job-name', $name)
    foreach ($kv in $job.Hp.GetEnumerator()) { $launchArgs += @('--hp', ("{0}={1}" -f $kv.Key, $kv.Value)) }

    Write-Log "LAUNCH $name"
    # El SDK de SageMaker escribe warnings (deprecacion v2) por stderr: con 'Stop' PowerShell
    # los tomaria como error. Se valida por el exit code de launch.py, no por stderr.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Py @launchArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile -Append
        $launchExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $prevEap
    }
    if ($launchExit -ne 0) { throw "launch.py fallo para $name (exit $launchExit)." }

    Start-Sleep -Seconds 10
    $status = Get-JobStatus $name
    if ($status -eq 'NotFound') { throw "Despues de lanzar, $name no aparece en SageMaker." }
    Write-Log "POLL $name -> $status"

    while ($status -in @('Starting', 'InProgress', 'Stopping')) {
        Start-Sleep -Seconds $PollSec
        $status = Get-JobStatus $name
    }
    Write-Log "END $name -> $status"

    if ($status -ne 'Completed') { throw "$name termino en '$status'. Serie detenida." }
}

Write-Log ("Serie terminada. Jobs en el log: {0}" -f $LogFile)
exit 0
