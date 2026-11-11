# Ejecuta en serie los Training Jobs de 11.2 (change ae-reconstruccion).
#
# - Verifica que la cuenta sea la personal antes de lanzar nada.
# - Lanza cada job sin --wait y consulta su estado con describe-training-job.
# - Se detiene en el primer job que no termine en Completed.
# - Un job ya Completed con el mismo nombre se saltea (re-ejecutar es seguro).
# - Un job con el mismo nombre en otro estado detiene el script (revisar antes).
#
# Uso:
#   .\scripts\ae\run_11_2_serie.ps1            # lanza la serie
#   .\scripts\ae\run_11_2_serie.ps1 -DryRun    # solo muestra los comandos
#
# Log: %TEMP%\sinca-ae-11-2-<fecha>.log

param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$Region    = 'us-east-1'
$RoleArn   = "arn:aws:iam::${AccountId}:role/sinca-training-role"
$Image     = "${AccountId}.dkr.ecr.us-east-1.amazonaws.com/sinca-train:tf-cpu-v1"
$MaxRun    = 14400
$PollSec   = 60

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
}$Py       = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Launcher = 'pipeline/train/launch.py'
$LogFile  = Join-Path $env:TEMP ("sinca-ae-11-2-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmm'))

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

# Hiperparametros comunes de conv1d L=24 (11.2); cada job sobreescribe lo que cambia.
function New-FinalHp([hashtable]$over) {
    $hp = [ordered]@{
        mode = 'final'; arch = 'conv1d'; length = 24; d = 16
        hidden = 64; dropout = 0.1; lr = 0.001; batch_size = 128
        w_aux = 0.1; aux_input = 'true'; seeds = 0; epochs = 60; patience = 8
        dmd_after_final = 'false'; version = 2
    }
    foreach ($k in $over.Keys) { $hp[$k] = $over[$k] }
    return $hp
}

$jobs = @(
    @{ Name = 'sinca-ae-112-base-v2';   Hp = (New-FinalHp @{}) },
    @{ Name = 'sinca-ae-112-noaux-v2';  Hp = (New-FinalHp @{ aux_input = 'false' }) },
    @{ Name = 'sinca-ae-112-waux0-v2';  Hp = (New-FinalHp @{ w_aux = 0 }) },
    @{ Name = 'sinca-ae-112-waux05-v2'; Hp = (New-FinalHp @{ w_aux = 0.5 }) },
    @{ Name = 'sinca-ae-112-waux1-v2';  Hp = (New-FinalHp @{ w_aux = 1 }) },
    @{ Name = 'sinca-ae-112-d8-v2';     Hp = (New-FinalHp @{ d = 8 }) },
    @{ Name = 'sinca-ae-112-d32-v2';    Hp = (New-FinalHp @{ d = 32 }) },
    @{ Name = 'sinca-ae-112-ref-d8-v2';  Hp = [ordered]@{ mode = 'reference'; length = 24; d = 8;  version = 2 } },
    @{ Name = 'sinca-ae-112-ref-d32-v2'; Hp = [ordered]@{ mode = 'reference'; length = 24; d = 32; version = 2 } }
)

Set-Location $RepoRoot
Write-Log "Inicio. Log: $LogFile  DryRun=$DryRun  Jobs=$($jobs.Count)"

# 1) Cuenta correcta
$acc = (Invoke-Aws @('sts', 'get-caller-identity', '--profile', $Profile, '--query', 'Account', '--output', 'text')).Out
if ($acc -ne $AccountId) { throw "Cuenta inesperada para profile ${Profile}: '$acc' (esperada $AccountId). Abortado." }
Write-Log "Cuenta verificada: $acc"

if (-not (Test-Path $Py)) { throw "No existe el Python del venv: $Py" }

# 2) Serie
foreach ($job in $jobs) {
    # SageMaker no permite reutilizar un nombre de job: si el intento anterior fallo o se detuvo,
    # se relanza con sufijo -r2, -r3, ... (el run de MLflow no depende del nombre del job).
    $name = $job.Name
    $status = Get-JobStatus $name
    $attempt = 1
    while ($status -in @('Failed', 'Stopped')) {
        $attempt++
        $name = "$($job.Name)-r$attempt"
        $status = Get-JobStatus $name
    }

    if ($status -eq 'Completed') { Write-Log "SKIP $name (ya Completed)"; continue }
    if ($status -ne 'NotFound') { throw "El job $name ya existe con estado '$status'. Revisar antes de relanzar." }

    $launchArgs = @($Launcher, '--profile', $Profile, '--region', $Region,
                    '--role-arn', $RoleArn, '--image-uri', $Image,
                    '--entry-point', 'ae/run.py', '--max-run', $MaxRun, '--job-name', $name)
    foreach ($kv in $job.Hp.GetEnumerator()) {
        $v = if ($kv.Value -is [double] -or $kv.Value -is [single]) {
            ([double]$kv.Value).ToString([System.Globalization.CultureInfo]::InvariantCulture)
        } else { "$($kv.Value)" }
        $launchArgs += @('--hp', ("{0}={1}" -f $kv.Key, $v))
    }

    if ($DryRun) {
        Write-Log ("DRYRUN {0} {1}" -f $Py, ($launchArgs -join ' '))
        continue
    }

    Write-Log "LAUNCH $name"
    & $Py @launchArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) { throw "launch.py fallo para $name (exit $LASTEXITCODE)." }

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
