param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$MonitorRoot = $PSScriptRoot
$MonitorUrl = "http://127.0.0.1:$Port"
try {
    $MonitorHealth = Invoke-RestMethod "$MonitorUrl/api/health" -TimeoutSec 2
    if ($MonitorHealth.service -eq 'a800-monitor') {
        Write-Output "A800 monitor is already running: $MonitorUrl"
        exit 0
    }
} catch {}
$MonitorPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $MonitorPython)) {
    throw 'Python runtime not found. Start server.py with an installed Python 3.11+ interpreter.'
}
$MonitorProcess = Start-Process -FilePath $MonitorPython -ArgumentList @('-u', ('"' + (Join-Path $MonitorRoot 'server.py') + '"'), '--port', "$Port") -WorkingDirectory $MonitorRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $MonitorRoot 'server.stdout.log') -RedirectStandardError (Join-Path $MonitorRoot 'server.stderr.log')
for ($MonitorAttempt = 0; $MonitorAttempt -lt 15; $MonitorAttempt++) {
    Start-Sleep -Milliseconds 400
    if ($MonitorProcess.HasExited) {
        throw (Get-Content -LiteralPath (Join-Path $MonitorRoot 'server.stderr.log') -Raw)
    }
    try {
        $MonitorHealth = Invoke-RestMethod "$MonitorUrl/api/health" -TimeoutSec 2
        if ($MonitorHealth.service -eq 'a800-monitor') {
            Write-Output "A800 monitor started: $MonitorUrl (PID $($MonitorProcess.Id))"
            exit 0
        }
    } catch {}
}
throw 'Monitor did not respond; inspect server.stderr.log.'
