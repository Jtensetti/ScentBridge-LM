$ErrorActionPreference = 'Stop'
$releaseRoot = Split-Path -Parent $PSScriptRoot
$localPython = Join-Path $releaseRoot '.venv/Scripts/python.exe'
$workspacePython = Join-Path $releaseRoot '../../work/scentbridge-env/Scripts/python.exe'
# The bundled GGUF runtime needs only Python's standard library.
$runPython = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = Join-Path $releaseRoot 'src'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$url = 'http://127.0.0.1:8765'
try { $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2 } catch { $health = $null }
if (-not $health.ready) {
    $runArgs = @('-m', 'scentbridge.server', '--model', ('"' + (Join-Path $releaseRoot 'scentbridge-q8.gguf') + '"'), '--catalog', ('"' + (Join-Path $releaseRoot 'catalog.json') + '"'), '--ui', ('"' + (Join-Path $releaseRoot 'app/index.html') + '"'))
    Start-Process -FilePath $runPython -ArgumentList $runArgs -WorkingDirectory $releaseRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $releaseRoot 'server.log') -RedirectStandardError (Join-Path $releaseRoot 'server-errors.log') | Out-Null
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try { $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2; if ($health.ready) { break } } catch { }
        Start-Sleep -Seconds 1
    }
}
if (-not $health.ready) { throw 'ScentBridge kunde inte startas. Läs server-errors.log i modellmappen.' }
Start-Process $url
