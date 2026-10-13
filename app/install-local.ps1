$ErrorActionPreference = 'Stop'
$releaseRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $releaseRoot
python --version
if ($LASTEXITCODE -ne 0) { throw 'Installera Python 3.11–3.13 först.' }
if (-not (Test-Path -LiteralPath (Join-Path $releaseRoot 'runtime/llama-server.exe'))) { throw 'Den medföljande modellmotorn saknas.' }
if (-not (Test-Path -LiteralPath (Join-Path $releaseRoot 'scentbridge-q8.gguf'))) { throw 'Modellfilen saknas.' }
Write-Output 'Inga extra Python-paket behövs för den medföljande doftverkstaden.'
Write-Output 'Klart. Öppna Starta-ScentBridge.cmd i app-mappen.'
