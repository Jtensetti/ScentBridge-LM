$ErrorActionPreference = 'Stop'
Invoke-RestMethod 'http://127.0.0.1:8765/api/shutdown' -Method Post -ContentType 'application/json' -Body '{}' -TimeoutSec 10
