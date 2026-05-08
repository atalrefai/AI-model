param(
  [string]$OutputDir = "./backups"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $OutputDir)) {
  New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$zipPath = Join-Path $OutputDir ("memory-backup-" + $ts + ".zip")

$paths = @("chat_memory", "training_memory")
$existing = @()
foreach ($p in $paths) {
  if (Test-Path $p) { $existing += $p }
}

if ($existing.Count -eq 0) {
  throw "No memory folders found to back up."
}

Compress-Archive -Path $existing -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Backup created: $zipPath"
