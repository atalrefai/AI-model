param(
  [Parameter(Mandatory = $true)]
  [string]$BackupZip,
  [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $BackupZip)) {
  throw "Backup zip not found: $BackupZip"
}

if ($Overwrite) {
  if (Test-Path "chat_memory") { Remove-Item -Recurse -Force "chat_memory" }
  if (Test-Path "training_memory") { Remove-Item -Recurse -Force "training_memory" }
}

Expand-Archive -Path $BackupZip -DestinationPath $root -Force
Write-Host "Restore completed from: $BackupZip"
