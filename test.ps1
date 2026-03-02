$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

python .\scripts\check_coverage.py
exit $LASTEXITCODE
