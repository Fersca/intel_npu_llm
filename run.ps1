param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

python .\chat_npu_13.py @Args
exit $LASTEXITCODE
