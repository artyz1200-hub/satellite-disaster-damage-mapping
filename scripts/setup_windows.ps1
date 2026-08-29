param(
    [Parameter(Mandatory = $true)]
    [int]$DownloadPid
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

Write-Output "Waiting for xBD downloader PID $DownloadPid"
Wait-Process -Id $DownloadPid

$archive = Join-Path $ProjectRoot 'xbd-dataset.zip'
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "Downloader exited without creating $archive"
}

Write-Output 'Checking ZIP central directory'
tar.exe -tf $archive | Out-Null

Write-Output 'Extracting xBD archive'
tar.exe -xf $archive

Write-Output 'Validating dataset layout and split assignment'
& .\.venv\Scripts\python.exe .\scripts\prepare_splits.py --src .\xbd --out .\data --dry-run
if ($LASTEXITCODE -ne 0) {
    throw "prepare_splits.py validation failed with exit code $LASTEXITCODE"
}

Write-Output 'LOCAL SETUP COMPLETE: archive extraction and split validation succeeded.'
Write-Output 'Next: run notebooks/data_cleaning_eda.ipynb to create the shared patch arrays.'
