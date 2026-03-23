$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$thesisDir = Join-Path $scriptDir "docs\NEU-Thesis-main\NEU-Thesis-main"
$mainTex = "Thesis.tex"
$outputDir = "Tmp"
$outputPath = Join-Path $thesisDir $outputDir

if (-not (Test-Path $thesisDir)) {
    throw "Thesis directory not found: $thesisDir"
}

if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    throw "xelatex not found in PATH."
}

if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) {
    throw "bibtex not found in PATH."
}

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

Push-Location $thesisDir
try {
    Write-Host "Compiling $mainTex with bibliography..."

    & xelatex "--output-directory=$outputDir" $mainTex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 1 failed." }

    & bibtex "$outputDir/Thesis"
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed." }

    & xelatex "--output-directory=$outputDir" $mainTex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 2 failed." }

    & xelatex "--output-directory=$outputDir" $mainTex
    if ($LASTEXITCODE -ne 0) { throw "xelatex pass 3 failed." }

    $pdfPath = Join-Path $outputPath "Thesis.pdf"
    Write-Host ""
    Write-Host "Done."
    Write-Host "PDF: $pdfPath"
}
finally {
    Pop-Location
}
