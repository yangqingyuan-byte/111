$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$thesisDir = Join-Path $scriptDir "docs\NEU-Thesis-main\NEU-Thesis-main"
$mainTex = "Thesis.tex"
$mainTexPath = Join-Path $thesisDir $mainTex
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

# Keep the thesis template aligned with Windows system fonts. The macOS helper
# script rewrites this option to `fontset=mac`, which breaks XeLaTeX on Windows
# because fonts such as STHeiti do not exist here.
$mainTexContent = Get-Content -LiteralPath $mainTexPath -Raw
if ($mainTexContent -match 'fontset=(mac|fandol|adobe)') {
    $updatedContent = $mainTexContent -replace 'fontset=(mac|fandol|adobe)', 'fontset=windows'
    if ($updatedContent -ne $mainTexContent) {
        Set-Content -LiteralPath $mainTexPath -Value $updatedContent -Encoding UTF8
        Write-Host "Normalized Thesis.tex fontset to windows."
    }
}

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
