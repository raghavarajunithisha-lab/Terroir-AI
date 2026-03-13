# ============================================================
# Terroir AI - PRJNA556362 Strawberry Microbiome Download Script
# ============================================================
# Paper: "Taxonomical and functional composition of strawberry
#         microbiome is genotype-dependent"
# DOI:   10.1016/j.jare.2022.02.009
# Journal: Journal of Advanced Research, 2022
# ============================================================

$PREFETCH = "d:\ChitoseProject\sratoolkit.3.3.0-win64\bin\prefetch.exe"
$ACC_LIST = "d:\ChitoseProject\terroir_ai\SraAccList_PRJNA556362.txt"
$OUT_DIR  = "d:\ChitoseProject\terroir_ai\sra_data"

# Create output directory
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null

Write-Host "================================================" -ForegroundColor Green
Write-Host " Terroir AI - Downloading PRJNA556362 (96 runs)" -ForegroundColor Green
Write-Host " Strawberry Microbiome x Fruit Quality Study"     -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Download each accession
$accessions = Get-Content $ACC_LIST | Where-Object { $_.Trim() -ne "" }
$total = $accessions.Count
$current = 0

foreach ($acc in $accessions) {
    $current++
    Write-Host "[$current/$total] Downloading $acc ..." -ForegroundColor Cyan
    & $PREFETCH $acc --output-directory $OUT_DIR --max-size 50G
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: Failed to download $acc" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " Download complete! Files saved to: $OUT_DIR"      -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Convert SRA to FASTQ:  fasterq-dump --split-files <SRR_ID>" -ForegroundColor White
Write-Host "  2. Download the paper's Supplementary Tables from:" -ForegroundColor White
Write-Host "     https://doi.org/10.1016/j.jare.2022.02.009" -ForegroundColor White
Write-Host "     (Tables contain fruit quality traits + mineral nutrient data)" -ForegroundColor White
Write-Host "  3. Run QIIME2/DADA2 pipeline to get OTU abundance tables" -ForegroundColor White
Write-Host "  4. Build bipartite graph: OTU nodes <-> Fruit quality edges" -ForegroundColor White
