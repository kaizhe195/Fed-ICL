$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment Python was not found at $PythonExe. Activate your environment manually and update this script if needed."
}

$Configs = @(
    "configs/main_mmlu_random_seed42.yaml",
    "configs/main_mmlu_random_seed43.yaml",
    "configs/main_mmlu_random_seed44.yaml",
    "configs/main_mmlu_knn_seed42.yaml",
    "configs/main_mmlu_knn_seed43.yaml",
    "configs/main_mmlu_knn_seed44.yaml"
)

foreach ($Config in $Configs) {
    $ConfigPath = Join-Path $ProjectRoot $Config
    if (-not (Test-Path $ConfigPath)) {
        throw "Missing config: $ConfigPath"
    }

    $ConfigContent = Get-Content $ConfigPath -Raw
    $OutputDirRelative = [regex]::Match($ConfigContent, 'output_dir:\s*"([^"]+)"').Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($OutputDirRelative)) {
        throw "Could not parse output_dir from $Config"
    }

    $OutputDir = Join-Path $ProjectRoot $OutputDirRelative
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    $LogPath = Join-Path $OutputDir "run.log"

    Write-Host "Running $Config"
    & $PythonExe "scripts/run_mmlu.py" --config $Config *>&1 | Tee-Object -FilePath $LogPath

    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $Config. See $LogPath"
    }
}

Write-Host "All main experiments completed."
