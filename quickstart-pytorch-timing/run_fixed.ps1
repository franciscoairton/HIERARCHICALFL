$ErrorActionPreference = "Stop"

$pyproject = Get-Content ".\pyproject.toml" -Raw
if ($pyproject -notmatch 'num-supernodes\s*=\s*(\d+)') {
    throw "Nao encontrei num-supernodes no pyproject.toml"
}
$numSupernodes = $Matches[1]

Write-Host "Configurando Flower local com $numSupernodes SuperNodes..."
python -m flwr federation simulation-config --num-supernodes $numSupernodes local

Write-Host "Iniciando simulacao Flower..."
python -m flwr run . local --stream
