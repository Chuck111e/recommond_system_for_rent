$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
python -m pip install -e ".[dev]"
python -m rental_reco.cli demo
