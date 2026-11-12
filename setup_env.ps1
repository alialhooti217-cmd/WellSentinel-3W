# WellSentinel-3W — one-time environment setup (Windows PowerShell)
# Run from the project folder:   powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m ipykernel install --user --name wellsentinel --display-name "Python (WellSentinel)"
Write-Host "`nDone. In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> .venv" -ForegroundColor Green
Write-Host "Next: Terminal -> Run Task -> '2. Build inventory + folds'" -ForegroundColor Green
