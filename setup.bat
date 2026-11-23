@echo off
rem One-time setup. Needs internet. After this the app runs fully offline.
setlocal
cd /d "%~dp0"

echo [1/3] Installing Python packages ...
pushd backend
uv sync --python 3.11 || goto :fail
popd

echo [2/3] Downloading face models (about 280 MB, once) ...
uv run --project backend python scripts\download_models.py || goto :fail

echo [3/3] Building the user interface ...
pushd frontend
call npm install --no-audit --no-fund || goto :fail
call npm run build || goto :fail
popd

echo.
echo Setup complete. Double-click start.bat to run the app.
pause
exit /b 0

:fail
echo.
echo Setup failed. See the message above.
pause
exit /b 1
