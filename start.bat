@echo off
rem Starts the app and opens it in your browser. Close this window to stop it.
cd /d "%~dp0backend"
echo Starting "Who is in the photo?" at http://127.0.0.1:8000 ...
uv run --no-sync python run.py
pause
