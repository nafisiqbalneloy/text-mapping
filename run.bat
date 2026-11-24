@echo off
rem Double-click to start the helper. Pass a model if you like:  run.bat qwen2.5:7b
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if "%~1"=="" (
  ".venv\Scripts\python.exe" src\chat.py
) else (
  ".venv\Scripts\python.exe" src\chat.py --model %~1
)
pause
