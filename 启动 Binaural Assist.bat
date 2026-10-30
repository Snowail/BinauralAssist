@echo off
setlocal
if exist "%~dp0Binaural Assist.exe" (
  start "" "%~dp0Binaural Assist.exe"
  exit /b
)
set "ROOT=%~dp0"
if not exist "%ROOT%python\pythonw.exe" if exist "%CD%\python\pythonw.exe" set "ROOT=%CD%\"
cd /d "%ROOT%"
if exist "python\pythonw.exe" (
  start "" "python\pythonw.exe" "launcher.pyw"
) else if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "launcher.pyw"
) else (
  echo Python runtime not found.
  pause
)
