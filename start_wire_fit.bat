@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "APP_PY=%~dp0native_app.py"
if not exist "%APP_PY%" (
  echo Could not find native_app.py
  pause
  exit /b 1
)
set "PYTHON_EXE=C:\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" "%APP_PY%"
if errorlevel 1 (
  echo.
  echo Wire Fit Detection failed to start.
  pause
  exit /b 1
)
