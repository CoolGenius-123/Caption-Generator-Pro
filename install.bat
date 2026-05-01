@echo off
setlocal
cd /d "%~dp0"

echo Installing dependencies from requirements.txt...
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
)

echo.
echo Installation completed successfully.
pause