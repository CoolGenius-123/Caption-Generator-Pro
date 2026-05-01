@echo off
setlocal
cd /d "%~dp0"

echo Starting caption generator...
python "%~dp0code\caption_generator.py"
if errorlevel 1 (
    echo.
    echo Failed to launch the application.
    pause
    exit /b 1
)

pause