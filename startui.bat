@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Run install.bat first to create .venv and install dependencies.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo Failed to activate the virtual environment.
    pause
    exit /b 1
)

echo Starting caption generator...
python "%~dp0code\caption_generator.py"
if errorlevel 1 (
    echo.
    echo Failed to launch the application.
    pause
    exit /b 1
)

pause
