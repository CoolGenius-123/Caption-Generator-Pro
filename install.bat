@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "PYTHON_EXE="
set "CUDA_MAX="
set "CUDA_MAJOR="
set "CUDA_MINOR="
set "PYTORCH_CUDA_TAG="
set "PYTORCH_INDEX_URL="

echo Installing dependencies for Caption Generator Pro.
echo.

for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON_EXE (
        echo %%P | findstr /i "\\WindowsApps\\python.exe" >nul
        if errorlevel 1 (
            "%%P" --version >nul 2>&1
            if not errorlevel 1 set "PYTHON_EXE=%%P"
        )
    )
)

if not defined PYTHON_EXE (
    if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/windows/
    echo During installation, enable "Add python.exe to PATH", then run this script again.
    echo.
    pause
    exit /b 1
)

echo Using Python: "%PYTHON_EXE%"
echo.

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in "%VENV_DIR%"...
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo Failed to activate the virtual environment.
    pause
    exit /b 1
)

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    call :install_pytorch_cuda
    if errorlevel 1 (
        pause
        exit /b 1
    )
) else (
    echo CUDA-enabled PyTorch is already installed.
)

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

echo Installing app dependencies from requirements.txt...
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

exit /b 0

:install_pytorch_cuda
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo NVIDIA GPU tools were not found.
    echo This app requires CUDA-enabled PyTorch, but nvidia-smi is unavailable.
    echo Install or update the NVIDIA driver, then run this script again.
    echo.
    exit /b 1
)

for /f "tokens=1,* delims=," %%A in ('nvidia-smi --query-gpu^=name^,driver_version --format^=csv^,noheader 2^>nul') do (
    echo Detected GPU: %%A
    echo NVIDIA driver:%%B
    goto :gpu_detected
)

:gpu_detected
for /f "tokens=9" %%C in ('nvidia-smi 2^>nul ^| findstr /c:"CUDA Version"') do set "CUDA_MAX=%%C"

if not defined CUDA_MAX (
    echo.
    echo Could not detect the maximum CUDA version supported by the NVIDIA driver.
    echo Update the NVIDIA driver, then run this script again.
    echo.
    exit /b 1
)

for /f "tokens=1,2 delims=." %%A in ("%CUDA_MAX%") do (
    set "CUDA_MAJOR=%%A"
    set "CUDA_MINOR=%%B"
)
if not defined CUDA_MINOR set "CUDA_MINOR=0"

if %CUDA_MAJOR% GEQ 13 set "PYTORCH_CUDA_TAG=cu128"
if "%CUDA_MAJOR%"=="12" (
    if %CUDA_MINOR% GEQ 8 (
        set "PYTORCH_CUDA_TAG=cu128"
    ) else if %CUDA_MINOR% GEQ 6 (
        set "PYTORCH_CUDA_TAG=cu126"
    )
)
if "%CUDA_MAJOR%"=="11" (
    if %CUDA_MINOR% GEQ 8 set "PYTORCH_CUDA_TAG=cu118"
)

if not defined PYTORCH_CUDA_TAG (
    echo.
    echo Detected NVIDIA CUDA support: %CUDA_MAX%
    echo PyTorch CUDA wheels require CUDA 11.8 or newer driver support.
    echo Update the NVIDIA driver, then run this script again.
    echo.
    exit /b 1
)

set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/%PYTORCH_CUDA_TAG%"

echo Detected NVIDIA CUDA support: %CUDA_MAX%
echo Installing PyTorch CUDA packages from %PYTORCH_INDEX_URL% ...
python -m pip install --upgrade torch torchvision torchaudio --index-url "%PYTORCH_INDEX_URL%"
if errorlevel 1 (
    echo.
    echo Failed to install CUDA-enabled PyTorch.
    echo You can manually choose a command from https://pytorch.org/get-started/locally/
    echo.
    exit /b 1
)

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyTorch installed, but CUDA is still not available.
    echo Update the NVIDIA driver or manually choose a command from https://pytorch.org/get-started/locally/
    echo.
    exit /b 1
)

python -c "import torch; print('Installed PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA device:', torch.cuda.get_device_name(0))"
echo.
exit /b 0
