@echo off
echo ========================================
echo  NEA-CSProject Environment Setup
echo ========================================
echo.

REM 
echo [1] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists.
    choice /C YN /M "Recreate it"
    if errorlevel 2 goto :skip_venv_create
    echo Removing old venv...
    rmdir /s /q venv
)

python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo [2] Activating virtual environment...
call venv\Scripts\activate

:skip_venv_create
if not defined VIRTUAL_ENV (
    call venv\Scripts\activate
)

echo.
echo [3] Installing dependencies...
setup_python.bat

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo To activate this environment in future:
echo     venv\Scripts\activate
echo.
pause