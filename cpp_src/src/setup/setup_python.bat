@echo off
echo ========================================
echo  NEA-CSProject Python Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Detect if we're using Anaconda/Miniconda
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected Anaconda/Miniconda environment
    echo Installing packages with conda...
    echo.
    conda install -y numpy scipy pillow pybind11
) else (
    echo Using pip to install packages...
    echo.
    python -m pip install --upgrade pip
    python -m pip install numpy scipy pillow pybind11
)

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.

REM Verify installations
echo Verifying installations...
python -c "import numpy; print('numpy version:', numpy.__version__)"
python -c "import scipy; print('scipy version:', scipy.__version__)"
python -c "import PIL; print('Pillow version:', PIL.__version__)"
python -c "import pybind11; print('pybind11 version:', pybind11.__version__)"

echo.
echo All packages installed successfully!
echo You can now run your application.
echo.
pause