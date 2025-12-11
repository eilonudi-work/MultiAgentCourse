@echo off
REM Windows batch script for running the experiment

echo ========================================
echo Starting Experiment Runner
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment not found
    echo Please run setup first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo Warning: Ollama service is not running
    echo Please start Ollama and try again
    pause
    exit /b 1
)

echo Ollama service is running
echo.

REM Run the experiment
cd src
python baseline_experiment.py %*

if errorlevel 1 (
    echo.
    echo Experiment failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Experiment Complete!
echo ========================================
echo.
echo Results saved to 'results\' directory
echo.

cd ..
pause
