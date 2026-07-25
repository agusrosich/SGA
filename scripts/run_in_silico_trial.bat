@echo off
setlocal

REM Censoring-aware causal survival pipeline for T3/T4 salivary gland cancer.

cd /d "%~dp0\.."

set BOOTSTRAP_ITERATIONS=1000
set SIMULATION_ITERATIONS=1000
set LOG_DIR=outputs\logs
set LOG_FILE=%LOG_DIR%\run_in_silico_trial.log

if not "%~1"=="" set BOOTSTRAP_ITERATIONS=%~1
if not "%~2"=="" set SIMULATION_ITERATIONS=%~2

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set PYTHON_CMD=
where py >nul 2>nul
if not errorlevel 1 set PYTHON_CMD=py -3

if "%PYTHON_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set PYTHON_CMD=python
)

echo ============================================================
echo Causal Survival Pipeline
echo T3/T4 disease with radiotherapy recorded
echo Bootstrap=%BOOTSTRAP_ITERATIONS%, secondary simulations=%SIMULATION_ITERATIONS%
echo Log file: %LOG_FILE%
echo ============================================================

echo Predictive Chemotherapy Benefit Pipeline > "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo Bootstrap=%BOOTSTRAP_ITERATIONS%, simulations=%SIMULATION_ITERATIONS% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if "%PYTHON_CMD%"=="" (
  echo Python was not found. Install Python 3 and add it to PATH.
  echo Python was not found. Install Python 3 and add it to PATH. >> "%LOG_FILE%"
  pause
  exit /b 1
)

echo Using Python command: %PYTHON_CMD%
echo Using Python command: %PYTHON_CMD% >> "%LOG_FILE%"
%PYTHON_CMD% --version >> "%LOG_FILE%" 2>&1

%PYTHON_CMD% -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo Dependency installation failed.
  echo See: %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

%PYTHON_CMD% src\causal_survival_pipeline.py --bootstrap %BOOTSTRAP_ITERATIONS% --simulations %SIMULATION_ITERATIONS% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo Pipeline failed.
  echo See: %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

echo.
echo Done.
echo Results: outputs\causal_survival
echo Log: %LOG_FILE%
echo.
echo Finished: %DATE% %TIME% >> "%LOG_FILE%"

pause
endlocal
