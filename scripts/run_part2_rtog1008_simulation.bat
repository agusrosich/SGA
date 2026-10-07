@echo off
setlocal

REM Part 2: RTOG 1008-like high-risk cohort and simulated-trial prediction.

cd /d "%~dp0\.."

set BOOTSTRAP_ITERATIONS=1000
set SIMULATION_ITERATIONS=1000
set LOG_DIR=outputs\logs
set LOG_FILE=%LOG_DIR%\run_part2_rtog1008_simulation.log

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
echo Part 2: RTOG 1008-like Simulation
echo T3-4 OR N1-3, radiotherapy recorded, comparable histology
echo Bootstrap=%BOOTSTRAP_ITERATIONS%, simulated trials=%SIMULATION_ITERATIONS%
echo Log file: %LOG_FILE%
echo ============================================================

echo Part 2: RTOG 1008-like Simulation > "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo Bootstrap=%BOOTSTRAP_ITERATIONS%, simulated trials=%SIMULATION_ITERATIONS% >> "%LOG_FILE%"
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

%PYTHON_CMD% 02_simulacion_rtog1008\simulate_rtog1008.py --bootstrap %BOOTSTRAP_ITERATIONS% --simulations %SIMULATION_ITERATIONS% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo Pipeline failed.
  echo See: %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

echo.
echo Done.
echo Results: 02_simulacion_rtog1008\outputs
echo Log: %LOG_FILE%
echo.
echo Finished: %DATE% %TIME% >> "%LOG_FILE%"

pause
endlocal
