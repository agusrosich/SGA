@echo off
setlocal

REM Publication pipeline: chemotherapy benefit in T3/T4 salivary/parotid cancer
REM Default simulation size follows the RTOG 1008 Phase III planned sample size: N=252.

cd /d "%~dp0\.."

set TRIAL_N=252
set BOOTSTRAP_ITERATIONS=1000
set LOG_DIR=outputs\logs
set LOG_FILE=%LOG_DIR%\run_in_silico_trial.log

if not "%~1"=="" set TRIAL_N=%~1
if not "%~2"=="" set BOOTSTRAP_ITERATIONS=%~2

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set PYTHON_CMD=
where py >nul 2>nul
if not errorlevel 1 set PYTHON_CMD=py -3

if "%PYTHON_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set PYTHON_CMD=python
)

echo ============================================================
echo Predictive Chemotherapy Benefit Pipeline
echo T3/T4 disease: chemoradiation vs radiation alone
echo Simulated trial N=%TRIAL_N%, bootstrap=%BOOTSTRAP_ITERATIONS%
echo Log file: %LOG_FILE%
echo ============================================================

echo Predictive Chemotherapy Benefit Pipeline > "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo Trial N=%TRIAL_N%, bootstrap=%BOOTSTRAP_ITERATIONS% >> "%LOG_FILE%"
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

%PYTHON_CMD% src\parotid_chemo_benefit_pipeline.py --trial-n %TRIAL_N% --bootstrap %BOOTSTRAP_ITERATIONS% >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo Pipeline failed.
  echo See: %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

copy /Y outputs\tables\publication_results.json web\publication_results.json >nul
if errorlevel 1 (
  echo Could not copy publication_results.json to web.
  echo Could not copy publication_results.json to web. >> "%LOG_FILE%"
  pause
  exit /b 1
)

echo.
echo Done.
echo Tables: outputs\tables
echo Summary: outputs\reports\publication_summary.txt
echo Focused in-silico trial output: outputs\in_silico_salivary_gland_trial
echo Log: %LOG_FILE%
echo.
echo Finished: %DATE% %TIME% >> "%LOG_FILE%"

pause
endlocal
