@echo off
setlocal

cd /d "%~dp0\.."

set LOG_DIR=outputs\logs
set LOG_FILE=%LOG_DIR%\build_manuscript_docx.log
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set PYTHON_CMD=
where py >nul 2>nul
if not errorlevel 1 set PYTHON_CMD=py -3

if "%PYTHON_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
  echo Python was not found. Install Python 3 and add it to PATH.
  pause
  exit /b 1
)

echo Building manuscript DOCX...
echo Building manuscript DOCX... > "%LOG_FILE%"
%PYTHON_CMD% -m pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo Dependency installation failed. See %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

%PYTHON_CMD% scripts\make_docx.py manuscript\manuscript.md manuscript\output\salivary_gland_in_silico_trial_manuscript.docx >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo DOCX generation failed. See %LOG_FILE%
  type "%LOG_FILE%"
  pause
  exit /b 1
)

echo Done.
echo Output: manuscript\output\salivary_gland_in_silico_trial_manuscript.docx
echo Log: %LOG_FILE%
pause

endlocal
