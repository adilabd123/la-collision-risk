@echo off
REM =====================================================================
REM  LA Collision Insurance Risk - one-click runner for Windows
REM  Double-click this file, or run it from a terminal in this folder.
REM  First run: creates a Python environment and installs packages
REM  (needs internet once, takes a few minutes).
REM =====================================================================
setlocal
cd /d "%~dp0"
set "GRAPH=%~dp0data\cache\test_grid.graphml"

REM ---- keep everything on this drive (nothing written to C:) ------------
REM pip's cache and temporary files normally go to C:\Users\...; send them
REM into a .tmp folder next to this file instead, and don't keep a cache.
if not exist "%~dp0.tmp" mkdir "%~dp0.tmp"
set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
set "PIP_NO_CACHE_DIR=1"
set "MPLCONFIGDIR=%~dp0.tmp\matplotlib"

REM ---- find Python ---------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
    echo Python was not found. Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

REM ---- one-time setup ------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment in .venv ...
    %PY% -m venv .venv || goto :fail
    echo Installing packages ^(first run only, a few minutes^) ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

:menu
echo.
echo =============================================================
echo   LA Collision Insurance Risk System
echo =============================================================
echo   1. Run the full pipeline (data, hotspots, models)   ~5-10 min
echo   2. Run the automated tests
echo   3. Open the dashboard (starts the API)
echo   4. Rebuild the Word report  (needs Node.js)
echo   5. Do everything: pipeline, tests, then start the API
echo   0. Exit
echo =============================================================
set "CHOICE="
set /p CHOICE="Choose an option: "
if "%CHOICE%"=="1" (call :pipeline & goto :menu)
if "%CHOICE%"=="2" (call :tests & goto :menu)
if "%CHOICE%"=="3" (call :api & goto :menu)
if "%CHOICE%"=="4" (call :report & goto :menu)
if "%CHOICE%"=="5" (call :pipeline && call :tests && call :api & goto :menu)
if "%CHOICE%"=="0" exit /b 0
echo Please type a number from the menu.
goto :menu

:pipeline
echo.
echo Running the full pipeline ...
"%VPY%" -W ignore scripts\run_pipeline_offline.py || goto :fail
"%VPY%" -W ignore scripts\compare_oversamplers.py || goto :fail
"%VPY%" -W ignore scripts\make_report_figures.py || goto :fail
echo.
echo Pipeline finished. Results are in data\processed, models and reports.
exit /b 0

:tests
echo.
echo Running tests ...
"%VPY%" -m pytest -q -p no:warnings || goto :fail
exit /b 0

:api
if not exist "models\risk_model.joblib" (
    echo No trained model yet - running the pipeline first.
    call :pipeline || exit /b 1
)
if not exist "%GRAPH%" (
    echo Road grid missing - running the pipeline first.
    call :pipeline || exit /b 1
)
echo.
echo Starting the dashboard at http://localhost:8000/dashboard  (API docs: /docs)
echo (Loading the road network takes ~20-60 s. Press Ctrl+C to stop the API.)
set "CRS_RISK_GRAPHML=%GRAPH%"
start "" cmd /c "timeout /t 25 >nul & start http://localhost:8000/dashboard"
"%VPY%" -m uvicorn api.app:app --host 127.0.0.1 --port 8000
exit /b 0

:report
where node >nul 2>nul || (echo Node.js not found. Install it from https://nodejs.org & exit /b 1)
if not exist "reports\model_comparison.csv" call :pipeline
if not exist "node_modules\docx" (call npm install docx || goto :fail)
"%VPY%" -W ignore scripts\report\build_data.py || goto :fail
node scripts\report\build_report.js || goto :fail
echo Report written to reports\LA_Collision_Insurance_Risk_Report.docx
exit /b 0

:fail
echo.
echo *** Something went wrong - see the messages above. ***
pause
exit /b 1
