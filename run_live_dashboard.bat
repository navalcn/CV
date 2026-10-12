@echo off
echo ========================================================
echo  Live Traffic Surveillance & Telemetry Dashboard
echo ========================================================
echo.
echo Starting web server on http://localhost:8080 ...
echo Open your browser at: http://localhost:8080
echo.
echo Press Ctrl+C in this window to stop the dashboard.
echo.

if exist ".\venv\Scripts\python.exe" (
    .\venv\Scripts\python.exe live_dashboard.py --port 8080
) else (
    python live_dashboard.py --port 8080
)

pause
