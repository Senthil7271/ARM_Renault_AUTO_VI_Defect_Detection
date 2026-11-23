@echo off
title AutoVI Industrial QC Cockpit
cd /d "%~dp0"
echo =========================================================================
echo  AUTOVI INDUSTRIAL OPERATOR ASSISTANCE DASHBOARD
echo  Opening on: http://localhost:5000
echo =========================================================================
start "" "http://localhost:5000"
"C:\Users\linga\AppData\Local\Programs\Python\Python310\python.exe" app.py --category tank_screw --mode conveyor --port 5000
pause
