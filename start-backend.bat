@echo off
echo Starting GapFinder Backend...
echo.
cd /d "%~dp0backend"
call venv\Scripts\activate.bat
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause

