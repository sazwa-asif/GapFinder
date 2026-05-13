@echo off
echo Installing frontend dependencies and starting GapFinder...
echo.
cd /d "%~dp0frontend"
npm install
npm run dev
pause

