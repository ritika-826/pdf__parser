@echo off
cd /d "%~dp0\frontend"
echo ========================================================
echo Starting PDF_Parser React Frontend...
echo ========================================================
call npm run dev
pause
