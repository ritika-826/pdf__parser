@echo off
cd /d "%~dp0\backend"
echo ========================================================
echo Starting PDF_Parser FastAPI Backend...
echo ========================================================
".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
