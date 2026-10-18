@echo off
title ContractShield Backend
echo ===================================================
echo   Starting ContractShield AI Legal Assistant Backend
echo   FastAPI API running at: http://127.0.0.1:8000
echo   Interactive Swagger Docs: http://127.0.0.1:8000/docs
echo ===================================================
echo.

set PYTHON_PATH="C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe"

if exist %PYTHON_PATH% (
    %PYTHON_PATH% -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
) else (
    python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
)

pause
