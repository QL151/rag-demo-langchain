@echo off
rem ---------------------------------------------------------------
rem  Do NOT put Chinese text in this file.
rem  cmd parses .cmd with the system OEM codepage (GBK on zh-CN),
rem  so UTF-8 Chinese here would be mangled and break the commands.
rem ---------------------------------------------------------------
chcp 65001 >nul
cd /d "%~dp0"
title RAG Knowledge Base Demo

echo ========================================
echo   rag-demo-langchain
echo   http://127.0.0.1:8000   Ctrl+C to stop
echo ========================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py 8000
) else (
    echo [warn] .venv not found, falling back to system python
    python run.py 8000
)

pause
