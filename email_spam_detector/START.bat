@echo off
echo ========================================
echo Email Spam Detection System
echo ========================================
echo.
echo Starting pipeline...
echo.
cd /d %~dp0
python pipeline.py
pause


