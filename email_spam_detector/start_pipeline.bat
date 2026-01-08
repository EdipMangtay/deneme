@echo off
echo ======================================================================
echo Email Spam Detection - Complete Pipeline
echo ======================================================================
echo.
echo Starting pipeline...
echo.

cd /d "%~dp0"
python run_complete_pipeline.py

pause

