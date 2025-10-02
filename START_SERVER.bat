@echo off
echo ============================================================
echo Starting Digit Recognition Server with Feedback Features
echo ============================================================
echo.
cd /d "%~dp0"
python predict.py
pause