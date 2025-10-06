@echo off
REM Morning Trading Launcher Script for Windows
REM This script runs the morning trading analysis

REM Change to the Stock AI directory
cd /d "C:\Users\mimik\Documents\VS code\Stock AI"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run the morning trader
python morning_trader.py

REM Log the completion
echo %date% %time%: Morning trading analysis completed >> morning_trading.log