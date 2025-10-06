#!/bin/bash
# Morning Trading Launcher Script
# This script runs the morning trading analysis

# Change to the Stock AI directory
cd /home/mimik/Stock_AI

# Activate virtual environment
source .venv/bin/activate

# Run the morning trader
python3 morning_trader.py

# Log the completion
echo "$(date): Morning trading analysis completed" >> morning_trading.log