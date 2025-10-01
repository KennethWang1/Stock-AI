import json
import os
from datetime import datetime, date
from typing import Dict, Optional

TRADING_LOG_FILE = './trading_sessions.json'

def get_today_session_id() -> str:
    return date.today().strftime('%Y-%m-%d')

def has_traded_today() -> bool:
    today_id = get_today_session_id()
    
    try:
        with open(TRADING_LOG_FILE, 'r') as f:
            sessions = json.load(f)
        return today_id in sessions
    except FileNotFoundError:
        return False

def record_trading_session(action: float, reward: float, portfolio_value: float) -> None:
    today_id = get_today_session_id()
    
    session_data = {
        'date': today_id,
        'timestamp': datetime.now().isoformat(),
        'action': float(action),
        'reward': float(reward),
        'portfolio_value': float(portfolio_value)
    }
    
    try:
        with open(TRADING_LOG_FILE, 'r') as f:
            sessions = json.load(f)
    except FileNotFoundError:
        sessions = {}
    
    sessions[today_id] = session_data
    
    with open(TRADING_LOG_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def get_trading_history() -> Dict:
    try:
        with open(TRADING_LOG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def clear_trading_history() -> None:
    if os.path.exists(TRADING_LOG_FILE):
        os.remove(TRADING_LOG_FILE)
        print(f"Trading history cleared: {TRADING_LOG_FILE}")
    else:
        print(f"No trading history file found: {TRADING_LOG_FILE}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear":
            clear_trading_history()
        elif command == "history":
            history = get_trading_history()
            if history:
                print("Trading History:")
                for session_id, data in sorted(history.items()):
                    print(f"   {session_id}: Action={data['action']:+.3f}, Reward={data['reward']:+.2f}%, Value=${data['portfolio_value']:.2f}")
            else:
                print("No trading history found")
        elif command == "today":
            if has_traded_today():
                print(f"Already traded today ({get_today_session_id()})")
            else:
                print(f"No trade recorded for today ({get_today_session_id()})")
    else:
        print("Usage:")
        print("  python trading_tracker.py clear    # Clear trading history")
        print("  python trading_tracker.py history  # Show trading history") 
        print("  python trading_tracker.py today    # Check if traded today")
