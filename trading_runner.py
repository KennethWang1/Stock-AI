import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime, date, timedelta
from config import STOCK_SYMBOL

def log_message(message: str, log_file: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def check_trading_sessions():
    try:
        with open("trading_sessions.json", 'r') as f:
            sessions = json.load(f)
        
        today = date.today().strftime('%Y-%m-%d')
        yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        return {
            "total_sessions": len(sessions),
            "today_traded": today in sessions,
            "yesterday_traded": yesterday in sessions,
            "latest_session": max(sessions.keys()) if sessions else None,
            "sessions": sessions
        }
    except FileNotFoundError:
        return {"error": "No trading sessions file found"}
    except Exception as e:
        return {"error": f"Error reading trading sessions: {e}"}

def check_experience_buffer():
    try:
        from trading.buffer import ExperienceReplayBuffer
        
        buffer = ExperienceReplayBuffer()
        buffer.load("./rl_experience_buffer.pkl")
        
        return {
            "buffer_exists": True,
            "experience_count": buffer.size(),
            "buffer_capacity": buffer.capacity,
            "last_reward": buffer.buffer[-1]['reward'] if buffer.size() > 0 else None,
            "last_action": buffer.buffer[-1]['action'] if buffer.size() > 0 else None
        }
    except FileNotFoundError:
        return {"buffer_exists": False, "error": "No experience buffer found"}
    except Exception as e:
        return {"error": f"Error reading experience buffer: {e}"}

def check_portfolio_updates():
    try:
        with open("stock.json", 'r') as f:
            portfolio = json.load(f)
        
        timestamp_str = portfolio.get('timestamp', '')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
            hours_since_update = (datetime.now() - timestamp).total_seconds() / 3600
        else:
            hours_since_update = None
        
        return {
            "portfolio_exists": True,
            "last_update": timestamp_str,
            "hours_since_update": hours_since_update,
            "portfolio_value": portfolio.get('total_value', 0),
            "cash": portfolio.get('cash', 0),
            "shares": portfolio.get('shares', 0)
        }
    except FileNotFoundError:
        return {"portfolio_exists": False, "error": "No portfolio file found"}
    except Exception as e:
        return {"error": f"Error reading portfolio: {e}"}

def main():
    script_dir = Path(__file__).parent.absolute()
    
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    success_log = logs_dir / "trading_success.log"
    error_log = logs_dir / "trading_errors.log"
    
    os.chdir(script_dir)
    
    log_message("Trading Runner Started", success_log)
    log_message(f"Working Directory: {os.getcwd()}", success_log)
    log_message(f"Python Version: {sys.version}", success_log)
    log_message("-" * 50, success_log)
    
    try:
        from main import main as trading_main
        trading_main()
            
        log_message("Trading execution completed successfully", success_log)
        log_message(f"Check buffer status with: python buffer_manager.py info", success_log)
        log_message(f"Check trading history with: python trading_tracker.py history", success_log)
        log_message("=" * 50, success_log)
        
        sys.exit(0)
        
    except Exception as e:
        error_msg = f"Trading execution failed: {e}"
        log_message(error_msg, error_log)
        log_message(error_msg, success_log)
        
        full_traceback = traceback.format_exc()
        log_message(full_traceback, error_log)
        log_message("=" * 50, error_log)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
