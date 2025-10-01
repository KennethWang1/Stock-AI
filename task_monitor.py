import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

def check_task_logs() -> Dict:
    logs_dir = Path("./logs")
    success_log = logs_dir / "trading_success.log"
    error_log = logs_dir / "trading_errors.log"
    
    result = {
        "success_log_exists": success_log.exists(),
        "error_log_exists": error_log.exists(),
        "last_success": None,
        "last_error": None,
        "recent_executions": []
    }
    
    if success_log.exists():
        try:
            with open(success_log, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if "Trading execution completed successfully" in line:
                    timestamp = line.split(']')[0][1:]
                    result["last_success"] = timestamp
                    break
            
            execution_lines = [line for line in lines if "Trading Runner Started" in line]
            result["recent_executions"] = [line.strip() for line in execution_lines[-10:]]
            
        except Exception as e:
            result["success_log_error"] = str(e)
    
    if error_log.exists():
        try:
            with open(error_log, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if "Trading execution failed" in line:
                    timestamp = line.split(']')[0][1:]
                    result["last_error"] = timestamp
                    break
                    
        except Exception as e:
            result["error_log_error"] = str(e)
    
    return result

def check_trading_sessions() -> Dict:
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

def check_experience_buffer() -> Dict:
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

def check_portfolio_updates() -> Dict:
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

def print_status_report():
    print("TRADING TASK STATUS REPORT")
    print("=" * 50)
    
    print("\nLOG FILES:")
    log_status = check_task_logs()
    
    if log_status["success_log_exists"]:
        print("Success log exists")
        if log_status["last_success"]:
            print(f"   Last successful run: {log_status['last_success']}")
        print(f"   Total executions logged: {len(log_status['recent_executions'])}")
    else:
        print("No success log found")
    
    if log_status["error_log_exists"]:
        print("Error log exists")
        if log_status["last_error"]:
            print(f"   Last error: {log_status['last_error']}")
    else:
        print("No error log (good!)")
    
    print("\nTRADING SESSIONS:")
    session_status = check_trading_sessions()
    
    if "error" not in session_status:
        print(f"Trading sessions tracked: {session_status['total_sessions']}")
        print(f"Today traded: {'Yes' if session_status['today_traded'] else 'No'}")
        print(f"Latest session: {session_status['latest_session']}")
    else:
        print(f"{session_status['error']}")
    
    print("\nEXPERIENCE BUFFER:")
    buffer_status = check_experience_buffer()
    
    if "error" not in buffer_status:
        print(f"Buffer exists with {buffer_status['experience_count']} experiences")
        if buffer_status['last_reward'] is not None:
            print(f"Last reward: {buffer_status['last_reward']:.2f}%")
            print(f"Last action: {buffer_status['last_action']:+.3f}")
    else:
        print(f"{buffer_status['error']}")
    
    print("\nPORTFOLIO:")
    portfolio_status = check_portfolio_updates()
    
    if "error" not in portfolio_status:
        print(f"Portfolio value: ${portfolio_status['portfolio_value']:.2f}")
        if portfolio_status['hours_since_update'] is not None:
            hours_old = portfolio_status['hours_since_update']
            print(f"Last update: {hours_old:.1f} hours ago", end="")
            if hours_old > 25:
                print(" (stale - over 25 hours)")
            else:
                print(" (fresh)")
        print(f"Cash: ${portfolio_status['cash']:.2f}")
        print(f"Shares: {portfolio_status['shares']}")
    else:
        print(f"{portfolio_status['error']}")
    
    print("\nOVERALL STATUS:")
    
    success_indicators = 0
    total_checks = 4
    
    if log_status["success_log_exists"] and log_status["last_success"]:
        success_indicators += 1
    if "error" not in session_status and session_status["total_sessions"] > 0:
        success_indicators += 1
    if "error" not in buffer_status and buffer_status["experience_count"] > 0:
        success_indicators += 1
    portfolio_is_healthy = ("error" not in portfolio_status and 
                           portfolio_status.get('hours_since_update') is not None and 
                           portfolio_status['hours_since_update'] < 25)
    if portfolio_is_healthy:
        success_indicators += 1
    
    if success_indicators == total_checks:
        print("ALL SYSTEMS OPERATIONAL - Task is running successfully!")
    elif success_indicators >= total_checks - 1:
        print("MOSTLY OPERATIONAL - Minor issues detected")
    else:
        print("ISSUES DETECTED - Task may not be running properly")
    
    print(f"Health Score: {success_indicators}/{total_checks}")

def check_windows_task_scheduler():
    print("\nWINDOWS TASK SCHEDULER CHECK:")
    print("1. Open Task Scheduler (taskschd.msc)")
    print("2. Navigate to your 'Trading Script' task")
    print("3. Right-click → Properties → History tab")
    print("4. Look for recent events:")
    print("   - Event ID 100: Task started")
    print("   - Event ID 200: Task completed successfully") 
    print("   - Event ID 201: Task completed with errors")
    print("5. Check 'Last Run Result' in main view (should be 0x0 for success)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "logs":
            print(json.dumps(check_task_logs(), indent=2))
        elif command == "sessions":
            print(json.dumps(check_trading_sessions(), indent=2))
        elif command == "buffer":
            print(json.dumps(check_experience_buffer(), indent=2))
        elif command == "portfolio":
            print(json.dumps(check_portfolio_updates(), indent=2))
        elif command == "scheduler":
            check_windows_task_scheduler()
        else:
            print("Usage:")
            print("  python task_monitor.py          # Full status report")
            print("  python task_monitor.py logs     # Check log files")
            print("  python task_monitor.py sessions # Check trading sessions")
            print("  python task_monitor.py buffer   # Check experience buffer")
            print("  python task_monitor.py portfolio # Check portfolio updates")
            print("  python task_monitor.py scheduler # Windows Task Scheduler help")
    else:
        print_status_report()
        print("\n" + "="*50)
        check_windows_task_scheduler()