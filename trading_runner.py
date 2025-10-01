import sys
import os
import traceback
from pathlib import Path
from datetime import datetime

def log_message(message: str, log_file: Path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

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
