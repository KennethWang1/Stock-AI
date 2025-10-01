import os
from config import EXPERIENCE_BUFFER_FILE
from trading.buffer import ExperienceReplayBuffer


def clear_experience_buffer():
    try:
        if os.path.exists(EXPERIENCE_BUFFER_FILE):
            os.remove(EXPERIENCE_BUFFER_FILE)
            print(f"Experience buffer cleared: {EXPERIENCE_BUFFER_FILE}")
        else:
            print(f"No buffer file found: {EXPERIENCE_BUFFER_FILE}")
    except Exception as e:
        print(f"Error clearing buffer: {e}")


def get_buffer_info():
    try:
        if os.path.exists(EXPERIENCE_BUFFER_FILE):
            buffer = ExperienceReplayBuffer()
            buffer.load(EXPERIENCE_BUFFER_FILE)
            print(f"Buffer Info:")
            print(f"   File: {EXPERIENCE_BUFFER_FILE}")
            print(f"   Experiences: {buffer.size()}")
            print(f"   Capacity: {buffer.capacity}")
            
            if buffer.size() > 0:
                last_exp = buffer.buffer[-1]
                print(f"   Last reward: {last_exp['reward']:.2f}%")
                print(f"   Last action: {last_exp['action']:+.3f}")
        else:
            print(f"No buffer file found: {EXPERIENCE_BUFFER_FILE}")
    except Exception as e:
        print(f"Error reading buffer: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "clear":
            clear_experience_buffer()
        elif command == "info":
            get_buffer_info()
        else:
            print("Usage:")
            print("  python buffer_manager.py clear  # Clear the buffer")
            print("  python buffer_manager.py info   # Show buffer info")
    else:
        get_buffer_info()
